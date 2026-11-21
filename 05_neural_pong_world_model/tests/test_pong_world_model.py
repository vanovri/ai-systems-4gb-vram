"""Comprehensive pytest suite for Project #5: Neural Pong World Model."""

from __future__ import annotations

import ast
import base64
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from src.collect_data import collect_dataset
from src.dataset import PongSequenceDataset, compute_valid_indices
from src.model import (
    MODEL_CONFIGS,
    NeuralPongWorldModel,
    ste_binarize,
    world_model_loss,
)
from src.neural_runner import NeuralPongSession, autoregressive_rollout
from src.play_web import create_app
from src.pong_env import (
    ACTION_DOWN,
    ACTION_NONE,
    ACTION_UP,
    FRAME_H,
    FRAME_W,
    LEFT_PADDLE_X,
    PADDLE_H,
    PADDLE_MAX_Y,
    PADDLE_MIN_Y,
    PADDLE_W,
    PLAY_BOTTOM,
    PLAY_TOP,
    RIGHT_PADDLE_X,
    PongEnv,
    PongState,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _decode_frame_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    assert len(raw) == FRAME_H * FRAME_W
    return np.frombuffer(raw, dtype=np.uint8).reshape(FRAME_H, FRAME_W)


# ---------------------------------------------------------------------------
# 1-4. Deterministic Headless Pong Environment Physics & Rendering
# ---------------------------------------------------------------------------


def test_env_deterministic_rollout() -> None:
    """Identical seed + action sequence MUST yield bit-identical frames & states."""
    actions = [ACTION_UP, ACTION_UP, ACTION_NONE, ACTION_DOWN, ACTION_DOWN, ACTION_NONE] * 10
    env_a = PongEnv(seed=1234, ai_jitter=0.05)
    env_b = PongEnv(seed=1234, ai_jitter=0.05)

    fa = env_a.reset(seed=1234, randomize_positions=True)
    fb = env_b.reset(seed=1234, randomize_positions=True)
    assert np.array_equal(fa, fb)

    for act in actions:
        fa, ra, da, _ = env_a.step(act)
        fb, rb, db, _ = env_b.step(act)
        assert np.array_equal(fa, fb)
        assert ra == rb and da == db
        assert env_a.get_state() == env_b.get_state()


def test_env_frame_format_and_geometry() -> None:
    """Frames must be 84x84 uint8 strictly in {0, 255} with intact walls, paddles, ball."""
    env = PongEnv(seed=7, ai_jitter=0.0)
    frame = env.reset(randomize_positions=False)

    assert frame.shape == (84, 84)
    assert frame.dtype == np.uint8
    assert set(np.unique(frame).tolist()).issubset({0, 255})
    # Top and bottom walls
    assert np.all(frame[:PLAY_TOP, :] == 255)
    assert np.all(frame[PLAY_BOTTOM:, :] == 255)
    # Left & right paddles present at y=35..49
    assert np.all(frame[35 : 35 + PADDLE_H, LEFT_PADDLE_X : LEFT_PADDLE_X + PADDLE_W] == 255)
    assert np.all(frame[35 : 35 + PADDLE_H, RIGHT_PADDLE_X : RIGHT_PADDLE_X + PADDLE_W] == 255)
    # 3x3 ball at (40, 40)
    assert np.all(frame[40:43, 40:43] == 255)


def test_env_player_paddle_controls_and_clamping() -> None:
    """UP decreases right_y, DOWN increases right_y, NONE holds steady, walls clamp cleanly."""
    env = PongEnv(seed=0, ai_jitter=0.0)
    env.reset(randomize_positions=False)
    y0 = env.state.right_y

    env.step(ACTION_NONE)
    assert env.state.right_y == y0

    env.step(ACTION_UP)
    assert env.state.right_y < y0
    y_up = env.state.right_y

    env.step(ACTION_DOWN)
    assert env.state.right_y == y0

    # Clamp at top wall
    for _ in range(50):
        env.step(ACTION_UP)
    assert env.state.right_y == PADDLE_MIN_Y
    assert y_up < y0

    # Clamp at bottom wall
    for _ in range(60):
        env.step(ACTION_DOWN)
    assert env.state.right_y == PADDLE_MAX_Y


def test_env_ball_wall_bounce_paddle_bounce_and_serve() -> None:
    """Verify wall bounce, right-paddle reflection, and deterministic center serve after miss."""
    env = PongEnv(seed=0, ai_jitter=0.0)

    # Top wall bounce
    env.reset(state=PongState(left_y=35, right_y=35, ball_x=40, ball_y=3, ball_vx=2, ball_vy=-2))
    _, _, _, info = env.step(ACTION_NONE)
    assert info["event"] == "wall_bounce"
    assert env.state.ball_vy > 0

    # Right paddle bounce
    env.reset(state=PongState(left_y=35, right_y=34, ball_x=72, ball_y=40, ball_vx=2, ball_vy=0))
    _, _, _, info = env.step(ACTION_NONE)
    assert info["event"] == "right_paddle_bounce"
    assert env.state.ball_vx == -2

    # Miss on right side -> left scores and ball serves cleanly from center (40, 40)
    env.reset(state=PongState(left_y=35, right_y=5, ball_x=80, ball_y=50, ball_vx=2, ball_vy=0))
    _, reward, _, info = env.step(ACTION_NONE)
    assert info["event"] == "left_score"
    assert reward == -1.0
    assert (env.state.ball_x, env.state.ball_y) == (40, 40)


# ---------------------------------------------------------------------------
# 5-6. Memory-Mapped Dataset Integrity & Zero Cross-Episode Bleed
# ---------------------------------------------------------------------------


def test_dataset_memmap_zero_cross_episode_bleed(tmp_path: Path) -> None:
    """4-frame input stacks and target frames MUST NEVER cross episode boundaries."""
    data_dir = tmp_path / "mini_pong"
    meta = collect_dataset(
        out_dir=data_dir,
        num_frames=320,
        seed=99,
        ai_jitter=0.0,
        max_points=1,
        max_steps=40,
    )
    assert meta["num_episodes"] >= 5

    ep_starts = np.load(data_dir / "episode_starts.npy")
    ds = PongSequenceDataset(data_dir=data_dir, stack_size=4, rollout_steps=1, augment_noise=False)
    assert len(ds) > 0

    # Every valid anchor t MUST have episode_starts[t-2 .. t+1] all False!
    for t in ds.indices:
        assert not np.any(ep_starts[t - 2 : t + 2]), f"Cross-episode bleed at anchor t={t}!"

    # Check returned tensor shapes & value ranges
    x, a, y = ds[0]
    assert x.shape == (4, 84, 84)
    assert a.ndim == 0 and int(a.item()) in (0, 1, 2)
    assert y.shape == (1, 84, 84)
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_dataset_multistep_rollout_bounds() -> None:
    """Explicit synthetic episode boundary test for compute_valid_indices with K=4."""
    # Episodes start at 0, 10, 25 (total 35 frames)
    ep_starts = np.zeros(35, dtype=bool)
    ep_starts[[0, 10, 25]] = True

    idx_k1 = compute_valid_indices(ep_starts, stack_size=4, rollout_steps=1)
    # Ep1 [0..9]: valid t = 3..8
    # Ep2 [10..24]: valid t = 13..23
    # Ep3 [25..34]: valid t = 28..33
    assert set(idx_k1.tolist()) == (set(range(3, 9)) | set(range(13, 24)) | set(range(28, 34)))

    idx_k4 = compute_valid_indices(ep_starts, stack_size=4, rollout_steps=4)
    # Ep1 [0..9]: t+4 <= 9 -> t in 3..5
    # Ep2 [10..24]: t+4 <= 24 -> t in 13..20
    # Ep3 [25..34]: t+4 <= 34 -> t in 28..30
    assert set(idx_k4.tolist()) == (set(range(3, 6)) | set(range(13, 21)) | set(range(28, 31)))


# ---------------------------------------------------------------------------
# 7-12. Model Architecture, Slide #6 Bugfixes, Action Conditioning & Rollout
# ---------------------------------------------------------------------------


def test_model_small_and_medium_shapes_and_param_counts() -> None:
    """Both small and medium configs must output finite raw logits (B, 1, 84, 84)."""
    x = torch.randint(0, 2, (2, 4, 84, 84), dtype=torch.float32)
    a = torch.tensor([0, 2], dtype=torch.long)

    for size_name in ("small", "medium"):
        assert size_name in MODEL_CONFIGS
        model = NeuralPongWorldModel(size_name)
        n_params = model.count_parameters()
        if size_name == "small":
            assert 800_000 <= n_params <= 4_000_000
        else:
            assert 5_000_000 <= n_params <= 12_000_000

        logits = model(x, a)
        assert logits.shape == (2, 1, 84, 84)
        assert torch.isfinite(logits).all()


def test_loss_gradient_flow_fixes_slide6_threshold_bug() -> None:
    """Verify gradients flow cleanly through ALL model params (no Slide #6 >0.5 cut!)."""
    torch.manual_seed(0)
    model = NeuralPongWorldModel("small")
    x = torch.randint(0, 2, (2, 4, 84, 84), dtype=torch.float32)
    a = torch.tensor([1, 2], dtype=torch.long)
    y = torch.zeros((2, 1, 84, 84), dtype=torch.float32)
    y[:, :, 35:49, 76:79] = 1.0  # right paddle
    y[:, :, 40:43, 40:43] = 1.0  # 3x3 ball

    # 1) Regression proof: Slide #6's `(sigmoid(logits) > 0.5)` breaks autograd completely
    raw_test = torch.zeros((1, 1, 4, 4), requires_grad=True)
    broken_frame = (torch.sigmoid(raw_test) > 0.5).float()
    assert not broken_frame.requires_grad

    # 2) Our world_model_loss computes directly on raw logits -> full gradient flow!
    logits = model(x, a)
    loss, metrics = world_model_loss(logits, y, last_frame=x[:, -1:])
    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert metrics["raw_bce"] > 0.0

    model.zero_grad()
    loss.backward()

    grads_present = 0
    grads_nonzero = 0
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Missing gradient for parameter: {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for: {name}"
        grads_present += 1
        if p.grad.abs().sum().item() > 0:
            grads_nonzero += 1

    assert grads_present > 0
    assert grads_nonzero >= int(grads_present * 0.9)


def test_loss_prevents_all_black_collapse() -> None:
    """Predicting all-black logits must incur much higher loss than matching sparse objects."""
    y = torch.zeros((2, 1, 84, 84), dtype=torch.float32)
    y[:, :, :2, :] = 1.0
    y[:, :, 82:, :] = 1.0
    y[:, :, 35:49, 5:8] = 1.0
    y[:, :, 35:49, 76:79] = 1.0
    y[:, :, 40:43, 40:43] = 1.0  # 3x3 ball

    all_black_logits = torch.full_like(y, -8.0)
    accurate_logits = torch.where(y > 0.5, torch.full_like(y, 6.0), torch.full_like(y, -6.0))

    loss_black, _ = world_model_loss(all_black_logits, y, pos_weight=8.0, dice_weight=0.75)
    loss_good, _ = world_model_loss(accurate_logits, y, pos_weight=8.0, dice_weight=0.75)

    assert loss_black.item() > 1.5, "All-black prediction must be strongly penalized!"
    assert loss_good.item() < 0.05
    assert loss_black.item() > loss_good.item() * 30.0


def test_ste_binarize_forward_binary_backward_gradient() -> None:
    """STE outputs strict binary {0, 1} forward while keeping non-zero gradients backward."""
    logits = torch.tensor([[[[-2.0, -0.1, 0.2, 3.0]]]], requires_grad=True)
    y = ste_binarize(logits)
    assert set(y.detach().unique().tolist()).issubset({0.0, 1.0})
    y.sum().backward()
    assert logits.grad is not None
    assert (logits.grad > 0).all()


def test_action_conditioning_up_vs_down_sensitivity() -> None:
    """Given identical 4 input frames, action UP vs DOWN MUST produce different logits."""
    torch.manual_seed(42)
    model = NeuralPongWorldModel("small").eval()
    x = torch.randint(0, 2, (1, 4, 84, 84), dtype=torch.float32)

    with torch.no_grad():
        logits_none = model(x, torch.tensor([ACTION_NONE]))
        logits_up = model(x, torch.tensor([ACTION_UP]))
        logits_down = model(x, torch.tensor([ACTION_DOWN]))

    diff_up_down = (logits_up - logits_down).abs().max().item()
    diff_up_none = (logits_up - logits_none).abs().max().item()
    assert diff_up_down > 1e-4, "Model logits failed to respond to UP vs DOWN action conditioning!"
    assert diff_up_none > 1e-4, "Model logits failed to respond to UP vs NONE action conditioning!"


def test_autoregressive_rollout_shape_and_finiteness() -> None:
    """Pure neural multi-step rollout runs 15 steps with ZERO game engine & finite outputs."""
    model = NeuralPongWorldModel("small").eval()
    init_4 = torch.randint(0, 2, (2, 4, 84, 84), dtype=torch.float32)
    acts = torch.randint(0, 3, (2, 15), dtype=torch.long)

    pred_frames, pred_probs = autoregressive_rollout(
        model=model,
        initial_frames=init_4,
        actions=acts,
        use_amp=False,
    )
    assert pred_frames.shape == (2, 15, 84, 84)
    assert pred_probs.shape == (2, 15, 84, 84)
    assert torch.isfinite(pred_frames).all()
    assert torch.isfinite(pred_probs).all()
    assert set(pred_frames.unique().tolist()).issubset({0.0, 1.0})
    assert float(pred_probs.min()) >= 0.0 and float(pred_probs.max()) <= 1.0


# ---------------------------------------------------------------------------
# 13-16. Zero-Game-Code Runtime Isolation & FastAPI Playable Web App
# ---------------------------------------------------------------------------


def test_neural_runner_zero_pong_env_import() -> None:
    """`src/neural_runner.py` and top-level `src/play_web.py` MUST NOT import `pong_env`."""
    for rel_path in ("src/neural_runner.py", "src/play_web.py"):
        source = (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        # Check all top-level imports
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "pong_env" not in alias.name, f"Forbidden top-level import in {rel_path}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "pong_env" not in mod, f"Forbidden top-level from-import in {rel_path}"

        # For neural_runner.py, forbid pong_env ANYWHERE in the entire AST!
        if rel_path == "src/neural_runner.py":
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "pong_env" not in alias.name
                elif isinstance(node, ast.ImportFrom):
                    assert "pong_env" not in (node.module or "")


@pytest.fixture()
def test_client() -> TestClient:
    """Create isolated FastAPI TestClient on CPU for fast deterministic endpoint testing."""
    model = NeuralPongWorldModel("small").eval()
    seeds = np.zeros((2, 4, 84, 84), dtype=np.uint8)
    seeds[:, :, :2, :] = 255
    seeds[:, :, 82:, :] = 255
    seeds[:, :, 35:49, 5:8] = 255
    seeds[:, :, 35:49, 76:79] = 255
    for k in range(4):
        seeds[:, k, 40:43, 36 + 2 * k : 39 + 2 * k] = 255
    states = [
        PongState(left_y=35, right_y=35, ball_x=42, ball_y=40, ball_vx=2, ball_vy=1).to_dict(),
        PongState(left_y=25, right_y=45, ball_x=42, ball_y=40, ball_vx=-2, ball_vy=-1).to_dict(),
    ]
    app = create_app(model=model, seed_frames=seeds, seed_states=states)
    return TestClient(app)


def test_fastapi_health_and_root_html(test_client: TestClient) -> None:
    """GET /health and GET / return valid metadata and playable HTML5 Canvas UI."""
    r_health = test_client.get("/health")
    assert r_health.status_code == 200
    body = r_health.json()
    assert body["status"] == "ok"
    assert body["neural_only_ready"] is True
    assert body["param_count"] > 0

    r_root = test_client.get("/")
    assert r_root.status_code == 200
    assert "text/html" in r_root.headers["content-type"]
    assert "<canvas id=\"neuralCanvas\"" in r_root.text
    assert "NEURAL-ONLY MODE" in r_root.text


def test_fastapi_rest_reset_and_step_neural_only(test_client: TestClient) -> None:
    """POST /api/reset + /api/step works purely in Neural-Only mode (no engine_b64)."""
    r_reset = test_client.post("/api/reset", json={"mode": "neural", "seed_idx": 0})
    assert r_reset.status_code == 200
    d0 = r_reset.json()
    sid = d0["session_id"]
    assert d0["mode"] == "neural"
    assert "engine_b64" not in d0
    f0 = _decode_frame_b64(d0["neural_b64"])
    assert f0.shape == (84, 84)

    for step_idx, act in enumerate([ACTION_UP, ACTION_DOWN, ACTION_NONE], start=1):
        r_step = test_client.post("/api/step", json={"session_id": sid, "action": act})
        assert r_step.status_code == 200
        ds = r_step.json()
        assert ds["step_count"] == step_idx
        assert "engine_b64" not in ds
        fs = _decode_frame_b64(ds["neural_b64"])
        assert set(np.unique(fs).tolist()).issubset({0, 255})


def test_fastapi_websocket_neural_and_compare_modes(test_client: TestClient) -> None:
    """WebSocket /ws/play streams crisp frames in both neural-only and split-screen modes."""
    with test_client.websocket_connect("/ws/play") as ws:
        ws.send_json({"type": "reset", "mode": "neural", "seed_idx": 0})
        m0 = ws.receive_json()
        assert m0["mode"] == "neural"
        assert m0["step_count"] == 0
        assert "engine_b64" not in m0
        assert _decode_frame_b64(m0["neural_b64"]).shape == (84, 84)

        ws.send_json({"type": "step", "action": ACTION_UP})
        m1 = ws.receive_json()
        assert m1["step_count"] == 1
        assert "engine_b64" not in m1

        # Toggle Split-Screen Comparison Mode
        ws.send_json({"type": "reset", "mode": "compare", "seed_idx": 1})
        mc0 = ws.receive_json()
        assert mc0["mode"] == "compare"
        assert "engine_b64" in mc0
        assert _decode_frame_b64(mc0["engine_b64"]).shape == (84, 84)

        ws.send_json({"type": "step", "action": ACTION_DOWN})
        mc1 = ws.receive_json()
        assert mc1["step_count"] == 1
        assert "engine_b64" in mc1
        assert _decode_frame_b64(mc1["neural_b64"]).shape == (84, 84)
        assert _decode_frame_b64(mc1["engine_b64"]).shape == (84, 84)


def test_trained_checkpoint_end_to_end_neural_play_and_responsiveness() -> None:
    """Verify the trained `artifacts/neural_pong_best.pt` checkpoint loads into FastAPI
    in Neural-Only mode and visibly moves the right paddle UP vs DOWN with ZERO game engine."""
    ckpt = PROJECT_ROOT / "artifacts" / "neural_pong_best.pt"
    seeds = PROJECT_ROOT / "artifacts" / "seed_frames.npy"
    if not ckpt.exists() or not seeds.exists():
        pytest.skip("Trained checkpoint or seed pack not present yet.")

    app = create_app(checkpoint_path=ckpt, seed_frames_path=seeds, device="cpu")
    client = TestClient(app)

    rh = client.get("/health")
    assert rh.status_code == 200
    h = rh.json()
    assert h["checkpoint_loaded"] is True
    assert h["neural_only_ready"] is True
    assert h["param_count"] == 1_575_441

    # Roll 5 UP steps vs 5 DOWN steps from the exact same seed -> right paddle centroid MUST diverge!
    def _run_5(action: int) -> float:
        r0 = client.post("/api/reset", json={"mode": "neural", "seed_idx": 0}).json()
        sid = r0["session_id"]
        frame = _decode_frame_b64(r0["neural_b64"])
        for _ in range(5):
            rs = client.post("/api/step", json={"session_id": sid, "action": action}).json()
            assert "engine_b64" not in rs
            frame = _decode_frame_b64(rs["neural_b64"])
        ys = np.where(frame[PLAY_TOP:PLAY_BOTTOM, RIGHT_PADDLE_X : RIGHT_PADDLE_X + PADDLE_W] > 127)[0]
        assert ys.size >= 10, "Right paddle disappeared during pure neural rollout!"
        return float(ys.mean() + PLAY_TOP)

    cy_up = _run_5(ACTION_UP)
    cy_none = _run_5(ACTION_NONE)
    cy_down = _run_5(ACTION_DOWN)
    assert cy_up < cy_none - 4.0, f"Expected UP ({cy_up:.2f}) significantly above NONE ({cy_none:.2f})"
    assert cy_down > cy_none + 4.0, f"Expected DOWN ({cy_down:.2f}) significantly below NONE ({cy_none:.2f})"

