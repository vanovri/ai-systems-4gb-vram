"""Playable Neural Pong Web App (FastAPI + WebSocket + single-file HTML5 Canvas UI).

Run locally:
    uvicorn src.play_web:app --host 127.0.0.1 --port 8000

Architecture Guarantee:
- Top-level module NEVER imports `pong_env.py`!
- In default **Neural-Only Mode**, every frame after the 4 initial seed frames is
  generated 100% autoregressively by `artifacts/neural_pong_best.pt` via `NeuralPongSession`.
- Only when the user explicitly activates **Split-Screen Comparison Mode** does a lazy
  helper import `PongEnv` to render the ground-truth physics engine side-by-side.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.model import NeuralPongWorldModel
from src.neural_runner import NeuralPongSession, load_world_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = PROJECT_ROOT / "artifacts" / "neural_pong_best.pt"
DEFAULT_SEEDS = PROJECT_ROOT / "artifacts" / "seed_frames.npy"
DEFAULT_SEED_STATES = PROJECT_ROOT / "artifacts" / "seed_states.json"


def _encode_frame_b64(frame_u8: np.ndarray) -> str:
    """Encode an 84x84 uint8 frame (7056 bytes) as compact base64 for fast Canvas blitting."""
    arr = np.ascontiguousarray(frame_u8, dtype=np.uint8)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _create_compare_env(seed_state: dict[str, Any] | None) -> Any:
    """Lazily import PongEnv ONLY when user explicitly chooses Split-Screen Comparison Mode."""
    from src.pong_env import PongEnv, PongState

    env = PongEnv(seed=0, ai_jitter=0.0, max_points=99, max_steps=100_000)
    if seed_state is not None:
        env.reset(state=PongState.from_dict(seed_state))
    else:
        env.reset(randomize_positions=False)
    return env


class ResetRequest(BaseModel):
    mode: Literal["neural", "compare"] = "neural"
    seed_idx: int = Field(default=0, ge=0)


class StepRequest(BaseModel):
    session_id: str
    action: int = Field(default=0, ge=0, le=2)


@dataclass
class ActiveSession:
    mode: Literal["neural", "compare"]
    neural: NeuralPongSession
    compare_env: Any | None = None
    seed_idx: int = 0


HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Neural Pong World Model — Zero Game Code</title>
<style>
  :root {
    --bg: #070b14;
    --panel: #0f172a;
    --border: #1e293b;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --neural: #22c55e;
    --engine: #f59e0b;
    --danger: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: radial-gradient(circle at top, #111c33 0%, var(--bg) 60%);
    color: var(--text);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px 16px 32px;
  }
  .header {
    width: 100%;
    max-width: 920px;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }
  .title-block h1 {
    font-size: 1.35rem;
    letter-spacing: 0.04em;
    color: #f8fafc;
  }
  .title-block p {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 4px;
  }
  .badge {
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    border: 1px solid;
  }
  .badge.neural {
    color: #86efac;
    background: rgba(34, 197, 94, 0.14);
    border-color: rgba(34, 197, 94, 0.5);
  }
  .badge.compare {
    color: #fcd34d;
    background: rgba(245, 158, 11, 0.14);
    border-color: rgba(245, 158, 11, 0.5);
  }
  .hud {
    width: 100%;
    max-width: 920px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    gap: 10px;
    margin-bottom: 16px;
  }
  .card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
  }
  .card .label {
    font-size: 0.68rem;
    text-transform: uppercase;
    color: var(--muted);
    letter-spacing: 0.08em;
  }
  .card .val {
    font-size: 1.05rem;
    font-weight: 700;
    margin-top: 4px;
    color: #f8fafc;
  }
  .stage {
    width: 100%;
    max-width: 920px;
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 20px;
    margin-bottom: 16px;
  }
  .viewport {
    background: #020617;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.45);
  }
  .viewport-title {
    width: 100%;
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    margin-bottom: 8px;
    color: var(--muted);
  }
  .viewport-title .dot {
    font-weight: 700;
  }
  .viewport-title .dot.neural { color: var(--neural); }
  .viewport-title .dot.engine { color: var(--engine); }
  canvas {
    width: 378px;
    height: 378px;
    image-rendering: pixelated;
    image-rendering: crisp-edges;
    background: #000;
    border: 2px solid #334155;
    border-radius: 6px;
  }
  .controls {
    width: 100%;
    max-width: 920px;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
  }
  .btn-group {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }
  button, select {
    background: #1e293b;
    color: var(--text);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 13px;
    font-family: inherit;
    font-size: 0.8rem;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }
  button:hover { background: #334155; border-color: var(--accent); }
  button.active {
    background: rgba(56, 189, 248, 0.2);
    border-color: var(--accent);
    color: #e0f2fe;
  }
  button.key-btn { min-width: 72px; font-weight: 700; }
  button.key-btn.pressed {
    background: rgba(34, 197, 94, 0.25);
    border-color: var(--neural);
    color: #bbf7d0;
  }
  .hint {
    width: 100%;
    max-width: 920px;
    margin-top: 12px;
    font-size: 0.76rem;
    color: var(--muted);
    text-align: center;
  }
  .hidden { display: none !important; }
</style>
</head>
<body>
  <div class="header">
    <div class="title-block">
      <h1>NEURAL PONG WORLD MODEL</h1>
      <p>Play Pong generated pixel-by-pixel by an Action-Conditioned U-Net CNN</p>
    </div>
    <div id="modeBadge" class="badge neural">NEURAL-ONLY MODE &bull; ZERO GAME CODE</div>
  </div>

  <div class="hud">
    <div class="card"><div class="label">Live FPS</div><div class="val" id="hudFps">--</div></div>
    <div class="card"><div class="label">Neural Latency</div><div class="val" id="hudMs">-- ms</div></div>
    <div class="card"><div class="label">Autoregressive Step</div><div class="val" id="hudStep">0</div></div>
    <div class="card"><div class="label">Active Key</div><div class="val" id="hudKey">NONE</div></div>
    <div class="card"><div class="label">Model Params</div><div class="val" id="hudParams">--</div></div>
    <div class="card"><div class="label">Device</div><div class="val" id="hudDevice">--</div></div>
  </div>

  <div class="stage">
    <div id="enginePanel" class="viewport hidden">
      <div class="viewport-title">
        <span class="dot engine">&#9679; REAL PONG ENGINE (GROUND TRUTH)</span>
        <span>84 &times; 84</span>
      </div>
      <canvas id="engineCanvas" width="84" height="84"></canvas>
    </div>

    <div class="viewport">
      <div class="viewport-title">
        <span class="dot neural">&#9679; PURE NEURAL WORLD MODEL</span>
        <span id="seedLabel">Seed #0</span>
      </div>
      <canvas id="neuralCanvas" width="84" height="84"></canvas>
    </div>
  </div>

  <div class="controls">
    <div class="btn-group">
      <button id="btnNeuralMode" class="active">Neural-Only</button>
      <button id="btnCompareMode">Split-Screen Compare</button>
      <button id="btnReset">Reset Seed (Next)</button>
      <button id="btnPause">Pause</button>
      <select id="fpsSelect" title="Target FPS">
        <option value="15">15 FPS</option>
        <option value="20" selected>20 FPS</option>
        <option value="30">30 FPS</option>
      </select>
    </div>
    <div class="btn-group">
      <button id="btnUp" class="key-btn">&uarr; UP (W)</button>
      <button id="btnDown" class="key-btn">&darr; DOWN (S)</button>
    </div>
  </div>

  <div class="hint">
    Control the <b>right paddle</b> with <b>ArrowUp / ArrowDown</b> or <b>W / S</b> (or hold buttons above).
    Press <b>R</b> to cycle a fresh 4-frame seed. After frame 0, Neural-Only mode runs 100% from neural weights!
  </div>

<script>
(() => {
  const neuralCanvas = document.getElementById("neuralCanvas");
  const neuralCtx = neuralCanvas.getContext("2d");
  const engineCanvas = document.getElementById("engineCanvas");
  const engineCtx = engineCanvas.getContext("2d");
  const enginePanel = document.getElementById("enginePanel");

  const hudFps = document.getElementById("hudFps");
  const hudMs = document.getElementById("hudMs");
  const hudStep = document.getElementById("hudStep");
  const hudKey = document.getElementById("hudKey");
  const hudParams = document.getElementById("hudParams");
  const hudDevice = document.getElementById("hudDevice");
  const modeBadge = document.getElementById("modeBadge");
  const seedLabel = document.getElementById("seedLabel");

  const btnNeuralMode = document.getElementById("btnNeuralMode");
  const btnCompareMode = document.getElementById("btnCompareMode");
  const btnReset = document.getElementById("btnReset");
  const btnPause = document.getElementById("btnPause");
  const fpsSelect = document.getElementById("fpsSelect");
  const btnUp = document.getElementById("btnUp");
  const btnDown = document.getElementById("btnDown");

  const ACTION_NAMES = ["NONE", "UP", "DOWN"];
  let mode = "neural";
  let seedIdx = 0;
  let numSeeds = 16;
  let paused = false;
  let targetFps = 20;
  let ws = null;
  let inFlight = false;
  let lastFrameTs = performance.now();
  let emaFps = 0;

  const keys = { up: false, down: false };
  const neuralImg = neuralCtx.createImageData(84, 84);
  const engineImg = engineCtx.createImageData(84, 84);

  function getAction() {
    if (keys.up && !keys.down) return 1;
    if (keys.down && !keys.up) return 2;
    return 0;
  }

  function updateKeyHud() {
    const a = getAction();
    hudKey.textContent = ACTION_NAMES[a];
    btnUp.classList.toggle("pressed", a === 1);
    btnDown.classList.toggle("pressed", a === 2);
  }

  function blitBase64(b64, ctx, imgData, tintRGB) {
    const bin = atob(b64);
    const data = imgData.data;
    const [tr, tg, tb] = tintRGB;
    for (let i = 0; i < 7056; i++) {
      const v = bin.charCodeAt(i);
      const o = i * 4;
      if (v > 127) {
        data[o] = tr;
        data[o + 1] = tg;
        data[o + 2] = tb;
      } else {
        data[o] = 5;
        data[o + 1] = 8;
        data[o + 2] = 16;
      }
      data[o + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }

  async function loadHealth() {
    const res = await fetch("/health");
    const info = await res.json();
    hudParams.textContent = (info.param_count / 1e6).toFixed(2) + "M (" + info.model_size + ")";
    hudDevice.textContent = info.device.toUpperCase();
    numSeeds = info.num_seeds || 16;
  }

  function setModeUI(newMode) {
    mode = newMode;
    const isNeural = mode === "neural";
    btnNeuralMode.classList.toggle("active", isNeural);
    btnCompareMode.classList.toggle("active", !isNeural);
    enginePanel.classList.toggle("hidden", isNeural);
    modeBadge.className = "badge " + (isNeural ? "neural" : "compare");
    modeBadge.textContent = isNeural
      ? "NEURAL-ONLY MODE \u2022 ZERO GAME CODE"
      : "SPLIT-SCREEN \u2022 REAL ENGINE vs NEURAL MODEL";
  }

  function sendReset(nextSeed) {
    if (nextSeed) seedIdx = (seedIdx + 1) % numSeeds;
    seedLabel.textContent = "Seed #" + seedIdx;
    if (ws && ws.readyState === WebSocket.OPEN) {
      inFlight = true;
      ws.send(JSON.stringify({ type: "reset", mode, seed_idx: seedIdx }));
    }
  }

  function connectWS() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/play`);
    ws.onopen = () => sendReset(false);
    ws.onmessage = (ev) => {
      inFlight = false;
      const msg = JSON.parse(ev.data);
      if (msg.neural_b64) {
        blitBase64(msg.neural_b64, neuralCtx, neuralImg, [235, 255, 245]);
      }
      if (msg.engine_b64) {
        blitBase64(msg.engine_b64, engineCtx, engineImg, [255, 243, 210]);
      }
      if (typeof msg.step_count === "number") {
        hudStep.textContent = String(msg.step_count);
      }
      if (typeof msg.infer_ms === "number") {
        hudMs.textContent = msg.infer_ms.toFixed(1) + " ms";
      }
      const now = performance.now();
      const dt = (now - lastFrameTs) / 1000;
      lastFrameTs = now;
      if (dt > 0.001 && dt < 1.0) {
        const inst = 1.0 / dt;
        emaFps = emaFps === 0 ? inst : emaFps * 0.85 + inst * 0.15;
        hudFps.textContent = emaFps.toFixed(1);
      }
    };
    ws.onclose = () => {
      inFlight = false;
      setTimeout(connectWS, 1000);
    };
  }

  function tickLoop() {
    const intervalMs = 1000 / targetFps;
    if (!paused && ws && ws.readyState === WebSocket.OPEN && !inFlight) {
      inFlight = true;
      ws.send(JSON.stringify({ type: "step", action: getAction() }));
    }
    setTimeout(tickLoop, intervalMs);
  }

  window.addEventListener("keydown", (e) => {
    if (e.key === "ArrowUp" || e.key === "w" || e.key === "W") {
      e.preventDefault();
      keys.up = true;
      updateKeyHud();
    } else if (e.key === "ArrowDown" || e.key === "s" || e.key === "S") {
      e.preventDefault();
      keys.down = true;
      updateKeyHud();
    } else if (e.key === "r" || e.key === "R") {
      sendReset(true);
    }
  });
  window.addEventListener("keyup", (e) => {
    if (e.key === "ArrowUp" || e.key === "w" || e.key === "W") {
      keys.up = false;
      updateKeyHud();
    } else if (e.key === "ArrowDown" || e.key === "s" || e.key === "S") {
      keys.down = false;
      updateKeyHud();
    }
  });

  const bindHold = (el, prop) => {
    const down = (e) => { e.preventDefault(); keys[prop] = true; updateKeyHud(); };
    const up = (e) => { e.preventDefault(); keys[prop] = false; updateKeyHud(); };
    el.addEventListener("mousedown", down);
    el.addEventListener("touchstart", down);
    window.addEventListener("mouseup", up);
    window.addEventListener("touchend", up);
  };
  bindHold(btnUp, "up");
  bindHold(btnDown, "down");

  btnNeuralMode.onclick = () => { setModeUI("neural"); sendReset(false); };
  btnCompareMode.onclick = () => { setModeUI("compare"); sendReset(false); };
  btnReset.onclick = () => sendReset(true);
  btnPause.onclick = () => {
    paused = !paused;
    btnPause.textContent = paused ? "Resume" : "Pause";
    btnPause.classList.toggle("active", paused);
  };
  fpsSelect.onchange = () => { targetFps = Number(fpsSelect.value) || 20; };

  loadHealth().then(() => {
    connectWS();
    tickLoop();
  });
})();
</script>
</body>
</html>
"""


def create_app(
    checkpoint_path: str | Path | None = None,
    seed_frames_path: str | Path | None = None,
    seed_states_path: str | Path | None = None,
    device: str | None = None,
    model: NeuralPongWorldModel | None = None,
    seed_frames: np.ndarray | None = None,
    seed_states: list[dict[str, Any]] | None = None,
) -> FastAPI:
    """Create the FastAPI Neural Pong app (injectable for clean unit testing)."""
    app = FastAPI(title="Neural Pong World Model", version="1.0.0")

    ckpt_p = Path(checkpoint_path or os.environ.get("NEURAL_PONG_CKPT", DEFAULT_CKPT))
    seeds_p = Path(seed_frames_path or os.environ.get("NEURAL_PONG_SEEDS", DEFAULT_SEEDS))
    states_p = Path(seed_states_path or os.environ.get("NEURAL_PONG_SEED_STATES", DEFAULT_SEED_STATES))

    loaded_model: NeuralPongWorldModel
    model_meta: dict[str, Any]
    checkpoint_loaded = False

    if model is not None:
        loaded_model = model.eval()
        dev = next(loaded_model.parameters()).device
        model_meta = {
            "model_size": loaded_model.config.name,
            "param_count": loaded_model.count_parameters(),
            "device": str(dev),
            "checkpoint_path": "in-memory",
        }
        checkpoint_loaded = True
    elif ckpt_p.exists():
        loaded_model, model_meta = load_world_model(ckpt_p, device=device)
        checkpoint_loaded = True
    else:
        # Safe fallback if inspected before training finishes
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        loaded_model = NeuralPongWorldModel("small").to(dev).eval()
        model_meta = {
            "model_size": "small",
            "param_count": loaded_model.count_parameters(),
            "device": str(dev),
            "checkpoint_path": "untrained-fallback",
        }

    if seed_frames is not None:
        loaded_seeds = np.asarray(seed_frames, dtype=np.uint8)
    elif seeds_p.exists():
        loaded_seeds = np.load(seeds_p)
    else:
        # Minimal static fallback seed (4x84x84) so app never crashes before data collection
        fallback = np.zeros((1, 4, 84, 84), dtype=np.uint8)
        fallback[:, :, :2, :] = 255
        fallback[:, :, 82:, :] = 255
        fallback[:, :, 35:49, 5:8] = 255
        fallback[:, :, 35:49, 76:79] = 255
        for k in range(4):
            fallback[0, k, 40:43, 36 + 2 * k : 39 + 2 * k] = 255
        loaded_seeds = fallback

    if seed_states is not None:
        loaded_states = seed_states
    elif states_p.exists():
        with open(states_p, "r", encoding="utf-8") as f:
            loaded_states = json.load(f)
    else:
        loaded_states = []

    sessions: dict[str, ActiveSession] = {}

    def _build_session(mode: Literal["neural", "compare"], seed_idx: int) -> tuple[ActiveSession, np.ndarray, np.ndarray | None]:
        neural = NeuralPongSession(model=loaded_model, seed_frames=loaded_seeds)
        idx = int(seed_idx) % neural.num_seeds
        neural_frame = neural.reset(idx)
        compare_env = None
        engine_frame = None
        if mode == "compare":
            st = loaded_states[idx] if idx < len(loaded_states) else None
            compare_env = _create_compare_env(st)
            engine_frame = compare_env.render()
        return ActiveSession(mode=mode, neural=neural, compare_env=compare_env, seed_idx=idx), neural_frame, engine_frame

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return HTML_UI

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_size": model_meta["model_size"],
            "param_count": int(model_meta["param_count"]),
            "device": model_meta["device"],
            "checkpoint_loaded": checkpoint_loaded,
            "neural_only_ready": True,
            "num_seeds": int(loaded_seeds.shape[0]),
        }

    @app.post("/api/reset")
    async def api_reset(req: ResetRequest) -> dict[str, Any]:
        sess, n_frame, e_frame = _build_session(req.mode, req.seed_idx)
        sid = uuid.uuid4().hex
        sessions[sid] = sess
        # Keep memory bounded for long-running servers
        if len(sessions) > 64:
            oldest = next(iter(sessions))
            if oldest != sid:
                sessions.pop(oldest, None)
        out: dict[str, Any] = {
            "session_id": sid,
            "mode": sess.mode,
            "seed_idx": sess.seed_idx,
            "step_count": 0,
            "frame_shape": [84, 84],
            "neural_b64": _encode_frame_b64(n_frame),
        }
        if e_frame is not None:
            out["engine_b64"] = _encode_frame_b64(e_frame)
        return out

    @app.post("/api/step")
    async def api_step(req: StepRequest) -> dict[str, Any]:
        sess = sessions.get(req.session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="Unknown session_id; call /api/reset first.")
        n_frame, infer_ms = sess.neural.step(req.action)
        out: dict[str, Any] = {
            "session_id": req.session_id,
            "mode": sess.mode,
            "action": req.action,
            "step_count": sess.neural.step_count,
            "infer_ms": round(infer_ms, 3),
            "neural_b64": _encode_frame_b64(n_frame),
        }
        if sess.compare_env is not None:
            e_frame, _, _, _ = sess.compare_env.step(req.action)
            out["engine_b64"] = _encode_frame_b64(e_frame)
        return out

    @app.websocket("/ws/play")
    async def ws_play(ws: WebSocket) -> None:
        await ws.accept()
        sess, n_frame, e_frame = _build_session("neural", 0)
        try:
            while True:
                msg = await ws.receive_json()
                mtype = msg.get("type", "step")
                if mtype == "reset":
                    new_mode = msg.get("mode", "neural")
                    if new_mode not in ("neural", "compare"):
                        new_mode = "neural"
                    new_seed = int(msg.get("seed_idx", 0))
                    sess, n_frame, e_frame = _build_session(new_mode, new_seed)
                    payload: dict[str, Any] = {
                        "type": "frame",
                        "mode": sess.mode,
                        "seed_idx": sess.seed_idx,
                        "step_count": 0,
                        "infer_ms": 0.0,
                        "neural_b64": _encode_frame_b64(n_frame),
                    }
                    if e_frame is not None:
                        payload["engine_b64"] = _encode_frame_b64(e_frame)
                    await ws.send_json(payload)
                elif mtype == "step":
                    action = int(msg.get("action", 0))
                    if action not in (0, 1, 2):
                        action = 0
                    n_frame, infer_ms = sess.neural.step(action)
                    payload = {
                        "type": "frame",
                        "mode": sess.mode,
                        "action": action,
                        "step_count": sess.neural.step_count,
                        "infer_ms": round(infer_ms, 3),
                        "neural_b64": _encode_frame_b64(n_frame),
                    }
                    if sess.compare_env is not None:
                        e_frame, _, _, _ = sess.compare_env.step(action)
                        payload["engine_b64"] = _encode_frame_b64(e_frame)
                    await ws.send_json(payload)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
