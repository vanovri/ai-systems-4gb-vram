"""Unit tests for pure NumPy INT8/INT4 group-wise quantizer, nibble packing, GPTQ-lite, and PPL math."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.quantizer import (  # noqa: E402
    compute_quant_metrics,
    dequantize_groupwise,
    get_group_snap_levels,
    gptq_lite_quantize,
    pack_int4,
    quantize_groupwise,
    unpack_int4,
)


def test_pack_unpack_int4_all_16_values_lossless():
    """Every 4-bit value 0..15 and all 256 (even, odd) pairs roundtrip losslessly."""
    pairs = np.array([[e, o] for e in range(16) for o in range(16)], dtype=np.uint8).ravel()
    packed = pack_int4(pairs)
    assert packed.dtype == np.uint8
    assert packed.nbytes == pairs.size // 2
    unpacked = unpack_int4(packed, pairs.size)
    assert np.array_equal(unpacked, pairs)


def test_pack_unpack_int4_odd_length():
    """Odd-length arrays pad cleanly on pack and restore exact length on unpack."""
    vals = np.array([0, 15, 7, 8, 3, 12, 9], dtype=np.uint8)
    packed = pack_int4(vals)
    assert packed.size == 4
    unpacked = unpack_int4(packed, vals.size)
    assert unpacked.shape == (7,)
    assert np.array_equal(unpacked, vals)


def test_pack_int4_rejects_out_of_range():
    with pytest.raises(ValueError):
        pack_int4(np.array([0, 16], dtype=np.uint8))
    with pytest.raises(ValueError):
        pack_int4(np.array([-1, 5], dtype=np.int16))


def test_constant_tensor_zero_and_nonzero_no_nan_inf():
    """Constant groups (w_max == w_min) must never produce NaN or Inf."""
    for val in [0.0, 3.25, -7.5, 1e-6]:
        w = np.full((8, 64), val, dtype=np.float32)
        for mode in ("asymmetric", "symmetric"):
            for bits in (4, 8):
                qt = quantize_groupwise(w, bits=bits, group_size=64, mode=mode, pack=(bits == 4))
                w_hat = dequantize_groupwise(qt)
                assert np.all(np.isfinite(w_hat)), f"Non-finite for val={val}, mode={mode}, bits={bits}"
                if mode == "asymmetric":
                    assert np.allclose(w_hat, np.float16(val).astype(np.float32), atol=1e-2)


def test_monotonic_error_int8_strictly_better_than_int4():
    """On realistic Gaussian weights, error(INT8) must be strictly less than error(INT4)."""
    rng = np.random.default_rng(42)
    w = rng.normal(0.0, 0.04, size=(256, 256)).astype(np.float32)

    for mode in ("asymmetric", "symmetric"):
        qt8 = quantize_groupwise(w, bits=8, group_size=64, mode=mode, pack=False)
        qt4 = quantize_groupwise(w, bits=4, group_size=64, mode=mode, pack=True)
        m8 = compute_quant_metrics(w, dequantize_groupwise(qt8), qt8)
        m4 = compute_quant_metrics(w, dequantize_groupwise(qt4), qt4)
        assert m8["rel_l1_error"] < m4["rel_l1_error"]
        assert m8["rmse"] < m4["rmse"]
        assert m8["rel_l1_error"] < 0.01  # INT8 should be < 1% relative L1 error
        assert m4["rel_l1_error"] < 0.15  # INT4 should be reasonable (< 15%)


def test_exact_byte_size_accounting_int4_asymmetric():
    """INT4 asymmetric g=64: 32B packed + 2B scale + 2B min = 36B per 64 weights (4.5 bits/w, 3.56x)."""
    w = np.random.default_rng(7).normal(0.0, 0.05, size=(128, 256)).astype(np.float32)
    numel = w.size  # 32,768
    num_groups = numel // 64  # 512

    qt = quantize_groupwise(w, bits=4, group_size=64, mode="asymmetric", pack=True)
    assert qt.is_packed is True
    assert qt.data_nbytes == numel // 2
    assert qt.meta_nbytes == num_groups * 4  # fp16 scale + fp16 min
    assert qt.total_nbytes == numel // 2 + num_groups * 4
    assert math.isclose(qt.effective_bits_per_weight, 4.5, rel_tol=1e-6)
    assert math.isclose(qt.compression_ratio_vs_fp16, 128.0 / 36.0, rel_tol=1e-6)
    assert 3.5 <= qt.compression_ratio_vs_fp16 <= 3.8


def test_exact_byte_size_accounting_int4_symmetric():
    """INT4 symmetric g=64: 32B packed + 2B scale = 34B per 64 weights (4.25 bits/w, 3.76x)."""
    w = np.random.default_rng(8).normal(0.0, 0.05, size=(128, 256)).astype(np.float32)
    numel = w.size
    num_groups = numel // 64

    qt = quantize_groupwise(w, bits=4, group_size=64, mode="symmetric", pack=True)
    assert qt.is_packed is True
    assert qt.zeros is None
    assert qt.data_nbytes == numel // 2
    assert qt.meta_nbytes == num_groups * 2
    assert math.isclose(qt.effective_bits_per_weight, 4.25, rel_tol=1e-6)
    assert math.isclose(qt.compression_ratio_vs_fp16, 128.0 / 34.0, rel_tol=1e-6)
    assert 3.5 <= qt.compression_ratio_vs_fp16 <= 3.8


def test_exact_byte_size_accounting_int8_asymmetric():
    """INT8 asymmetric g=64: 64B data + 4B meta = 68B per 64 weights (8.5 bits/w, 1.88x)."""
    w = np.random.default_rng(9).normal(0.0, 0.05, size=(64, 128)).astype(np.float32)
    qt = quantize_groupwise(w, bits=8, group_size=64, mode="asymmetric", pack=False)
    assert qt.data_nbytes == w.size
    assert qt.meta_nbytes == (w.size // 64) * 4
    assert math.isclose(qt.effective_bits_per_weight, 8.5, rel_tol=1e-6)
    assert math.isclose(qt.compression_ratio_vs_fp16, 128.0 / 68.0, rel_tol=1e-6)


def test_non_multiple_of_group_size_shape_preserved():
    """Tensors whose element count is not a multiple of group_size pad & unpad cleanly."""
    rng = np.random.default_rng(123)
    w = rng.normal(0.0, 0.1, size=(13, 17)).astype(np.float32)
    qt = quantize_groupwise(w, bits=4, group_size=64, mode="asymmetric", pack=True)
    assert qt.pad_len == (64 - (w.size % 64)) % 64
    w_hat = dequantize_groupwise(qt)
    assert w_hat.shape == (13, 17)
    assert np.all(np.isfinite(w_hat))


def test_asymmetric_handles_skewed_positive_distribution_better_than_symmetric():
    """For shifted/skewed positive weights [1.0, 2.0], asymmetric wastes 0 dynamic range."""
    rng = np.random.default_rng(99)
    w = rng.uniform(1.0, 2.0, size=(32, 64)).astype(np.float32)
    m_asym = compute_quant_metrics(
        w, dequantize_groupwise(quantize_groupwise(w, bits=4, group_size=64, mode="asymmetric"))
    )
    m_sym = compute_quant_metrics(
        w, dequantize_groupwise(quantize_groupwise(w, bits=4, group_size=64, mode="symmetric"))
    )
    assert m_asym["rmse"] < m_sym["rmse"]


def test_get_group_snap_levels_has_16_levels_for_int4():
    """64 real weights snapped to INT4 must land exactly on the 16 allowed snap levels."""
    rng = np.random.default_rng(10)
    w64 = rng.normal(0.0, 0.03, size=(64,)).astype(np.float32)
    w_hat, levels = get_group_snap_levels(w64, bits=4, mode="asymmetric")
    assert levels.shape == (16,)
    assert w_hat.shape == (64,)
    # Every dequantized weight must equal one of the 16 allowed levels
    dists = np.min(np.abs(w_hat[:, None] - levels[None, :]), axis=1)
    assert float(np.max(dists)) < 1e-6


def test_gptq_lite_beats_rtn_on_correlated_activations():
    """GPTQ-lite Hessian compensation must achieve lower output MSE than RTN on correlated inputs."""
    rng = np.random.default_rng(2026)
    d_out, d_in, n_tok = 64, 128, 256
    W = rng.normal(0.0, 0.05, size=(d_out, d_in)).astype(np.float32)
    # Strongly correlated calibration activations (realistic LLM hidden states)
    low_rank = rng.normal(0.0, 1.0, size=(n_tok, 16))
    proj = rng.normal(0.0, 1.0, size=(16, d_in))
    X = (low_rank @ proj + 0.15 * rng.normal(0.0, 1.0, size=(n_tok, d_in))).astype(np.float32)

    w_gptq, stats = gptq_lite_quantize(W, X, bits=4, group_size=64, block_size=64)
    assert w_gptq.shape == W.shape
    assert np.all(np.isfinite(w_gptq))
    assert stats["gptq_output_rel_l2_error"] < stats["rtn_output_rel_l2_error"]
    assert stats["output_error_reduction_pct"] > 5.0


def test_ppl_math_verification():
    """Verify perplexity formula PPL = exp(-1/N sum log p_t) against analytical values."""
    # Uniform probability 1/V for every token -> NLL = ln(V) -> PPL = V
    vocab_size = 50
    n_tokens = 200
    log_probs = np.full(n_tokens, -math.log(vocab_size), dtype=np.float64)
    mean_nll = float(-np.mean(log_probs))
    ppl = math.exp(mean_nll)
    assert math.isclose(ppl, float(vocab_size), rel_tol=1e-9)

    # Certain predictions (p = 1.0) -> NLL = 0 -> PPL = 1.0
    ppl_perfect = math.exp(float(-np.mean(np.zeros(n_tokens))))
    assert math.isclose(ppl_perfect, 1.0, rel_tol=1e-12)


def test_npz_serialization_roundtrip(tmp_path: Path):
    """QuantizedTensor serialized via np.savez preserves exact byte footprint & reconstruction."""
    rng = np.random.default_rng(55)
    w = rng.normal(0.0, 0.02, size=(64, 128)).astype(np.float32)
    qt = quantize_groupwise(w, bits=4, group_size=64, mode="asymmetric", pack=True)
    out_file = tmp_path / "layer.npz"
    np.savez(out_file, **qt.to_npz_dict())
    loaded = np.load(out_file)
    assert np.array_equal(loaded["qdata"], qt.qdata)
    assert np.array_equal(loaded["scales"], qt.scales)
    assert np.array_equal(loaded["zeros"], qt.zeros)


def test_target_linear_selector_protects_embeddings_and_lm_head():
    """Verify quantizer targets only attention/MLP projections and keeps embeddings/lm_head FP16."""
    import torch.nn as nn
    from src.bench_qwen import is_target_linear

    lin = nn.Linear(64, 64)
    emb = nn.Embedding(100, 64)
    assert is_target_linear("model.layers.10.self_attn.q_proj", lin) is True
    assert is_target_linear("model.layers.10.mlp.down_proj", lin) is True
    assert is_target_linear("lm_head", lin) is False
    assert is_target_linear("model.embed_tokens", emb) is False
    assert is_target_linear("model.layers.10.input_layernorm", nn.LayerNorm(64)) is False


def test_layer10_snap_plot_generates_png(tmp_path: Path):
    """Verify Layer-10 INT4 snap visualization writes a valid non-empty PNG and accurate metadata."""
    from src.visualize import plot_layer10_int4_snap

    rng = np.random.default_rng(77)
    w64 = rng.normal(0.0, 0.03, size=(64,)).astype(np.float32)
    out_png = tmp_path / "layer10_snap.png"
    info = plot_layer10_int4_snap(w64, out_png, layer_label="synthetic_layer10")
    assert out_png.exists() and out_png.stat().st_size > 10_000
    assert info["num_weights"] == 64
    assert info["num_levels"] == 16
    assert info["packed_bytes"] == 36
    assert info["fp16_bytes"] == 128
