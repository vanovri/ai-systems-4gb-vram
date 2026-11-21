"""Pure NumPy group-wise INT8/INT4 quantizer with real 4-bit nibble packing and GPTQ-lite.

Design highlights:
- Zero deep-learning framework dependencies in the core quantizer (pure NumPy).
- Group-wise asymmetric AND symmetric quantization (default group_size=64).
- Safe div-by-zero guard: s = np.maximum(range / qmax, 1e-8) prevents NaN/Inf on constant groups.
- Real 4-bit nibble packing: two INT4 values packed per np.uint8 byte
  `(q_even & 0x0F) | ((q_odd & 0x0F) << 4)` for real ~3.5x-3.8x RAM/disk compression
  including FP16 scale/min metadata overhead.
- GPTQ-lite (Hessian-aware block Cholesky error compensation) for layer-level ablation vs RTN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np

QuantMode = Literal["asymmetric", "symmetric"]

EPS_SCALE = 1e-8


def pack_int4(q: np.ndarray) -> np.ndarray:
    """Pack 4-bit unsigned integers (0..15) two per uint8 byte.

    Layout per byte: low nibble = even index, high nibble = odd index:
        byte[k] = (q[2k] & 0x0F) | ((q[2k + 1] & 0x0F) << 4)

    Works on arbitrary shapes by flattening to 1D (if odd length, pads one 0 nibble).
    """
    arr = np.asarray(q)
    if arr.size == 0:
        return np.empty(0, dtype=np.uint8)
    if arr.min() < 0 or arr.max() > 15:
        raise ValueError(f"pack_int4 expects values in [0, 15], got [{arr.min()}, {arr.max()}]")

    flat = arr.astype(np.uint8, copy=False).ravel()
    if flat.size % 2 != 0:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])

    q_even = flat[0::2] & np.uint8(0x0F)
    q_odd = flat[1::2] & np.uint8(0x0F)
    return (q_even | (q_odd << np.uint8(4))).astype(np.uint8)


def unpack_int4(packed: np.ndarray, orig_numel: int) -> np.ndarray:
    """Unpack uint8 nibble-packed buffer back to `orig_numel` uint8 values in [0, 15]."""
    if orig_numel < 0:
        raise ValueError(f"orig_numel must be non-negative, got {orig_numel}")
    if orig_numel == 0:
        return np.empty(0, dtype=np.uint8)

    buf = np.asarray(packed, dtype=np.uint8).ravel()
    expected_bytes = (orig_numel + 1) // 2
    if buf.size < expected_bytes:
        raise ValueError(
            f"Packed buffer too small: need at least {expected_bytes} bytes for "
            f"{orig_numel} 4-bit values, got {buf.size}"
        )

    buf = buf[:expected_bytes]
    out = np.empty(expected_bytes * 2, dtype=np.uint8)
    out[0::2] = buf & np.uint8(0x0F)
    out[1::2] = (buf >> np.uint8(4)) & np.uint8(0x0F)
    return out[:orig_numel]


@dataclass
class QuantizedTensor:
    """Self-contained quantized weight tensor with true packed storage & metadata."""

    qdata: np.ndarray  # uint8 (packed nibbles for bits=4, raw uint8/int8 for bits=8)
    scales: np.ndarray  # float16 (num_groups,)
    zeros: Optional[np.ndarray]  # float16 (num_groups,) for asymmetric, None for symmetric
    orig_shape: Tuple[int, ...]
    bits: int
    group_size: int
    mode: QuantMode
    pad_len: int = 0
    is_packed: bool = False

    @property
    def numel(self) -> int:
        return int(np.prod(self.orig_shape))

    @property
    def padded_numel(self) -> int:
        return self.numel + self.pad_len

    @property
    def data_nbytes(self) -> int:
        return int(self.qdata.nbytes)

    @property
    def meta_nbytes(self) -> int:
        total = int(self.scales.nbytes)
        if self.zeros is not None:
            total += int(self.zeros.nbytes)
        return total

    @property
    def total_nbytes(self) -> int:
        """Actual in-memory byte footprint: packed weights + FP16 scale/min metadata."""
        return self.data_nbytes + self.meta_nbytes

    @property
    def fp16_baseline_nbytes(self) -> int:
        return self.numel * 2

    @property
    def effective_bits_per_weight(self) -> float:
        if self.numel == 0:
            return 0.0
        return (self.total_nbytes * 8.0) / self.numel

    @property
    def compression_ratio_vs_fp16(self) -> float:
        if self.total_nbytes == 0:
            return 0.0
        return self.fp16_baseline_nbytes / self.total_nbytes

    def to_npz_dict(self) -> Dict[str, np.ndarray]:
        payload: Dict[str, np.ndarray] = {
            "qdata": self.qdata,
            "scales": self.scales,
            "meta": np.array(
                [self.bits, self.group_size, self.pad_len, int(self.is_packed)],
                dtype=np.int64,
            ),
            "orig_shape": np.array(self.orig_shape, dtype=np.int64),
            "mode": np.array(self.mode),
        }
        if self.zeros is not None:
            payload["zeros"] = self.zeros
        return payload


def _prepare_groups(w: np.ndarray, group_size: int) -> Tuple[np.ndarray, int]:
    """Flatten weight array and pad to a multiple of `group_size`."""
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    w_f32 = np.asarray(w, dtype=np.float32).ravel()
    rem = w_f32.size % group_size
    pad_len = (group_size - rem) % group_size
    if pad_len > 0:
        w_f32 = np.concatenate([w_f32, np.zeros(pad_len, dtype=np.float32)])
    return w_f32.reshape(-1, group_size), pad_len


def quantize_groupwise(
    w: np.ndarray,
    bits: int = 4,
    group_size: int = 64,
    mode: QuantMode = "asymmetric",
    pack: bool = True,
) -> QuantizedTensor:
    """Quantize arbitrary weight tensor into group-wise INT4 or INT8.

    Asymmetric formula (with div-by-zero guard):
        s = np.maximum((w_max - w_min) / (2**b - 1), 1e-8)
        q = np.clip(np.round((w - w_min) / s), 0, 2**b - 1).astype(np.uint8)

    Symmetric formula (zero-centered, 1 FP16 scale per group):
        qmax_sym = 2**(b - 1) - 1
        s = np.maximum(np.max(|w|) / qmax_sym, 1e-8)
        q_signed = np.clip(np.round(w / s), -qmax_sym, qmax_sym)
        (For bits=4 packed, stored with unsigned bias +8 -> [1..15] inside 4-bit nibbles.)
    """
    if bits not in (4, 8):
        raise ValueError(f"Only bits in {{4, 8}} supported, got {bits}")
    if mode not in ("asymmetric", "symmetric"):
        raise ValueError(f"Unsupported mode: {mode}")

    orig_shape = tuple(np.asarray(w).shape)
    groups, pad_len = _prepare_groups(w, group_size)

    if mode == "asymmetric":
        qmax = (1 << bits) - 1
        w_min = groups.min(axis=1, keepdims=True)
        w_max = groups.max(axis=1, keepdims=True)
        raw_scale = (w_max - w_min) / float(qmax)
        s = np.maximum(raw_scale, EPS_SCALE)

        # Store metadata in FP16 and quantize against the FP16-realizable scale/min
        # so dequantization matches real packed deployment with zero NaN/Inf.
        w_min_f16 = w_min.astype(np.float16)
        s_f16 = np.where(raw_scale > 0.0, np.maximum(s, 1e-7), 0.0).astype(np.float16)
        s_eff = np.maximum(s_f16.astype(np.float32), EPS_SCALE)
        w_min_eff = w_min_f16.astype(np.float32)

        q = np.clip(np.round((groups - w_min_eff) / s_eff), 0, qmax).astype(np.uint8)
        zeros_out: Optional[np.ndarray] = w_min_f16.ravel()
        scales_out = s_f16.ravel()

        if bits == 4 and pack:
            qdata = pack_int4(q)
            is_packed = True
        else:
            qdata = q.ravel()
            is_packed = False

    else:  # symmetric
        qmax_sym = (1 << (bits - 1)) - 1  # 7 for INT4, 127 for INT8
        w_abs_max = np.max(np.abs(groups), axis=1, keepdims=True)
        raw_scale = w_abs_max / float(qmax_sym)
        s = np.maximum(raw_scale, EPS_SCALE)

        s_f16 = np.where(raw_scale > 0.0, np.maximum(s, 1e-7), 0.0).astype(np.float16)
        s_eff = np.maximum(s_f16.astype(np.float32), EPS_SCALE)

        q_signed = np.clip(np.round(groups / s_eff), -qmax_sym, qmax_sym).astype(np.int8)
        scales_out = s_f16.ravel()
        zeros_out = None

        if bits == 4 and pack:
            # Offset by +8 so [-7, 7] maps cleanly into unsigned nibbles [1, 15].
            q_unsigned = (q_signed.astype(np.int16) + 8).astype(np.uint8)
            qdata = pack_int4(q_unsigned)
            is_packed = True
        elif bits == 4:
            qdata = q_signed.ravel()
            is_packed = False
        else:
            qdata = q_signed.ravel()
            is_packed = False

    return QuantizedTensor(
        qdata=qdata,
        scales=scales_out,
        zeros=zeros_out,
        orig_shape=orig_shape,
        bits=bits,
        group_size=group_size,
        mode=mode,
        pad_len=pad_len,
        is_packed=is_packed,
    )


def dequantize_groupwise(qt: QuantizedTensor, out_dtype: np.dtype = np.float32) -> np.ndarray:
    """Reconstruct floating-point weight array from a QuantizedTensor."""
    padded_numel = qt.padded_numel
    num_groups = padded_numel // qt.group_size

    if qt.bits == 4 and qt.is_packed:
        q_flat = unpack_int4(qt.qdata, padded_numel)
    else:
        q_flat = qt.qdata.ravel()[:padded_numel]

    q_groups = q_flat.reshape(num_groups, qt.group_size)
    scales = qt.scales.astype(np.float32).reshape(num_groups, 1)

    if qt.mode == "asymmetric":
        if qt.zeros is None:
            raise ValueError("Asymmetric QuantizedTensor missing `zeros` (w_min) metadata")
        zeros = qt.zeros.astype(np.float32).reshape(num_groups, 1)
        w_hat = q_groups.astype(np.float32) * scales + zeros
    else:
        if qt.bits == 4 and qt.is_packed:
            q_signed = q_groups.astype(np.float32) - 8.0
        else:
            q_signed = q_groups.astype(np.float32)
        w_hat = q_signed * scales

    flat = w_hat.ravel()
    if qt.pad_len > 0:
        flat = flat[: qt.numel]
    return flat.reshape(qt.orig_shape).astype(out_dtype, copy=False)


def compute_quant_metrics(
    w_orig: np.ndarray,
    w_hat: np.ndarray,
    qt: Optional[QuantizedTensor] = None,
) -> Dict[str, float]:
    """Compute relative L1 error epsilon, RMSE, max absolute error, and byte stats."""
    orig = np.asarray(w_orig, dtype=np.float64)
    recon = np.asarray(w_hat, dtype=np.float64)
    if orig.shape != recon.shape:
        raise ValueError(f"Shape mismatch: orig {orig.shape} vs recon {recon.shape}")

    diff = recon - orig
    abs_diff = np.abs(diff)
    l1_orig = float(np.sum(np.abs(orig)))
    l1_err = float(np.sum(abs_diff))
    rel_l1 = l1_err / max(l1_orig, EPS_SCALE)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0

    metrics: Dict[str, float] = {
        "rel_l1_error": rel_l1,
        "rmse": rmse,
        "max_abs_error": max_abs,
    }
    if qt is not None:
        metrics.update(
            {
                "fp16_nbytes": float(qt.fp16_baseline_nbytes),
                "quant_nbytes": float(qt.total_nbytes),
                "data_nbytes": float(qt.data_nbytes),
                "meta_nbytes": float(qt.meta_nbytes),
                "effective_bits_per_weight": qt.effective_bits_per_weight,
                "compression_ratio_vs_fp16": qt.compression_ratio_vs_fp16,
            }
        )
    return metrics


def get_group_snap_levels(
    w_slice: np.ndarray,
    bits: int = 4,
    mode: QuantMode = "asymmetric",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dequantized_weights, allowed_quantization_levels) for a single group slice.

    Ideal for visualizing the 16 allowed 4-bit snap levels against 64 real weights.
    """
    arr = np.asarray(w_slice, dtype=np.float32).ravel()
    qt = quantize_groupwise(arr, bits=bits, group_size=arr.size, mode=mode, pack=(bits == 4))
    w_hat = dequantize_groupwise(qt, out_dtype=np.float32)

    s = float(qt.scales[0].astype(np.float32))
    if mode == "asymmetric":
        z = float(qt.zeros[0].astype(np.float32))  # type: ignore[index]
        levels = np.arange(1 << bits, dtype=np.float32) * s + z
    else:
        qmax_sym = (1 << (bits - 1)) - 1
        levels = np.arange(-qmax_sym, qmax_sym + 1, dtype=np.float32) * s
    return w_hat, levels


def gptq_lite_quantize(
    w: np.ndarray,
    x_calib: np.ndarray,
    bits: int = 4,
    group_size: int = 64,
    block_size: int = 64,
    damp_percent: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Pure NumPy GPTQ-lite (Hessian-aware blocked Cholesky quantization) for a 2D Linear weight.

    Args:
        w: Weight matrix of shape (out_features, in_features).
        x_calib: Calibration input activations of shape (num_samples, in_features).
        bits: Target bitwidth (4 or 8).
        group_size: Group size along `in_features` for per-group asymmetric scales/mins.
        block_size: Column block size for lazy Hessian error updates.
        damp_percent: Diagonal damping fraction added to Hessian for positive definiteness.

    Returns:
        (w_gptq_f32, stats_dict) comparing output activation MSE & weight error vs RTN.
    """
    W = np.asarray(w, dtype=np.float64).copy()
    X = np.asarray(x_calib, dtype=np.float64)
    if W.ndim != 2 or X.ndim != 2:
        raise ValueError("gptq_lite_quantize expects 2D W (out, in) and 2D X (n_samples, in)")
    d_out, d_in = W.shape
    if X.shape[1] != d_in:
        raise ValueError(f"Feature mismatch: W has in_features={d_in}, X has {X.shape[1]}")

    # Empirical Hessian H = (2 / N) * X^T X + damp * I
    n_samples = max(X.shape[0], 1)
    H = (2.0 / n_samples) * (X.T @ X)
    diag = np.diag(H)
    dead = diag == 0.0
    if np.any(dead):
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
    damp = damp_percent * float(np.mean(np.diag(H)))
    H[np.arange(d_in), np.arange(d_in)] += max(damp, 1e-6)

    # Compute upper-triangular Cholesky factor U of H^{-1} such that U.T @ U == H_inv
    # (equivalent to torch.linalg.cholesky(H_inv, upper=True) in GPTQ Algorithm 1).
    # In NumPy, L = cholesky(H_inv) gives L @ L.T == H_inv, so U = L.T gives U.T @ U == H_inv.
    L_H = np.linalg.cholesky(H)
    L_inv = np.linalg.solve(L_H, np.eye(d_in, dtype=np.float64))
    H_inv = L_inv.T @ L_inv
    U = np.linalg.cholesky(H_inv).T

    qmax = float((1 << bits) - 1)
    Q = np.zeros_like(W)
    W_work = W.copy()

    # Precompute group scales/zeros per row-group on W_work as we advance
    scales_g = np.ones((d_out, 1), dtype=np.float64)
    zeros_g = np.zeros((d_out, 1), dtype=np.float64)

    for i1 in range(0, d_in, block_size):
        i2 = min(i1 + block_size, d_in)
        count = i2 - i1
        W1 = W_work[:, i1:i2].copy()
        Q1 = np.zeros_like(W1)
        Err1 = np.zeros_like(W1)
        U1 = U[i1:i2, i1:i2]

        for i in range(count):
            col_idx = i1 + i
            if col_idx % group_size == 0:
                g_end = min(col_idx + group_size, d_in)
                W_grp = W_work[:, col_idx:g_end]
                w_min = W_grp.min(axis=1, keepdims=True)
                w_max = W_grp.max(axis=1, keepdims=True)
                scales_g = np.maximum((w_max - w_min) / qmax, EPS_SCALE)
                zeros_g = w_min

            w_col = W1[:, i : i + 1]
            q_int = np.clip(np.round((w_col - zeros_g) / scales_g), 0.0, qmax)
            q_col = q_int * scales_g + zeros_g
            Q1[:, i : i + 1] = q_col

            d_val = U1[i, i]
            err = (w_col - q_col) / max(d_val, EPS_SCALE)
            Err1[:, i : i + 1] = err
            if i + 1 < count:
                W1[:, i + 1 :] -= err @ U1[i : i + 1, i + 1 :]

        Q[:, i1:i2] = Q1
        if i2 < d_in:
            W_work[:, i2:] -= Err1 @ U[i1:i2, i2:]

    w_gptq = Q.astype(np.float32)

    # Also compute RTN baseline with identical group_size along rows for apples-to-apples comparison
    qt_rtn = quantize_groupwise(
        w.astype(np.float32), bits=bits, group_size=group_size, mode="asymmetric", pack=(bits == 4)
    )
    w_rtn = dequantize_groupwise(qt_rtn, out_dtype=np.float32)

    # Compare output activation error ||Y_orig - Y_quant||_F / ||Y_orig||_F
    y_orig = X @ W.T
    y_norm = float(np.linalg.norm(y_orig)) + EPS_SCALE
    y_rtn_err = float(np.linalg.norm(y_orig - (X @ w_rtn.astype(np.float64).T))) / y_norm
    y_gptq_err = float(np.linalg.norm(y_orig - (X @ w_gptq.astype(np.float64).T))) / y_norm

    rtn_metrics = compute_quant_metrics(w, w_rtn)
    gptq_metrics = compute_quant_metrics(w, w_gptq)

    stats = {
        "bits": float(bits),
        "group_size": float(group_size),
        "rtn_rel_l1_weight_error": rtn_metrics["rel_l1_error"],
        "gptq_rel_l1_weight_error": gptq_metrics["rel_l1_error"],
        "rtn_output_rel_l2_error": y_rtn_err,
        "gptq_output_rel_l2_error": y_gptq_err,
        "output_error_reduction_pct": (1.0 - y_gptq_err / max(y_rtn_err, EPS_SCALE)) * 100.0,
    }
    return w_gptq, stats
