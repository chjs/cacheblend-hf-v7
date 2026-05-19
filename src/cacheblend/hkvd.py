"""HKVD (High KV Deviation) selection.

Phase 3 implementation. Spec aligned with LMCache `blender.py:89-101` and paper §4.2.

Formula (token-level deviation):
    diff_k[i] = sum_f (K_new[i, f] - K_old[i, f])^2          (squared L2 over feature axis)

- K only (V is not part of the deviation; LMCache same).
- fp32 cast on inputs (LMCache same — `to(torch.float32)`).
- No normalization.
- Top-K selection: int(N * ratio), max(1), then `torch.topk(..., largest=True)`,
  then sort indices ascending (causal-mask compatible). LMCache same.

Cross-references:
- LMCache: `external/LMCache/lmcache/v1/compute/blend/blender.py:89-91, 94-101`
- docs/lmcache-analysis.md §Q2 (a–d)
"""
from __future__ import annotations

import torch


def kv_deviation(K_new: torch.Tensor, K_old: torch.Tensor) -> torch.Tensor:
    """Per-token squared-L2 deviation between two K tensors.

    Inputs:
        K_new, K_old — same shape. We support both:
          (B=1, S, num_kv_heads*head_dim)  — k_proj output (post-RoPE if shifted)
          (S, num_kv_heads*head_dim)       — squeezed equivalent
        feature axis is the last dim. The leading axes are summed over only
        within the same shape (deviation is per-token along S).

    Returns:
        deviations: (S,) — fp32, one scalar per token position.

    Equivalent to LMCache `blender.py:89-91`:
        diff_k = torch.sum((k.to(fp32) - old_k.to(fp32))**2, dim=[1])
    """
    if K_new.shape != K_old.shape:
        raise ValueError(f"shape mismatch: K_new={K_new.shape}, K_old={K_old.shape}")

    K_new_f = K_new.to(torch.float32)
    K_old_f = K_old.to(torch.float32)
    diff = (K_new_f - K_old_f) ** 2

    # Sum over feature axis (and any batch axis if present), leaving S.
    if diff.ndim == 3:
        # (B, S, F) → sum over F → (B, S) → squeeze B (we only support B=1 here).
        if diff.shape[0] != 1:
            raise NotImplementedError("kv_deviation: batch>1 not supported in Phase 3")
        return diff.sum(dim=-1).squeeze(0)
    elif diff.ndim == 2:
        return diff.sum(dim=-1)
    else:
        raise ValueError(f"unsupported ndim: {diff.ndim}")


def select_top_k(deviations: torch.Tensor, ratio: float) -> torch.Tensor:
    """Select top-K indices by deviation, then resort ascending (causal-friendly).

    Returns:
        indices: (k,) int64 tensor on same device as `deviations`. Sorted ascending.

    Equivalent to LMCache `blender.py:94-101`:
        topk_num = int(total_len * recomp_ratios[0])
        topk_num = max(topk_num, 1)
        top_indices = torch.topk(diff_k, k=topk_num).indices
        top_indices, _ = torch.sort(top_indices)
    """
    total_len = deviations.shape[0]
    topk_num = int(total_len * ratio)
    topk_num = max(topk_num, 1)
    topk_num = min(topk_num, total_len)

    top_indices = torch.topk(deviations, k=topk_num, largest=True).indices
    top_indices, _ = torch.sort(top_indices)
    return top_indices
