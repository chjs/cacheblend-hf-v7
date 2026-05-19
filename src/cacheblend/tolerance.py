"""Tolerance categories for FP16 GPU correctness checks.

Frozen at Phase start, retroactive change forbidden [L16].
4 categories cover all v3-observed cases [L05, L13].

torch is imported lazily so that the cacheblend package can be smoke-imported
on environments without torch (e.g., mac stub during early Phase 0 work).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch  # type: ignore


class Tolerance(Enum):
    """Tolerance category for assert_logits_close().
    
    - IDENTICAL_PATH: max_diff == 0 (코드 경로 동일 보장, e.g. boundary safe-shortcut)
    - SAME_SHAPE: max_diff < 1e-3 (bit-exact 가능, e.g. layerwise vs standard)
    - MIXED_SHAPE: argmax exact AND max_diff < 5e-2 (cuBLAS shape difference)
    - RECOMPUTE_PATH: max_diff < 1e-3 (같은 fused shape, e.g. ratio=1 vs full_recompute)
    """
    IDENTICAL_PATH = "identical"
    SAME_SHAPE = "same_shape"
    MIXED_SHAPE = "mixed_shape"
    RECOMPUTE_PATH = "recompute_path"


@dataclass
class ToleranceResult:
    passed: bool
    max_diff: float
    argmax_match_ratio: float
    category: Tolerance
    detail: str


def assert_logits_close(
    actual: "torch.Tensor",
    expected: "torch.Tensor",
    category: Tolerance,
    name: str = "logits",
) -> ToleranceResult:
    """Compare actual vs expected with given tolerance category.
    
    Returns ToleranceResult. Raises AssertionError if failed.
    """
    if actual.shape != expected.shape:
        raise AssertionError(f"{name} shape mismatch: {actual.shape} vs {expected.shape}")
    
    diff = (actual.float() - expected.float()).abs()
    max_diff = float(diff.max().item())
    
    actual_argmax = actual.argmax(dim=-1)
    expected_argmax = expected.argmax(dim=-1)
    argmax_match = (actual_argmax == expected_argmax).float().mean().item()
    
    if category == Tolerance.IDENTICAL_PATH:
        passed = max_diff == 0.0
        bound_str = "max_diff == 0"
    elif category == Tolerance.SAME_SHAPE:
        passed = max_diff < 1e-3
        bound_str = "max_diff < 1e-3"
    elif category == Tolerance.MIXED_SHAPE:
        passed = (argmax_match == 1.0) and (max_diff < 5e-2)
        bound_str = "argmax == 1.0 AND max_diff < 5e-2"
    elif category == Tolerance.RECOMPUTE_PATH:
        passed = max_diff < 1e-3
        bound_str = "max_diff < 1e-3"
    else:
        raise ValueError(f"Unknown tolerance category: {category}")
    
    detail = (
        f"{name}: max_diff={max_diff:.3e}, argmax_match={argmax_match:.4f}, "
        f"category={category.value}, bound={bound_str}, passed={passed}"
    )
    
    result = ToleranceResult(
        passed=passed,
        max_diff=max_diff,
        argmax_match_ratio=argmax_match,
        category=category,
        detail=detail,
    )
    
    if not passed:
        raise AssertionError(detail)
    
    return result
