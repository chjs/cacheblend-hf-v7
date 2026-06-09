"""Compressed-KV blending (CompBlend goal 1: cacheblend + kvzip, only-HKVD).

Dep-light core (torch only): CompressedChunk + token_prune + to_blend_inputs +
CompressionBackend ABC. The KVzip backend (heavy: flash_attn + KVzip repo) is in
`cacheblend.compress.kvzip` and lazy-imported only when scoring.
"""
from __future__ import annotations

from cacheblend.compress.base import (
    CompressionBudget,
    CompressedChunk,
    CompressionBackend,
    token_prune,
    reduce_importance,
    to_blend_inputs,
)

__all__ = [
    "CompressionBudget",
    "CompressedChunk",
    "CompressionBackend",
    "token_prune",
    "reduce_importance",
    "to_blend_inputs",
]
