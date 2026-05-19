"""CacheBlend HF v4 — quality-only milestone with gradual filtering discovery.

All classes pre-stubbed for Phase 0 smoke test [L25].
Real implementations in respective modules.
"""
from __future__ import annotations

__version__ = "0.4.0"

# Re-export public API. Each module has a stub class that's smoke-importable
# even before its phase implementation is done.
from cacheblend.tolerance import Tolerance, ToleranceResult, assert_logits_close
from cacheblend.model import LayerwiseModel
from cacheblend.kv_store import KVStore
from cacheblend.chunker import Chunk, chunk_texts, fused_input_ids
from cacheblend.rope import apply_rope_shift
from cacheblend.precompute import precompute_chunk_kv, precompute_from_cache_prompt
from cacheblend.fusor import (
    fuse_full_recompute,
    fuse_full_reuse,
    fuse_selective,
    fuse_prefix_cache,
)
from cacheblend.hkvd import kv_deviation, select_top_k
from cacheblend.controller import LoadingController, StorageProfile

# Runners (mydata harness compatible) — Phase 5+
from cacheblend.runners import (
    FullRecomputeRunner,
    FullReuseRunner,
    PrefixCacheRunner,
    CacheBlendRunner,
)

__all__ = [
    "Tolerance",
    "ToleranceResult",
    "assert_logits_close",
    "LayerwiseModel",
    "KVStore",
    "Chunk",
    "chunk_texts",
    "fused_input_ids",
    "apply_rope_shift",
    "precompute_chunk_kv",
    "precompute_from_cache_prompt",
    "fuse_full_recompute",
    "fuse_full_reuse",
    "fuse_selective",
    "fuse_prefix_cache",
    "kv_deviation",
    "select_top_k",
    "LoadingController",
    "StorageProfile",
    "FullRecomputeRunner",
    "FullReuseRunner",
    "PrefixCacheRunner",
    "CacheBlendRunner",
]
