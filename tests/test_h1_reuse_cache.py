"""H1 regression test — reuse/prefix runners must decode against their OWN cache.

Background (docs/CODE-REVIEW-2026-06.md §H1): FullReuseRunner / PrefixCacheRunner
computed prefill logits via fuse_full_reuse / fuse_prefix_cache, then ran a
SECOND hook-less full forward and decoded against THAT cache = full-RECOMPUTE
KV (cross-attention present). So the "full reuse" baseline silently decoded
against a better-than-reuse cache, contaminating the comparison (and 2x cost).

The fix: use the cache fuse_*'s own forward_layerwise already builds (reused KV),
via return_layerwise_output=True. These tests prove:
  1. fuse_full_reuse(return_layerwise_output=True) returns a populated cache
     (refuting the old "KV cache wasn't saved through DynamicCache" comment).
  2. that reuse cache DIFFERS from the full-recompute cache the old code used
     (so the fix changes which cache decode sees, in the intended direction).

Needs transformers <4.53 (uses MistralModel._update_causal_mask via
LayerwiseModel). Runs on CPU with a tiny random Mistral (no download).
"""
import sys
from pathlib import Path

import torch
from transformers import MistralConfig, MistralForCausalLM
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cacheblend import LayerwiseModel  # noqa: E402
from cacheblend.chunker import Chunk, _stable_id  # noqa: E402
from cacheblend.kv_store import KVStore  # noqa: E402
from cacheblend.precompute import precompute_chunk_kv  # noqa: E402
from cacheblend.fusor import fuse_full_reuse  # noqa: E402


def _tiny_lw():
    torch.manual_seed(0)
    cfg = MistralConfig(
        vocab_size=320, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512,
    )
    model = MistralForCausalLM(cfg).eval()
    # Wire a LayerwiseModel view the same way runners._ensure_lw_and_store does.
    lw = LayerwiseModel.__new__(LayerwiseModel)
    lw.model = model
    lw.tokenizer = None
    lw.device = torch.device("cpu")
    lw.dtype = next(model.parameters()).dtype
    lw._inner = model.model
    lw.num_layers = len(lw._inner.layers)
    lw._pre_rope_k = {}
    lw._hook_handles = []
    lw._install_k_proj_hooks()
    return lw


def _chunks():
    specs = [[3, 7, 11, 5, 9], [21, 4, 8, 19], [40, 2, 17, 6, 1, 30]]
    return [Chunk(text=f"c{i}", token_ids=t, chunk_id=_stable_id(f"c{i}", t))
            for i, t in enumerate(specs)]


def test_full_reuse_returns_populated_cache():
    """fuse_full_reuse(return_layerwise_output=True) gives a full-length cache."""
    lw = _tiny_lw()
    chunks = _chunks()
    total = sum(len(c.token_ids) for c in chunks)
    store = KVStore()
    for c in chunks:
        K, V = precompute_chunk_kv(lw, c)
        store.put(c.chunk_id, K, V)

    out = fuse_full_reuse(lw, chunks, store, return_layerwise_output=True)
    assert out.past_key_values is not None, "no cache returned"
    assert out.past_key_values.get_seq_length() == total, (
        f"reuse cache seq_len={out.past_key_values.get_seq_length()} != {total} "
        f"(old comment 'KV cache wasn't saved' was wrong)"
    )


def test_reuse_cache_differs_from_full_recompute_cache():
    """The reuse cache (fix) must differ from the full-recompute cache (old bug).

    The old code decoded against a hook-less full forward = full-recompute KV.
    Prove that cache is NOT the same as the reuse cache, i.e. the old baseline
    was contaminated.
    """
    lw = _tiny_lw()
    chunks = _chunks()
    store = KVStore()
    for c in chunks:
        K, V = precompute_chunk_kv(lw, c)
        store.put(c.chunk_id, K, V)

    # NEW (fix): reuse cache.
    reuse_out = fuse_full_reuse(lw, chunks, store, return_layerwise_output=True)
    reuse_cache = reuse_out.past_key_values

    # OLD (bug): hook-less full forward = full-recompute cache.
    flat = [t for c in chunks for t in c.token_ids]
    input_ids = torch.tensor([flat], dtype=torch.long)
    with torch.inference_mode():
        full_out = lw.model(input_ids=input_ids, use_cache=True)
    full_cache = full_out.past_key_values

    # Compare K at the last layer (cross-attention effect is largest in deep
    # layers): reuse != full-recompute.
    li = lw.num_layers - 1
    k_reuse = reuse_cache.key_cache[li] if hasattr(reuse_cache, "key_cache") else reuse_cache[li][0]
    k_full = full_cache.key_cache[li] if hasattr(full_cache, "key_cache") else full_cache[li][0]
    max_diff = (k_reuse.float() - k_full.float()).abs().max().item()
    assert max_diff > 1e-3, (
        f"reuse and full-recompute caches are identical (max_diff={max_diff:.2e}); "
        f"expected them to differ — the H1 fix would be a no-op otherwise"
    )
    print(f"[H1] last-layer K  max|reuse - full_recompute| = {max_diff:.4f} "
          f"(>0 -> old code decoded against the WRONG cache)")


if __name__ == "__main__":
    import transformers
    print(f"transformers {transformers.__version__}, torch {torch.__version__}")
    test_full_reuse_returns_populated_cache()
    print("PASS test_full_reuse_returns_populated_cache")
    test_reuse_cache_differs_from_full_recompute_cache()
    print("PASS test_reuse_cache_differs_from_full_recompute_cache")
    print("\nALL H1 TESTS PASSED")
