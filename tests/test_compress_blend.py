"""CompBlend goal-1 blend-side test (no KVzip / no flash_attn).

Validates the compressed-KV blend path end-to-end on a tiny Mistral:
  CompressedChunk  --token_prune-->  shorter chunk  --to_blend_inputs-->
  (Chunk list, KVStore)  --EXISTING fuse_selective (only-HKVD)-->  decode.

A "compressed chunk" here is built from precompute_chunk_kv's standalone pre-RoPE
K/V (exactly what a backend captures) + synthetic importance. This exercises the
glue + reuse of fuse_selective without needing the real KVzip backend. Needs
transformers <4.53 (fuse_selective uses _update_causal_mask). Runs on CPU.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cacheblend import LayerwiseModel  # noqa: E402
from cacheblend.chunker import Chunk, _stable_id  # noqa: E402
from cacheblend.kv_store import KVStore  # noqa: E402
from cacheblend.precompute import precompute_chunk_kv  # noqa: E402
from cacheblend.fusor import fuse_selective  # noqa: E402
from cacheblend.compress import (  # noqa: E402
    CompressedChunk, CompressionBudget, token_prune, to_blend_inputs,
)


def _tiny_lw():
    from transformers import MistralConfig, MistralForCausalLM
    torch.manual_seed(0)
    cfg = MistralConfig(vocab_size=320, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    model = MistralForCausalLM(cfg).eval()
    lw = LayerwiseModel.__new__(LayerwiseModel)
    lw.model = model; lw.tokenizer = None; lw.device = torch.device("cpu")
    lw.dtype = next(model.parameters()).dtype
    lw._inner = model.model; lw.num_layers = len(lw._inner.layers)
    lw._pre_rope_k = {}; lw._hook_handles = []; lw._install_k_proj_hooks()
    return lw


def _scored_chunk(lw, token_ids, *, seed=0):
    """Build a FULL (unpruned) CompressedChunk: standalone pre-RoPE K/V + synthetic importance."""
    c = Chunk(text="", token_ids=token_ids, chunk_id=_stable_id("c", token_ids))
    K, V = precompute_chunk_kv(lw, c)               # exactly what a backend captures
    L, Hkv, T = lw.num_layers, lw._inner.layers[0].self_attn.config.num_key_value_heads, len(token_ids)
    g = torch.Generator().manual_seed(seed)
    importance = torch.rand(L, Hkv, T, generator=g)
    return CompressedChunk(
        algo_id="fake", chunk_id=c.chunk_id, model_id="tiny", tokenizer_id="tiny",
        token_ids=list(token_ids), key_cache=K, value_cache=V, importance=importance,
    )


def test_token_prune_shape_and_budget():
    lw = _tiny_lw()
    cc = _scored_chunk(lw, [3, 7, 11, 5, 9, 21, 4, 8, 19, 40])   # 10 tokens
    assert cc.compression_rate == 0.0 and cc.chunk_len == 10
    pruned = token_prune(cc, CompressionBudget(ratio=0.5))
    assert pruned.chunk_len == 5, pruned.chunk_len
    assert len(pruned.token_ids) == 5
    assert all(k.shape[1] == 5 for k in pruned.key_cache)
    assert all(v.shape[1] == 5 for v in pruned.value_cache)
    assert pruned.importance.shape[2] == 5
    assert abs(pruned.compression_rate - 0.5) < 1e-9
    # survivors are the top-5 by mean importance, in ascending position order
    imp_tok = cc.importance.mean(dim=(0, 1))
    want = sorted(torch.topk(imp_tok, 5).indices.tolist())
    assert [cc.token_ids[i] for i in want] == pruned.token_ids
    print(f"[compress] token_prune 10→{pruned.chunk_len} rate={pruned.compression_rate:.2f} reduce=mean OK")


def test_token_prune_reduce_max():
    lw = _tiny_lw()
    cc = _scored_chunk(lw, list(range(2, 22)))                   # 20 tokens
    p = token_prune(cc, CompressionBudget(ratio=0.3), reduce="max")
    assert p.chunk_len == 6   # round(20*0.3)=6
    print(f"[compress] token_prune reduce=max 20→{p.chunk_len} OK")


def test_compressed_blend_end_to_end():
    """Pruned compressed docs + fresh query → EXISTING fuse_selective (only-HKVD)."""
    lw = _tiny_lw()
    # two doc chunks (compressed) + a query chunk (fresh).
    doc0 = _scored_chunk(lw, [3, 7, 11, 5, 9, 21, 4, 8], seed=1)
    doc1 = _scored_chunk(lw, [40, 2, 17, 6, 1, 30, 12], seed=2)
    pruned = [token_prune(doc0, CompressionBudget(ratio=0.5)),
              token_prune(doc1, CompressionBudget(ratio=0.5))]

    # compressed docs → (chunks, store)
    blend_chunks, store = to_blend_inputs(pruned)
    # append the FRESH query chunk (precompute its standalone KV; force_last keeps it fresh)
    query = Chunk(text="", token_ids=[100, 5, 33, 7], chunk_id=_stable_id("q", [100, 5, 33, 7]))
    qK, qV = precompute_chunk_kv(lw, query)
    store.put(query.chunk_id, qK, qV)
    chunks = blend_chunks + [query]

    total = sum(len(c.token_ids) for c in chunks)
    qstart = total - len(query.token_ids)

    out, top = fuse_selective(lw, chunks, store, recompute_ratio=0.3, check_layer=1,
                              return_layerwise_output=True, return_hkvd_indices=True,
                              force_last_chunk=True)
    # cache populated to the pruned+query length; query fully fresh; decode valid.
    assert out.past_key_values.get_seq_length() == total, (out.past_key_values.get_seq_length(), total)
    qset = set(top.tolist())
    assert all(p in qset for p in range(qstart, total)), "force_last_chunk: query not all fresh"
    assert torch.isfinite(out.logits[0, -1]).all()
    # HKVD selected at least one DOC token to recompute (blend actually happened)
    doc_selected = [p for p in top.tolist() if p < qstart]
    assert len(doc_selected) >= 1, "no doc token recomputed — HKVD blend inert"
    print(f"[compress] e2e: chunks(pruned docs+query) total={total} qstart={qstart} "
          f"top={len(qset)} doc_recomputed={len(doc_selected)} OK")


if __name__ == "__main__":
    test_token_prune_shape_and_budget(); print("PASS prune shape")
    test_token_prune_reduce_max(); print("PASS prune max")
    test_compressed_blend_end_to_end(); print("PASS e2e")
    print("\nALL COMPRESS-BLEND TESTS PASSED")
