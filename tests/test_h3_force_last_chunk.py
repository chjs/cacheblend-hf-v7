"""H3 regression test — query (last chunk) treated as a fresh live suffix.

Background (docs/CODE-REVIEW-2026-06.md §H3): the legacy fuse_selective treats
the query chunk as just another reusable chunk — only the single last position
is forced fresh, the rest of the query is reused/HKVD-selected like a document.
That is the harness assumption behind the realistic-serving artifact (the query
should ALWAYS be prefilled fresh against the blended document KV, never reused).

The fix adds force_last_chunk: when True (multi-chunk), the WHOLE last chunk is
forced fresh and recompute_ratio budgets only the document context (query tokens
counted separately): recompute_k = n_forced + int((total - n_forced) * ratio).

Tests:
  A. force_last_chunk=False selection == legacy select_top_k + force-include-last
     (bit-identical → no regression).            [model-free]
  B. force_last_chunk=True forces the WHOLE query chunk; doc budget is separate. [model-free]
  C. end-to-end (tiny Mistral, 4.51.3): force_last_chunk=True recomputes every
     query position; cache populates; decode logits valid.
  D. ratio==0 + force_last_chunk: query stays fresh, docs fully reused (the
     boundary-shortcut gating fix).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cacheblend.hkvd import select_top_k, select_top_k_masked  # noqa: E402


def _legacy_select(deviations, ratio):
    """The pre-H3 selection: select_top_k(ratio) then force-include the last pos."""
    top = select_top_k(deviations, ratio)
    last = deviations.shape[0] - 1
    if last not in top.tolist():
        sel = deviations[top]
        drop = top[sel.argmin()].item()
        top = torch.tensor(sorted([i for i in top.tolist() if i != drop] + [last]),
                           dtype=top.dtype)
    return sorted(top.tolist())


def test_A_force_last_chunk_false_bit_identical_to_legacy():
    torch.manual_seed(0)
    for ratio in (0.1, 0.15, 0.3, 0.5):
        N = 47
        dev = torch.rand(N)
        legacy = _legacy_select(dev, ratio)
        forced = torch.zeros(N, dtype=torch.bool)
        forced[-1] = True
        rk = max(int(N * ratio), 1)
        new = sorted(select_top_k_masked(dev, rk, forced).tolist())
        assert new == legacy, f"ratio={ratio}: new {new} != legacy {legacy}"
    print("[H3] force_last_chunk=False selection == legacy (bit-identical) across ratios")


def test_B_force_last_chunk_true_forces_whole_query():
    torch.manual_seed(1)
    N, query_start, ratio = 50, 38, 0.2   # query = positions 38..49 (12 tokens)
    dev = torch.rand(N)
    forced = torch.zeros(N, dtype=torch.bool)
    forced[-1] = True
    forced[query_start:] = True
    n_forced = int(forced.sum().item())                       # 12
    rk = n_forced + int((N - n_forced) * ratio)               # 12 + int(38*0.2)=12+7=19
    sel = set(select_top_k_masked(dev, rk, forced).tolist())
    for p in range(query_start, N):
        assert p in sel, f"query pos {p} not forced"
    assert len(sel) == rk, f"total {len(sel)} != recompute_k {rk}"
    doc_sel = [p for p in sel if p < query_start]
    assert len(doc_sel) == rk - n_forced, f"doc budget {len(doc_sel)} != {rk - n_forced}"
    print(f"[H3] force_last_chunk=True: all {N-query_start} query forced + "
          f"{len(doc_sel)} doc HKVD (= int(N_doc*ratio)); total={len(sel)}")


# ── C/D need a model forward (transformers <4.53 for _update_causal_mask) ────

def _tiny_lw():
    from transformers import MistralConfig, MistralForCausalLM
    from cacheblend import LayerwiseModel
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


def _chunks():
    from cacheblend.chunker import Chunk, _stable_id
    specs = [[3, 7, 11, 5, 9], [21, 4, 8, 19], [40, 2, 17, 6, 1, 30]]  # 2 docs + query
    return [Chunk(text=f"c{i}", token_ids=t, chunk_id=_stable_id(f"c{i}", t))
            for i, t in enumerate(specs)]


def _store(lw, chunks):
    from cacheblend.kv_store import KVStore
    from cacheblend.precompute import precompute_chunk_kv
    st = KVStore()
    for c in chunks:
        K, V = precompute_chunk_kv(lw, c)
        st.put(c.chunk_id, K, V)
    return st


def test_C_e2e_force_last_chunk_true_recomputes_all_query():
    from cacheblend.fusor import fuse_selective
    lw = _tiny_lw(); chunks = _chunks(); store = _store(lw, chunks)
    total = sum(len(c.token_ids) for c in chunks)
    qstart = total - len(chunks[-1].token_ids)

    out_t, top_t = fuse_selective(lw, chunks, store, recompute_ratio=0.2, check_layer=1,
                                  return_layerwise_output=True, return_hkvd_indices=True,
                                  force_last_chunk=True)
    st = set(top_t.tolist())
    for p in range(qstart, total):
        assert p in st, f"force_last_chunk=True: query pos {p} not recomputed"
    assert out_t.past_key_values.get_seq_length() == total, "cache not full-length"
    assert torch.isfinite(out_t.logits[0, -1]).all(), "last-pos logits not finite"

    out_f, top_f = fuse_selective(lw, chunks, store, recompute_ratio=0.2, check_layer=1,
                                  return_layerwise_output=True, return_hkvd_indices=True,
                                  force_last_chunk=False)
    sf = set(top_f.tolist())
    q_true = sum(1 for p in range(qstart, total) if p in st)
    q_false = sum(1 for p in range(qstart, total) if p in sf)
    assert q_true == total - qstart and q_true >= q_false
    print(f"[H3] e2e query recomputed: True={q_true}/{total-qstart}  False={q_false}/{total-qstart}")


def test_D_ratio0_force_last_chunk_only_query_fresh():
    from cacheblend.fusor import fuse_selective
    lw = _tiny_lw(); chunks = _chunks(); store = _store(lw, chunks)
    total = sum(len(c.token_ids) for c in chunks)
    qstart = total - len(chunks[-1].token_ids)
    # ratio==0 must NOT dispatch to full_reuse when force_last_chunk (gating fix).
    out, top = fuse_selective(lw, chunks, store, recompute_ratio=0.0, check_layer=1,
                              return_layerwise_output=True, return_hkvd_indices=True,
                              force_last_chunk=True)
    s = set(top.tolist())
    assert s == set(range(qstart, total)), (
        f"ratio=0+force_last_chunk should recompute ONLY query {sorted(range(qstart,total))}, "
        f"got {sorted(s)}"
    )
    assert out.past_key_values.get_seq_length() == total
    print(f"[H3] ratio=0 + force_last_chunk: only query fresh ({len(s)} pos), docs reused")


if __name__ == "__main__":
    test_A_force_last_chunk_false_bit_identical_to_legacy()
    print("PASS A")
    test_B_force_last_chunk_true_forces_whole_query()
    print("PASS B")
    test_C_e2e_force_last_chunk_true_recomputes_all_query()
    print("PASS C")
    test_D_ratio0_force_last_chunk_only_query_fresh()
    print("PASS D")
    print("\nALL H3 TESTS PASSED")
