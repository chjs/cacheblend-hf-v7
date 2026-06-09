"""H2 regression test — every runner must consume an identical token sequence.

Background (docs/CODE-REVIEW-2026-06.md §H2): FullRecomputeRunner tokenized the
full prompt STRING with add_special_tokens=True (BOS prepended, single-string
tokenization), while the reuse/selective/prefix runners built chunks with
add_special_tokens=False and NO BOS, then concatenated per-chunk token ids. So
the baseline ran on a different token sequence (extra BOS + different chunk-
boundary BPE) than CacheBlend — an apples-to-oranges comparison.

Fix: all runners build chunks via _build_chunks (BOS on chunk 0 only) and
forward fused_input_ids(chunks). These tests verify the shared sequence.

Tokenization-only (no model forward) — runs on any transformers/torch. Uses a
small public tokenizer (hf-internal-testing/llama-tokenizer).
"""
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cacheblend.chunker import fused_input_ids  # noqa: E402
from cacheblend.runners import FullRecomputeRunner, CacheBlendRunner, FullReuseRunner  # noqa: E402

TOK = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
SYSTEM = "You are a helpful assistant. Answer within 5 words."
DOCS = ["Paris is the capital of France.", "The Eiffel Tower is in Paris."]
QUESTION = "What is the capital of France?"


def _prep(runner_cls, **kw):
    r = runner_cls(model=None, tokenizer=TOK, **kw)
    r.prepare(SYSTEM, DOCS, QUESTION)
    return r


def test_all_runners_share_one_token_sequence():
    fr = _prep(FullRecomputeRunner)
    cb = _prep(CacheBlendRunner, recompute_ratio=0.15, check_layer=1)
    reuse = _prep(FullReuseRunner)

    seqs = {name: fused_input_ids(r._build_chunks()).tolist()
            for name, r in [("full_recompute", fr), ("cacheblend", cb), ("full_reuse", reuse)]}
    base = seqs["full_recompute"]
    for name, s in seqs.items():
        assert s == base, f"{name} token sequence differs from full_recompute baseline"
    print(f"[H2] all runners share one sequence (len={len(base[0])})")


def test_bos_only_on_chunk0():
    cb = _prep(CacheBlendRunner, recompute_ratio=0.15, check_layer=1)
    chunks = cb._build_chunks()
    bos = TOK.bos_token_id
    assert chunks[0].token_ids[0] == bos, "chunk 0 must start with BOS"
    for i, c in enumerate(chunks[1:], start=1):
        assert c.token_ids[0] != bos, f"chunk {i} must NOT start with BOS"
    print(f"[H2] BOS(={bos}) on chunk0 only; {len(chunks)} chunks")


def test_old_fullrecompute_path_differed():
    """Document the bug: the OLD full-string tokenization != the fused sequence."""
    fr = _prep(FullRecomputeRunner)
    # OLD path: tokenize the full prompt string with default special tokens.
    old_ids = TOK(fr._build_prompt_text(), return_tensors="pt")["input_ids"][0].tolist()
    # NEW path: fused chunk ids.
    new_ids = fused_input_ids(fr._build_chunks())[0].tolist()
    # They should differ (BOS placement and/or chunk-boundary BPE) — proving the
    # old baseline ran on a different sequence than CacheBlend.
    same = old_ids == new_ids
    print(f"[H2] old_len={len(old_ids)} new_len={len(new_ids)} identical={same}")
    assert not same, "old full-string tokenization unexpectedly equals fused seq"


if __name__ == "__main__":
    import transformers
    print(f"transformers {transformers.__version__}")
    test_all_runners_share_one_token_sequence()
    print("PASS test_all_runners_share_one_token_sequence")
    test_bos_only_on_chunk0()
    print("PASS test_bos_only_on_chunk0")
    test_old_fullrecompute_path_differed()
    print("PASS test_old_fullrecompute_path_differed")
    print("\nALL H2 TESTS PASSED")
