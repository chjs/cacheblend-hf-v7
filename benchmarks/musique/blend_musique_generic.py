"""Generic musique workload — direct cacheblend API, NO vllm shim.

Reproduces the YaoJiayi/CacheBlend musique experiment for ANY HF instruction
model, calling our cacheblend implementation directly:

    precompute_chunk_kv  →  fuse_selective   (CacheBlend / "cache")
    fuse_full_recompute                       ("full prefill")

This is the shim-free counterpart of `blend_musique.py`. `blend_musique.py`
is the YaoJiayi verbatim original — it targets a custom vLLM fork and can
only run through `_shim/vllm/`. This file is OURS: it skips the vLLM facade
entirely and is a standalone script.

Why shim-free matters — tokenization consistency
-------------------------------------------------
The shim path passed prompt *strings* to a fake `vllm.LLM.generate`, which
re-encoded each chunk (`tokenizer.encode`) and, for the full-prefill path,
re-encoded the *joined* prompt. Because BPE is not associative
(`encode(A)+encode(B) != encode(A+B)`), the CacheBlend path and the
full-prefill path ran on slightly different token sequences at chunk
boundaries — so even at `recompute_ratio=1.0` the two diverged on a few
examples.

Here both paths consume the SAME `fused_input_ids(chunks)`:
  - chunks are tokenized ONCE (`tokenizer(text, add_special_tokens=False)`),
    BOS prepended only to chunk 0
  - `fuse_selective`     forwards `fused_input_ids(chunks)`
  - `fuse_full_recompute` forwards the identical `fused_input_ids(chunks)`
At `recompute_ratio=1.0`, `fuse_selective` dispatches to `fuse_full_recompute`
on the same chunks → the two are bit-identical. Tokenization is no longer a
confound; the only variable is the selective-recompute algorithm.

Model-agnostic
--------------
The only model-dependent part is the instruction wrapper (Mistral `[INST]`
vs Llama `<|start_header_id|>` vs Qwen `<|im_start|>`). It is resolved from a
small per-family table, or — for an unknown family — from the tokenizer's
chat template. New model = `CACHEBLEND_MODEL=...`, no new file.

Env vars:
    CACHEBLEND_MODEL         HF model id (default mistralai/Mistral-7B-Instruct-v0.2)
    CACHEBLEND_DTYPE         model dtype (default float16)
    CACHEBLEND_ATTN_IMPL     attn_implementation (default sdpa; eager OOMs at ~7K tokens)
    CACHEBLEND_CHECK_LAYER   check_layer for fuse_selective (default 1)
    CACHEBLEND_RECOMP_RATIO  recompute ratio (default 0.15)
    CACHEBLEND_MUSIQUE_N     run only the first N examples (default: all 150)

Run (standalone — no runner, no shim):
    CACHEBLEND_MODEL=meta-llama/Llama-3.1-8B-Instruct \
        python benchmarks/musique/blend_musique_generic.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Standalone bootstrap: resolve `utils` import + `inputs/` relative path ──
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
os.chdir(_HERE)

from cacheblend import LayerwiseModel
from cacheblend.chunker import Chunk, _stable_id
from cacheblend.kv_store import KVStore
from cacheblend.precompute import precompute_chunk_kv
from cacheblend.fusor import fuse_selective, fuse_full_recompute
from utils import load_dataset, build_qa_prompt, compute_f1


MODEL = os.environ.get("CACHEBLEND_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
DTYPE = os.environ.get("CACHEBLEND_DTYPE", "float16")
ATTN_IMPL = os.environ.get("CACHEBLEND_ATTN_IMPL", "sdpa")
CHECK_LAYER = int(os.environ.get("CACHEBLEND_CHECK_LAYER", "1"))
RECOMP_RATIO = float(os.environ.get("CACHEBLEND_RECOMP_RATIO", "0.15"))
MAX_NEW_TOKENS = 32

# Instruction prompts — VERBATIM from blend_musique.py (experiment definition).
PREFIX_PROMPT = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
QUERY_PROMPT = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

# Per-family instruction wrapper: (user_turn_open, assistant_turn_open).
# Single user turn, no system message — matches blend_musique.py, which places
# prefix_prompt directly after [INST]. BOS is NOT included here; it is
# prepended to chunk 0's token_ids in _build_chunks.
_WRAPPERS = {
    "mistral": ("[INST]", "[/INST]"),
    "llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama3":  ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "qwen":    ("<|im_start|>user\n",
                "<|im_end|>\n<|im_start|>assistant\n"),
}


def _resolve_wrapper(model_id: str, tokenizer):
    """Return (user_open, assistant_open) for the model — table or chat template."""
    mid = model_id.lower()
    for key, wrap in _WRAPPERS.items():
        if key in mid:
            return wrap
    sentinel = "\x00CONTENT\x00"
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False, add_generation_prompt=True,
    )
    if sentinel not in templated:
        raise RuntimeError(
            f"could not derive instruction wrapper for {model_id!r}; "
            f"add an entry to _WRAPPERS in blend_musique_generic.py"
        )
    pre, post = templated.split(sentinel, 1)
    bos = tokenizer.bos_token or ""
    if bos and pre.startswith(bos):
        pre = pre[len(bos):]
    return pre, post


def _build_chunks(tokenizer, chunk_texts: list[str]) -> list[Chunk]:
    """Tokenize each chunk text ONCE. BOS is prepended only to chunk 0 (it sits
    at fused position 0); chunks 1..N carry no BOS. `fused_input_ids(chunks)`
    then yields [BOS, chunk0, chunk1, ..., chunkN] — the single token sequence
    that BOTH fuse_selective and fuse_full_recompute consume.
    """
    bos = tokenizer.bos_token_id
    chunks: list[Chunk] = []
    for i, text in enumerate(chunk_texts):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if i == 0 and bos is not None:
            ids = [bos] + ids
        chunks.append(Chunk(text=text, token_ids=ids, chunk_id=_stable_id(text, ids)))
    return chunks


def _greedy_decode(model, tokenizer, prefill_logits, past_kv, device, t_prefill_start):
    """Greedy decode MAX_NEW_TOKENS from the prefill output. Returns (text, ttft).

    ttft = time from prefill start to the first decoded token.
    """
    eos = getattr(tokenizer, "eos_token_id", None)
    next_id = prefill_logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_first = time.perf_counter()
    generated = [int(next_id.item())]
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS - 1):
            if eos is not None and generated[-1] == eos:
                break
            out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            generated.append(int(next_id.item()))
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, t_first - t_prefill_start


def main() -> int:
    print(f"[blend_musique_generic] model={MODEL} dtype={DTYPE} attn={ATTN_IMPL} "
          f"check_layer={CHECK_LAYER} recomp_ratio={RECOMP_RATIO}", flush=True)

    lw = LayerwiseModel(MODEL, dtype=DTYPE, attn_implementation=ATTN_IMPL)
    tokenizer, model, device = lw.tokenizer, lw.model, lw.device

    user_open, assistant_open = _resolve_wrapper(MODEL, tokenizer)
    print(f"[blend_musique_generic] user_open={user_open!r}", flush=True)
    print(f"[blend_musique_generic] assistant_open={assistant_open!r}", flush=True)

    eval_dataset = load_dataset("inputs/musique_s.json")
    n_env = os.environ.get("CACHEBLEND_MUSIQUE_N")
    if n_env:
        eval_dataset = eval_dataset[:int(n_env)]
        print(f"[blend_musique_generic] CACHEBLEND_MUSIQUE_N={n_env} → "
              f"first {len(eval_dataset)} examples", flush=True)
    print("─" * 70, flush=True)

    ttft_blend, ttft_full, f1_blend, f1_full = [], [], [], []

    for ex in eval_dataset:
        answers = ex["answers"]
        doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)

        # Chunk layout: [user_open + prefix] [doc1]..[docN] [query + assistant_open]
        chunk_texts = [user_open + PREFIX_PROMPT]
        chunk_texts += list(doc_prompts)
        chunk_texts += [q_prompt + assistant_open]
        chunks = _build_chunks(tokenizer, chunk_texts)

        # Per-chunk KV precompute (the "collect" phase — amortized, not in TTFT).
        kv_store = KVStore()
        for c in chunks:
            K, V = precompute_chunk_kv(lw, c)
            kv_store.put(c.chunk_id, K, V)

        # ── CacheBlend (selective recompute) ───────────────────────────────
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fuse_selective(
            lw, chunks, kv_store,
            recompute_ratio=RECOMP_RATIO, check_layer=CHECK_LAYER,
            return_layerwise_output=True,
        )
        res, ttft = _greedy_decode(model, tokenizer, out.logits, out.past_key_values, device, t0)
        print(f"Cached generation: {res}")
        print(f"TTFT with cache: {ttft}")
        ttft_blend.append(ttft)
        f1_blend.append(max(compute_f1(res, a, tokenizer) for a in answers))

        # ── Full prefill — SAME fused_input_ids(chunks) as fuse_selective ──
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fuse_full_recompute(lw, chunks, return_layerwise_output=True)
        res, ttft = _greedy_decode(model, tokenizer, out.logits, out.past_key_values, device, t0)
        print(f"Normal generation: {res}")
        print(f"TTFT with full prefill: {ttft}")
        ttft_full.append(ttft)
        f1_full.append(max(compute_f1(res, a, tokenizer) for a in answers))
        print("------------", flush=True)

    print("---------------Result Summary---------------------")
    print(f"Model: {MODEL}")
    print(f"TTFT with cache: {np.mean(ttft_blend)}")
    print(f"TTFT with full prefill: {np.mean(ttft_full)}")
    print(f"F1 with cache: {np.mean(f1_blend)}")
    print(f"F1 with full prefill: {np.mean(f1_full)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
