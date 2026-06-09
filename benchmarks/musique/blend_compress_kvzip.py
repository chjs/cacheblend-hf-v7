"""CompBlend goal-1 make-or-break experiment: cacheblend + KVzip, ONLY-HKVD.

Composes KVzip (offline token-prune compression) with CacheBlend (HKVD selective
recompute) via the dep-light compress adapter + the EXISTING fuse_selective (no
gate, no fuse_selective_compblend). Per docs/COMPBLEND-GOAL1-PLAN.md the FIRST
experiment is the 4-arm make-or-break, NOT the full sink/selector grid.

Arms (force_last_chunk=True throughout — realistic serving, query prefilled fresh):
  full_prefill_all        recompute ALL original tokens (absolute ceiling).
  full_prefill_survivors  recompute the PRUNED survivor set jointly (post-compression
                          ceiling — the correct reference for the blending gap).
  full_reuse_kvzip        survivors reused, NO doc recompute (blending-gap floor).
  compblend               survivors + HKVD selective recompute (our method).
Gaps:  compression gap = full_prefill_all − full_prefill_survivors (KVzip's; HKVD
       can't touch). blending gap = full_prefill_survivors − full_reuse_kvzip (what
       HKVD must close).  Make-or-break: compblend closes the blending gap.

Two models (like compblend7): KVzip ModelKVzip (flash_attn, scoring) + a separate
sdpa LayerwiseModel (blend). Same weights; small attn-impl numeric mismatch accepted.

Env: CACHEBLEND_MODEL, CB_N (default 100), CB_KVZIP_RATIOS ("0.5,0.3"),
     CB_RECOMP_RATIOS ("0.1,0.2"), CB_REDUCE (mean|max|ranknorm_max, default mean),
     CB_PROTECT_FIRST (0), CB_FORCE_CHUNK_STARTS (0).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                                    # for inputs/musique_s.json
os.environ.setdefault("CACHEBLEND_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

from cacheblend import LayerwiseModel
from cacheblend.chunker import Chunk, _stable_id
from cacheblend.fusor import fuse_selective, fuse_full_recompute
from cacheblend.compress import CompressionBudget, token_prune, to_blend_inputs
from cacheblend.compress.kvzip import KVzipBackend, KVzipConfig
from utils import load_dataset, build_qa_prompt, compute_f1

MODEL = os.environ["CACHEBLEND_MODEL"]
N = int(os.environ.get("CB_N", "100"))
KVZIP_RATIOS = [float(x) for x in os.environ.get("CB_KVZIP_RATIOS", "0.5,0.3").split(",")]
RECOMP_RATIOS = [float(x) for x in os.environ.get("CB_RECOMP_RATIOS", "0.1,0.2").split(",")]
REDUCE = os.environ.get("CB_REDUCE", "mean")
PROTECT_FIRST = int(os.environ.get("CB_PROTECT_FIRST", "0"))
FORCE_CHUNK_STARTS = int(os.environ.get("CB_FORCE_CHUNK_STARTS", "0"))
CHECK_LAYER = int(os.environ.get("CACHEBLEND_CHECK_LAYER", "1"))
MAX_NEW_TOKENS = 32

# ── inlined prompt + chunk helpers (verbatim from blend_musique_generic.py) ──
PREFIX_PROMPT = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
QUERY_PROMPT = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
_WRAPPERS = {
    "mistral": ("[INST]", "[/INST]"),
    "llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama3":  ("<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "qwen":    ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"),
}


def _resolve_wrapper(model_id, tokenizer):
    mid = model_id.lower()
    for key, wrap in _WRAPPERS.items():
        if key in mid:
            return wrap
    sentinel = "\x00CONTENT\x00"
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentinel}], tokenize=False, add_generation_prompt=True)
    pre, post = templated.split(sentinel, 1)
    bos = tokenizer.bos_token or ""
    if bos and pre.startswith(bos):
        pre = pre[len(bos):]
    return pre, post


def _build_chunks(tokenizer, chunk_texts):
    bos = tokenizer.bos_token_id
    chunks = []
    for i, text in enumerate(chunk_texts):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if i == 0 and bos is not None:
            ids = [bos] + ids
        chunks.append(Chunk(text=text, token_ids=ids, chunk_id=_stable_id(text, ids)))
    return chunks


def _decode(model, tokenizer, logits, past_kv, device):
    eos = getattr(tokenizer, "eos_token_id", None)
    nxt = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    gen = [int(nxt.item())]
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS - 1):
            if eos is not None and gen[-1] == eos:
                break
            out = model(input_ids=nxt, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            gen.append(int(nxt.item()))
    return tokenizer.decode(gen, skip_special_tokens=True)


def main() -> int:
    print(f"[compress-kvzip] model={MODEL} N={N} kvzip={KVZIP_RATIOS} recomp={RECOMP_RATIOS} "
          f"reduce={REDUCE} protect_first={PROTECT_FIRST} force_chunk_starts={FORCE_CHUNK_STARTS}", flush=True)
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"        # isolated per-chunk compress (sink=0)
    lw = LayerwiseModel(MODEL, dtype="float16", device="cuda", attn_implementation="sdpa")
    model, tokenizer, device = lw.model, lw.tokenizer, lw.device
    user_open, assistant_open = _resolve_wrapper(MODEL, tokenizer)
    ds = load_dataset("inputs/musique_s.json")[:N]

    def dec(out):
        return _decode(model, tokenizer, out.logits, out.past_key_values, device)

    f1: dict[str, list] = {"full_prefill_all": []}
    for r in KVZIP_RATIOS:
        f1[f"full_reuse_kvzip@kv{r}"] = []
        for rr in RECOMP_RATIOS:
            f1[f"compblend@kv{r}_rc{rr}"] = []

    for qi, ex in enumerate(ds):
        answers = ex["answers"]
        doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)
        ctexts = [user_open + PREFIX_PROMPT] + list(doc_prompts) + [q_prompt + assistant_open]
        orig = _build_chunks(tokenizer, ctexts)
        doc_slice = slice(1, 1 + len(doc_prompts))

        out = fuse_full_recompute(lw, orig, return_layerwise_output=True)
        f1["full_prefill_all"].append(max(compute_f1(dec(out), a, tokenizer) for a in answers))
        del out
        if device.type == "cuda": torch.cuda.empty_cache()

        cmp_full = {}
        for c in orig:
            ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
            cmp_full[c.chunk_id] = backend.score(ids).to(device)

        for r in KVZIP_RATIOS:
            budget = CompressionBudget(ratio=r)
            survivor_cmps = []
            for ci, c in enumerate(orig):
                cc = cmp_full[c.chunk_id]
                if doc_slice.start <= ci < doc_slice.stop:
                    cc = token_prune(cc, budget, reduce=REDUCE, protect_first=PROTECT_FIRST)
                survivor_cmps.append(cc)

            blend_chunks, store = to_blend_inputs(survivor_cmps)

            out = fuse_selective(lw, blend_chunks, store, recompute_ratio=0.0, check_layer=CHECK_LAYER,
                                 return_layerwise_output=True, force_last_chunk=True,
                                 force_chunk_starts=FORCE_CHUNK_STARTS)
            f1[f"full_reuse_kvzip@kv{r}"].append(max(compute_f1(dec(out), a, tokenizer) for a in answers))
            del out
            if device.type == "cuda": torch.cuda.empty_cache()

            for rr in RECOMP_RATIOS:
                out = fuse_selective(lw, blend_chunks, store, recompute_ratio=rr, check_layer=CHECK_LAYER,
                                     return_layerwise_output=True, force_last_chunk=True,
                                     force_chunk_starts=FORCE_CHUNK_STARTS)
                f1[f"compblend@kv{r}_rc{rr}"].append(max(compute_f1(dec(out), a, tokenizer) for a in answers))
                del out
                if device.type == "cuda": torch.cuda.empty_cache()
        if (qi + 1) % 10 == 0 or qi == 0:
            print(f"  [{qi+1}/{len(ds)}] full_all={np.mean(f1['full_prefill_all']):.3f}", flush=True)

    means = {k: float(np.mean(v)) for k, v in f1.items() if v}
    rng = np.random.default_rng(0)

    def bootci(arm_key, ref_key, iters=2000):
        A, B = np.array(f1[arm_key]), np.array(f1[ref_key])
        d = A - B
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(iters)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        return float(d.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)

    print(f"\n{'='*70}\n== MEANS (N={len(ds)}) ==", flush=True)
    print(f"  full_prefill_all : {means['full_prefill_all']:.4f}", flush=True)
    for r in KVZIP_RATIOS:
        floor = means[f"full_reuse_kvzip@kv{r}"]
        print(f"  kv={r}: reuse_floor={floor:.4f}", flush=True)
        for rr in RECOMP_RATIOS:
            cb = means[f"compblend@kv{r}_rc{rr}"]
            dv, lo, hi, sig = bootci(f"compblend@kv{r}_rc{rr}", f"full_reuse_kvzip@kv{r}")
            print(f"    compblend rc={rr}: {cb:.4f}  | vs reuse_floor {dv:+.4f} CI[{lo:+.3f},{hi:+.3f}]{'★' if sig else ''}", flush=True)

    # ── JSON dump (axes + per-question lists + means) for plotting ──
    import json
    out_path = os.environ.get("CB_OUT", f"/root/results_{MODEL.split('/')[-1]}.json")
    with open(out_path, "w") as fh:
        json.dump({
            "model": MODEL, "N": len(ds),
            "kvzip_ratios": KVZIP_RATIOS, "recomp_ratios": RECOMP_RATIOS,
            "reduce": REDUCE, "protect_first": PROTECT_FIRST, "force_chunk_starts": FORCE_CHUNK_STARTS,
            "means": means, "f1": f1,
        }, fh)
    print(f"WROTE {out_path}", flush=True)
    print("COMPRESS_KVZIP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
