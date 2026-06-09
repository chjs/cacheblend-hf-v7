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

Env: CACHEBLEND_MODEL, CB_N (examples, default 100), CB_KVZIP_RATIOS ("0.5,0.3"),
     CB_RECOMP_RATIOS ("0.1,0.2"), CB_REDUCE (mean|max|ranknorm_max, default mean),
     CB_PROTECT_FIRST (default 0), CB_FORCE_CHUNK_STARTS (default 0).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
os.environ.setdefault("CACHEBLEND_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

from cacheblend import LayerwiseModel
from cacheblend.chunker import Chunk
from cacheblend.fusor import fuse_selective, fuse_full_recompute
from cacheblend.compress import CompressionBudget, token_prune, to_blend_inputs
from cacheblend.compress.kvzip import KVzipBackend, KVzipConfig
from utils import load_dataset, build_qa_prompt, compute_f1
from blend_musique_generic import _resolve_wrapper, _build_chunks, _greedy_decode, PREFIX_PROMPT, QUERY_PROMPT

MODEL = os.environ["CACHEBLEND_MODEL"]
N = int(os.environ.get("CB_N", "100"))
KVZIP_RATIOS = [float(x) for x in os.environ.get("CB_KVZIP_RATIOS", "0.5,0.3").split(",")]
RECOMP_RATIOS = [float(x) for x in os.environ.get("CB_RECOMP_RATIOS", "0.1,0.2").split(",")]
REDUCE = os.environ.get("CB_REDUCE", "mean")
PROTECT_FIRST = int(os.environ.get("CB_PROTECT_FIRST", "0"))
FORCE_CHUNK_STARTS = int(os.environ.get("CB_FORCE_CHUNK_STARTS", "0"))
CHECK_LAYER = int(os.environ.get("CACHEBLEND_CHECK_LAYER", "1"))


def main() -> int:
    print(f"[compress-kvzip] model={MODEL} N={N} kvzip={KVZIP_RATIOS} recomp={RECOMP_RATIOS} "
          f"reduce={REDUCE} protect_first={PROTECT_FIRST} force_chunk_starts={FORCE_CHUNK_STARTS}", flush=True)
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"        # isolated per-chunk compress (sink=0)
    lw = LayerwiseModel(MODEL, dtype="float16", device="cuda", attn_implementation="sdpa")
    model, tokenizer, device = lw.model, lw.tokenizer, lw.device
    user_open, assistant_open = _resolve_wrapper(MODEL, tokenizer)
    ds = load_dataset("inputs/musique_s.json")[:N]

    def decode(out):
        import time
        return _greedy_decode(model, tokenizer, out.logits, out.past_key_values, device, time.perf_counter())[0]

    f1: dict[str, list] = {"full_prefill_all": []}
    for r in KVZIP_RATIOS:
        f1[f"full_prefill_survivors@kv{r}"] = []
        f1[f"full_reuse_kvzip@kv{r}"] = []
        for rr in RECOMP_RATIOS:
            f1[f"compblend@kv{r}_rc{rr}"] = []

    for qi, ex in enumerate(ds):
        answers = ex["answers"]
        doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)
        ctexts = [user_open + PREFIX_PROMPT] + list(doc_prompts) + [q_prompt + assistant_open]
        orig = _build_chunks(tokenizer, ctexts)             # full token_ids (BOS on chunk0)
        doc_slice = slice(1, 1 + len(doc_prompts))

        # full_prefill_all (absolute ceiling) — token_ids only, no compression.
        out = fuse_full_recompute(lw, orig, return_layerwise_output=True)
        f1["full_prefill_all"].append(max(compute_f1(decode(out), a, tokenizer) for a in answers))
        del out
        if device.type == "cuda": torch.cuda.empty_cache()

        # score every chunk ONCE (isolated, sink=0) → full CompressedChunk + importance.
        cmp_full = {}
        for c in orig:
            ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
            cmp_full[c.chunk_id] = backend.score(ids).to(device)

        for r in KVZIP_RATIOS:
            budget = CompressionBudget(ratio=r)
            # prune ONLY doc chunks; prefix + query stay full.
            survivor_cmps = []
            for ci, c in enumerate(orig):
                cc = cmp_full[c.chunk_id]
                if doc_slice.start <= ci < doc_slice.stop:
                    cc = token_prune(cc, budget, reduce=REDUCE, protect_first=PROTECT_FIRST)
                survivor_cmps.append(cc)

            # full_prefill_survivors (post-compression ceiling) — joint full prefill of survivors.
            surv_chunks = [Chunk(text="", token_ids=list(cc.token_ids), chunk_id=cc.chunk_id)
                           for cc in survivor_cmps]
            out = fuse_full_recompute(lw, surv_chunks, return_layerwise_output=True)
            f1[f"full_prefill_survivors@kv{r}"].append(max(compute_f1(decode(out), a, tokenizer) for a in answers))
            del out
            if device.type == "cuda": torch.cuda.empty_cache()

            # blend inputs: (chunks, store) from compressed survivors (prefix/query full).
            blend_chunks, store = to_blend_inputs(survivor_cmps)

            # full_reuse_kvzip — survivors reused, query fresh (rr=0 + force_last).
            out = fuse_selective(lw, blend_chunks, store, recompute_ratio=0.0, check_layer=CHECK_LAYER,
                                 return_layerwise_output=True, force_last_chunk=True,
                                 force_chunk_starts=FORCE_CHUNK_STARTS)
            f1[f"full_reuse_kvzip@kv{r}"].append(max(compute_f1(decode(out), a, tokenizer) for a in answers))
            del out
            if device.type == "cuda": torch.cuda.empty_cache()

            # compblend — survivors + HKVD selective recompute, query fresh.
            for rr in RECOMP_RATIOS:
                out = fuse_selective(lw, blend_chunks, store, recompute_ratio=rr, check_layer=CHECK_LAYER,
                                     return_layerwise_output=True, force_last_chunk=True,
                                     force_chunk_starts=FORCE_CHUNK_STARTS)
                f1[f"compblend@kv{r}_rc{rr}"].append(max(compute_f1(decode(out), a, tokenizer) for a in answers))
                del out
                if device.type == "cuda": torch.cuda.empty_cache()
        if (qi + 1) % 10 == 0 or qi == 0:
            print(f"  [{qi+1}/{len(ds)}] full_all={np.mean(f1['full_prefill_all']):.3f}", flush=True)

    # ── aggregate + paired bootstrap CI ──
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
        ceil = means[f"full_prefill_survivors@kv{r}"]
        floor = means[f"full_reuse_kvzip@kv{r}"]
        print(f"  kv={r}: survivors_ceiling={ceil:.4f}  reuse_floor={floor:.4f}  "
              f"[compression_gap={means['full_prefill_all']-ceil:+.4f}, blending_gap={ceil-floor:+.4f}]", flush=True)
        for rr in RECOMP_RATIOS:
            cb = means[f"compblend@kv{r}_rc{rr}"]
            dv, lo, hi, sig = bootci(f"compblend@kv{r}_rc{rr}", f"full_reuse_kvzip@kv{r}")
            dc, lo2, hi2, sig2 = bootci(f"compblend@kv{r}_rc{rr}", f"full_prefill_survivors@kv{r}")
            print(f"    compblend rc={rr}: {cb:.4f}  | vs reuse_floor {dv:+.4f} CI[{lo:+.3f},{hi:+.3f}]{'★' if sig else ''}"
                  f"  | vs ceiling {dc:+.4f} CI[{lo2:+.3f},{hi2:+.3f}]{'★' if sig2 else ''}", flush=True)
    print("COMPRESS_KVZIP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
