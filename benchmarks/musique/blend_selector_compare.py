"""THE selector experiment: HKVD vs KVzip-importance (vs random / position controls).

Everything held equal — same compressed chunks, same recompute budget over the
same candidate pool (doc chunks only), same forced fresh query — ONLY the ranking
signal for recompute selection differs:

  hkvd      deviation ||K_fresh − K_cached||² at check_layer (CacheBlend §4)
  imp       KVzip importance: per-chunk [L,H_kv,T] → head/layer reduction
            (CB_SEL_REDUCE) → within-chunk percentile rank (CB_CHUNK_NORM=rank).
            Attention-sink signal (high importance at chunk starts) is
            deliberately PRESERVED — it is a real signal, and recomputing
            doc-beginning sinks is exactly the EPIC/LegoLink correction.
  random    fixed-seed random scores — signal floor (proves selectors carry info)
  position  earlier-in-chunk-first — pure chunk-start/sink heuristic; measures
            how much of imp's value is just "recompute the doc-beginning sinks"
  gated     Gated HKVD (the paper's selector): importance gates the candidate
            set to the top CB_GATE_PCT (default 0.7) of doc positions, HKVD
            picks within the gate. Budget equalized to the other arms.
  grad_imp  gradual filtering (paper §4.3) driven by OFFLINE per-layer
            importance: stage-0 set chosen by future-importance-mass, then
            narrowed each layer by that layer's remaining future mass. Equal
            token-layer FLOPs vs flat arms (Σ budgets = flat_k × n_stages).
  grad_hybrid  "HKVD decides WHO, importance decides HOW DEEP": stage-0 set by
            HKVD deviation (oversampled), narrowing by per-layer future mass.

Fairness guards (the realistic-serving-reversal lessons):
  * force_last_chunk=True everywhere — query always fresh; budget counts docs only
    (the old query-budget artifact cannot recur).
  * eligible_mask = doc positions for ALL selector arms — the exact-match prefix
    is not stale; HKVD knows this (deviation≈0) but importance does not, so
    allowing prefix would handicap imp unfairly. Equal budget over equal pool.
  * doc pruning (when kv<1) is FIXED to reduce=mean for all arms — arms differ
    only in selection, never in compression.

Diagnostics per cell: Jaccard overlap of hkvd-vs-imp selections; fraction of
selected doc tokens within the first 4 positions of their chunk (sink share).
Primary pre-registered comparison: imp − hkvd per (kv, rc), paired bootstrap 95% CI.
History (realistic-serving reversal) predicts hkvd ≥ imp; this clean re-test on
the new implementation is definitive either way.

Env: CACHEBLEND_MODEL, CB_N (150), CB_KVZIP_RATIOS ("1.0,0.5"),
     CB_RECOMP_RATIOS ("0.15"), CB_SEL_REDUCE (mean|max|ranknorm_max, def mean),
     CB_CHUNK_NORM (rank|raw, def rank), CB_OUT (def /tmp/selcmp_<model>.json).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
# Avoid shadowing KVzip's top-level `utils` package.
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != HERE]
os.chdir(HERE)
os.environ.setdefault("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

from cacheblend import LayerwiseModel
from cacheblend.chunker import Chunk, _stable_id
from cacheblend.fusor import fuse_selective, fuse_full_recompute
from cacheblend.compress import CompressionBudget, token_prune, reduce_importance, to_blend_inputs
from cacheblend.compress.kvzip import KVzipBackend, KVzipConfig

import importlib.util as _ilu
_us = _ilu.spec_from_file_location("_cbq_utils", str(HERE / "utils.py"))
_um = _ilu.module_from_spec(_us); _us.loader.exec_module(_um)
load_dataset, build_qa_prompt, compute_f1 = _um.load_dataset, _um.build_qa_prompt, _um.compute_f1

MODEL = os.environ["CACHEBLEND_MODEL"]
N = int(os.environ.get("CB_N", "150"))
KVZIP_RATIOS = [float(x) for x in os.environ.get("CB_KVZIP_RATIOS", "1.0,0.5").split(",")]
RECOMP_RATIOS = [float(x) for x in os.environ.get("CB_RECOMP_RATIOS", "0.15").split(",")]
SEL_REDUCE = os.environ.get("CB_SEL_REDUCE", "mean")      # importance [L,H,T]→[T]
CHUNK_NORM = os.environ.get("CB_CHUNK_NORM", "rank")      # cross-chunk comparability
PRUNE_REDUCE = "mean"                                      # FIXED for all arms
CHECK_LAYER = int(os.environ.get("CACHEBLEND_CHECK_LAYER", "1"))
GATE_PCT = float(os.environ.get("CB_GATE_PCT", "0.7"))     # gated: keep top 70% by imp
SCHED_START = float(os.environ.get("CB_SCHED_START", "1.5"))  # grad: stage-0 = 1.5×k → 0.5×k
MAX_NEW_TOKENS = 32
ARMS = ("hkvd", "imp", "random", "position", "gated", "grad_imp", "grad_hybrid")

PREFIX_PROMPT = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
QUERY_PROMPT = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
_WRAPPERS = {
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


def _rank01(x: torch.Tensor) -> torch.Tensor:
    """Within-chunk percentile rank in [0,1] (preserves order incl. sink peaks)."""
    n = x.numel()
    return x.argsort().argsort().float() / max(1, n - 1)


def main() -> int:
    print(f"[selector-compare] model={MODEL} N={N} kvzip={KVZIP_RATIOS} recomp={RECOMP_RATIOS} "
          f"sel_reduce={SEL_REDUCE} chunk_norm={CHUNK_NORM} prune_reduce={PRUNE_REDUCE} gate={GATE_PCT} sched_start={SCHED_START}", flush=True)
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"      # isolated per-chunk scoring (sink included)
    lw = LayerwiseModel(MODEL, dtype="float16", device="cuda", attn_implementation="sdpa")
    model, tokenizer, device = lw.model, lw.tokenizer, lw.device
    user_open, assistant_open = _resolve_wrapper(MODEL, tokenizer)
    ds = load_dataset("inputs/musique_s.json")[:N]

    def dec(out):
        return _decode(model, tokenizer, out.logits, out.past_key_values, device)

    f1: dict[str, list] = {"full_prefill_all": []}
    diag: dict[str, list] = {}
    for r in KVZIP_RATIOS:
        f1[f"reuse@kv{r}"] = []
        for rr in RECOMP_RATIOS:
            for a in ARMS:
                f1[f"{a}@kv{r}_rc{rr}"] = []
            diag[f"jaccard@kv{r}_rc{rr}"] = []
            for a in ("hkvd", "imp", "position"):
                diag[f"startfrac_{a}@kv{r}_rc{rr}"] = []

    for qi, ex in enumerate(ds):
        answers = ex["answers"]
        doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)
        ctexts = [user_open + PREFIX_PROMPT] + list(doc_prompts) + [q_prompt + assistant_open]
        orig = _build_chunks(tokenizer, ctexts)
        doc_slice = slice(1, 1 + len(doc_prompts))

        out = fuse_full_recompute(lw, orig, return_layerwise_output=True)
        f1["full_prefill_all"].append(max(compute_f1(dec(out), a, tokenizer) for a in answers))
        del out
        torch.cuda.empty_cache()

        cmp_full = {}
        for c in orig:
            ids = torch.tensor([c.token_ids], dtype=torch.long, device=device)
            cmp_full[c.chunk_id] = backend.score(ids).to(device)

        for r in KVZIP_RATIOS:
            budget = CompressionBudget(ratio=r)
            survivor_cmps = []
            for ci, c in enumerate(orig):
                cc = cmp_full[c.chunk_id]
                if doc_slice.start <= ci < doc_slice.stop and r < 1.0:
                    cc = token_prune(cc, budget, reduce=PRUNE_REDUCE)
                survivor_cmps.append(cc)
            blend_chunks, store = to_blend_inputs(survivor_cmps)

            # fused-sequence geometry + per-position scores
            lens = [len(c.token_ids) for c in blend_chunks]
            total = sum(lens)
            starts = np.cumsum([0] + lens[:-1]).tolist()
            doc_mask = torch.zeros(total, dtype=torch.bool)
            imp_scores = torch.zeros(total)
            pos_scores = torch.zeros(total)
            offset_in_chunk = torch.zeros(total, dtype=torch.long)
            for ci in range(len(blend_chunks)):
                s, T = starts[ci], lens[ci]
                offset_in_chunk[s:s + T] = torch.arange(T)
                if doc_slice.start <= ci < doc_slice.stop:
                    doc_mask[s:s + T] = True
                    tok = reduce_importance(survivor_cmps[ci].importance, SEL_REDUCE).cpu()
                    imp_scores[s:s + T] = _rank01(tok) if CHUNK_NORM == "rank" else tok
                    pos_scores[s:s + T] = -torch.arange(T, dtype=torch.float32)
            g = torch.Generator().manual_seed(10_000 + qi)
            rand_scores = torch.rand(total, generator=g)

            # Per-layer fused importance (per-layer within-chunk rank) + FUTURE
            # MASS F[l,t] = mean importance over layers >= l — "how much will this
            # token's fresh KV matter from layer l onward" (offline clairvoyance).
            n_layers = lw.num_layers
            imp_layers = torch.zeros(n_layers, total)
            for ci in range(len(blend_chunks)):
                if doc_slice.start <= ci < doc_slice.stop:
                    s, T = starts[ci], lens[ci]
                    M = survivor_cmps[ci].importance.float().mean(dim=1).cpu()   # [L,T]
                    Mr = M.argsort(dim=-1).argsort(dim=-1).float() / max(1, T - 1)
                    imp_layers[:, s:s + T] = Mr
            Fmass = torch.flip(torch.cumsum(torch.flip(imp_layers, [0]), 0), [0])
            Fmass = Fmass / torch.arange(n_layers, 0, -1, dtype=torch.float32).view(-1, 1)

            # reuse floor (rc=0 + forced query)
            out = fuse_selective(lw, blend_chunks, store, recompute_ratio=0.0,
                                 check_layer=CHECK_LAYER, return_layerwise_output=True,
                                 force_last_chunk=True)
            f1[f"reuse@kv{r}"].append(max(compute_f1(dec(out), a, tokenizer) for a in answers))
            del out
            torch.cuda.empty_cache()

            for rr in RECOMP_RATIOS:
                # equal-budget bookkeeping
                n_doc = int(doc_mask.sum())
                k_target = int(n_doc * rr)                 # flat non-forced budget
                n_forced = lens[-1]                        # query chunk (forced fresh)
                n_stages = n_layers - CHECK_LAYER
                # grad: linear decay SCHED_START×k → (2−SCHED_START)×k, mean = k
                # → Σ budgets == flat (n_forced + k) × n_stages (equal FLOPs)
                decay = np.linspace(SCHED_START, 2.0 - SCHED_START, n_stages)
                budgets = [int(n_forced + round(k_target * d)) for d in decay]
                # gated: top GATE_PCT of doc positions by importance; HKVD inside;
                # ratio adjusted so selected count == k_target (equal budget)
                doc_idx = torch.nonzero(doc_mask).flatten()
                k_gate = max(1, int(n_doc * GATE_PCT))
                gate_keep = doc_idx[torch.topk(imp_scores[doc_idx], k_gate).indices]
                gated_mask = torch.zeros(total, dtype=torch.bool)
                gated_mask[gate_keep] = True
                rr_gated = min(1.0, k_target / max(1, k_gate))

                arm_cfg = {
                    "hkvd":        dict(),
                    "imp":         dict(selection_scores=imp_scores),
                    "random":      dict(selection_scores=rand_scores),
                    "position":    dict(selection_scores=pos_scores),
                    "gated":       dict(eligible_mask=gated_mask, recompute_ratio=rr_gated),
                    "grad_imp":    dict(selection_scores=Fmass[CHECK_LAYER],
                                        layer_scores=Fmass, layer_budgets=budgets),
                    "grad_hybrid": dict(layer_scores=Fmass, layer_budgets=budgets),
                }
                sels = {}
                for arm in ARMS:
                    kw = dict(arm_cfg[arm])
                    kw.setdefault("eligible_mask", doc_mask)
                    kw.setdefault("recompute_ratio", rr)
                    out, top = fuse_selective(
                        lw, blend_chunks, store,
                        check_layer=CHECK_LAYER, return_layerwise_output=True,
                        return_hkvd_indices=True, force_last_chunk=True, **kw)
                    f1[f"{arm}@kv{r}_rc{rr}"].append(
                        max(compute_f1(dec(out), a, tokenizer) for a in answers))
                    sel_doc = {int(p) for p in top.tolist() if bool(doc_mask[int(p)])}
                    sels[arm] = sel_doc
                    del out
                    torch.cuda.empty_cache()
                ji = (len(sels["hkvd"] & sels["imp"]) / max(1, len(sels["hkvd"] | sels["imp"])))
                diag[f"jaccard@kv{r}_rc{rr}"].append(ji)
                for a in ("hkvd", "imp", "position"):
                    sel = sels[a]
                    frac = (sum(1 for p in sel if int(offset_in_chunk[p]) < 4) / max(1, len(sel)))
                    diag[f"startfrac_{a}@kv{r}_rc{rr}"].append(frac)
        if (qi + 1) % 10 == 0 or qi == 0:
            print(f"  [{qi+1}/{len(ds)}] full_all={np.mean(f1['full_prefill_all']):.3f}", flush=True)

    means = {k: float(np.mean(v)) for k, v in f1.items() if v}
    dmeans = {k: float(np.mean(v)) for k, v in diag.items() if v}
    rng = np.random.default_rng(0)

    def bootci(a_key, b_key, iters=2000):
        d = np.array(f1[a_key]) - np.array(f1[b_key])
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(iters)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        return float(d.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)

    print(f"\n{'='*72}\n== MEANS (N={len(ds)}) ==", flush=True)
    print(f"  full_prefill_all : {means['full_prefill_all']:.4f}", flush=True)
    for r in KVZIP_RATIOS:
        print(f"  kv={r}: reuse_floor={means[f'reuse@kv{r}']:.4f}", flush=True)
        for rr in RECOMP_RATIOS:
            vals = "  ".join(f"{a}={means[f'{a}@kv{r}_rc{rr}']:.4f}" for a in ARMS)
            print(f"    rc={rr}: {vals}", flush=True)
            for a_key, b_key, label in (
                ("imp", "hkvd", "imp−hkvd"),
                ("gated", "hkvd", "gated−hkvd"),
                ("grad_imp", "hkvd", "grad_imp−hkvd"),
                ("grad_hybrid", "hkvd", "grad_hybrid−hkvd"),
                ("random", "hkvd", "rnd−hkvd"),
                ("position", "imp", "pos−imp"),
            ):
                d_, lo, hi, sig = bootci(f"{a_key}@kv{r}_rc{rr}", f"{b_key}@kv{r}_rc{rr}")
                print(f"      {label:18s}: {d_:+.4f} CI[{lo:+.3f},{hi:+.3f}]{'★' if sig else ''}", flush=True)
            print(f"      diag: jaccard(hkvd,imp)={dmeans[f'jaccard@kv{r}_rc{rr}']:.3f}  "
                  f"start<4: hkvd={dmeans[f'startfrac_hkvd@kv{r}_rc{rr}']:.3f} "
                  f"imp={dmeans[f'startfrac_imp@kv{r}_rc{rr}']:.3f} "
                  f"pos={dmeans[f'startfrac_position@kv{r}_rc{rr}']:.3f}", flush=True)

    out_path = os.environ.get("CB_OUT", f"/tmp/selcmp_{MODEL.split('/')[-1]}.json")
    with open(out_path, "w") as fh:
        json.dump({"model": MODEL, "N": len(ds), "kvzip_ratios": KVZIP_RATIOS,
                   "recomp_ratios": RECOMP_RATIOS, "sel_reduce": SEL_REDUCE,
                   "chunk_norm": CHUNK_NORM, "means": means, "diag_means": dmeans,
                   "f1": f1, "diag": diag}, fh)
    print(f"WROTE {out_path}\nSELECTOR_COMPARE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
