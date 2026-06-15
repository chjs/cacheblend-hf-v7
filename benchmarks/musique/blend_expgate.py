"""Deletion-exposure (B_iso) experiment: can we exclude UNRECOVERABLE tokens?

Idea: under compression HKVD picks tokens whose deviation is inflated by
NEIGHBOR DELETION (component B) — recompute can't fix those. B is measurable
offline, in isolation (no cross-chunk):
    B_iso(i) = || K(i | survivors-only, isolated)  −  old_k(i | full chunk) ||²
Both isolated → no cross-chunk (A) → pure deletion damage. old_k = stored KVzip K.

Two questions:
  (Q1 diagnostic) Does HKVD over-select high-B_iso tokens as compression rises?
      → report B_iso percentile of HKVD-selected doc tokens (>0.5 = prefers unrecoverable).
  (Q2 intervention) Does gating OUT high-B_iso (keep low-B_iso) then HKVD beat
      plain HKVD? Arms expgate50/expgate70 = eligible is the low-B_iso bottom
      50%/70% of each chunk; HKVD within, budget equalized to plain HKVD.

Arms: hkvd, expgate50, expgate70, position (+ reuse floor, full ceiling).
force_last_chunk + eligible=docs throughout; prune fixed mean. Paired bootstrap
expgate−hkvd and expgate−position. Env: CACHEBLEND_MODEL, CB_N (100),
CB_KVZIP_RATIOS ("0.5,0.3,0.2"), CB_RECOMP_RATIOS ("0.15").
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != HERE]
os.chdir(HERE)
os.environ.setdefault("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

from cacheblend import LayerwiseModel
from cacheblend.chunker import Chunk, _stable_id, fused_input_ids
from cacheblend.fusor import fuse_selective, fuse_full_recompute
from cacheblend.compress import CompressionBudget, token_prune, to_blend_inputs
from cacheblend.compress.kvzip import KVzipBackend, KVzipConfig

import importlib.util as _ilu
_us = _ilu.spec_from_file_location("_cbq_utils", str(HERE / "utils.py"))
_um = _ilu.module_from_spec(_us); _us.loader.exec_module(_um)
load_dataset, build_qa_prompt, compute_f1 = _um.load_dataset, _um.build_qa_prompt, _um.compute_f1

MODEL = os.environ["CACHEBLEND_MODEL"]
N = int(os.environ.get("CB_N", "100"))
KVZIP_RATIOS = [float(x) for x in os.environ.get("CB_KVZIP_RATIOS", "0.5,0.3,0.2").split(",")]
RECOMP_RATIOS = [float(x) for x in os.environ.get("CB_RECOMP_RATIOS", "0.15").split(",")]
GATES = [0.5, 0.7]                      # keep low-B_iso bottom fraction
CHECK_LAYER = int(os.environ.get("CACHEBLEND_CHECK_LAYER", "1"))
MAX_NEW_TOKENS = 32
ARMS = ("hkvd", "expgate50", "expgate70", "position")

PREFIX_PROMPT = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
QUERY_PROMPT = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
_WRAPPERS = {
    "llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama3":  ("<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "qwen":    ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"),
}


def _resolve_wrapper(model_id, tok):
    mid = model_id.lower()
    for k, w in _WRAPPERS.items():
        if k in mid:
            return w
    s = "\x00C\x00"
    t = tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True)
    pre, post = t.split(s, 1)
    bos = tok.bos_token or ""
    return (pre[len(bos):] if bos and pre.startswith(bos) else pre), post


def _build_chunks(tok, texts):
    bos = tok.bos_token_id
    out = []
    for i, t in enumerate(texts):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if i == 0 and bos is not None:
            ids = [bos] + ids
        out.append(Chunk(text=t, token_ids=ids, chunk_id=_stable_id(t, ids)))
    return out


def _decode(model, tok, logits, pkv, device):
    eos = getattr(tok, "eos_token_id", None)
    nxt = logits[0, -1].argmax().view(1, 1); gen = [int(nxt)]
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS - 1):
            if eos is not None and gen[-1] == eos:
                break
            o = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values; nxt = o.logits[0, -1].argmax().view(1, 1); gen.append(int(nxt))
    return tok.decode(gen, skip_special_tokens=True)


def _rank01(x):
    n = len(x); r = np.empty(n); r[np.argsort(x)] = np.arange(n)
    return r / max(1, n - 1)


def main() -> int:
    print(f"[expgate] model={MODEL} N={N} kvzip={KVZIP_RATIOS} rc={RECOMP_RATIOS} gates={GATES} check_layer={CHECK_LAYER}", flush=True)
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"
    lw = LayerwiseModel(MODEL, dtype="float16", device="cuda", attn_implementation="sdpa")
    model, tok, device = lw.model, lw.tokenizer, lw.device
    user_open, asst_open = _resolve_wrapper(MODEL, tok)
    ds = load_dataset("inputs/musique_s.json")[:N]

    def dec(o):
        return _decode(model, tok, o.logits, o.past_key_values, device)

    f1 = {"full": []}
    diag = {}                            # B_iso percentile of HKVD-selected, per (kv,rc)
    for r in KVZIP_RATIOS:
        f1[f"reuse@{r}"] = []
        for rr in RECOMP_RATIOS:
            for a in ARMS:
                f1[f"{a}@{r}_{rr}"] = []
            diag[f"hkvd_Biso_pctile@{r}_{rr}"] = []
            diag[f"corr_dev_Biso@{r}_{rr}"] = []

    for qi, ex in enumerate(ds):
        ans = ex["answers"]
        docs, qp = build_qa_prompt(ex, QUERY_PROMPT)
        ctexts = [user_open + PREFIX_PROMPT] + list(docs) + [qp + asst_open]
        orig = _build_chunks(tok, ctexts)
        dslice = slice(1, 1 + len(docs))

        out = fuse_full_recompute(lw, orig, return_layerwise_output=True)
        f1["full"].append(max(compute_f1(dec(out), a, tok) for a in ans)); del out
        torch.cuda.empty_cache()

        cmp_full = {}
        for c in orig:
            cmp_full[c.chunk_id] = backend.score(torch.tensor([c.token_ids], dtype=torch.long, device=device)).to(device)

        for r in KVZIP_RATIOS:
            bud = CompressionBudget(ratio=r)
            surv = []
            for ci, c in enumerate(orig):
                cc = cmp_full[c.chunk_id]
                if dslice.start <= ci < dslice.stop and r < 1.0:
                    cc = token_prune(cc, bud, reduce="mean")
                surv.append(cc)
            blend_chunks, store = to_blend_inputs(surv)
            lens = [len(c.token_ids) for c in blend_chunks]
            total = sum(lens); starts = np.cumsum([0] + lens[:-1]).tolist()
            doc_mask = torch.zeros(total, dtype=torch.bool)
            pos_scores = torch.zeros(total)
            Biso = torch.full((total,), float("nan"))
            Biso_rank = torch.zeros(total)          # within-chunk rank of B_iso (doc only)

            for ci in range(len(blend_chunks)):
                s, T = starts[ci], lens[ci]
                if not (dslice.start <= ci < dslice.stop):
                    continue
                doc_mask[s:s + T] = True
                pos_scores[s:s + T] = -torch.arange(T, dtype=torch.float32)
                # B_iso: forward survivors of THIS chunk ALONE → pre-RoPE K @ check_layer
                ids = torch.tensor([surv[ci].token_ids], dtype=torch.long, device=device)
                with torch.inference_mode():
                    lw.forward_layerwise(ids, use_cache=True)
                k_iso = lw.get_pre_rope_k(CHECK_LAYER)[0].float()          # [T, Hkv*D]
                k_old = surv[ci].key_cache[CHECK_LAYER][0].float()         # stored full-context K
                b = ((k_iso - k_old) ** 2).sum(-1).cpu().numpy()           # [T]
                Biso[s:s + T] = torch.tensor(b)
                Biso_rank[s:s + T] = torch.tensor(_rank01(b))              # 0=low B_iso, 1=high
                torch.cuda.empty_cache()

            # reuse floor
            out = fuse_selective(lw, blend_chunks, store, recompute_ratio=0.0, check_layer=CHECK_LAYER,
                                 return_layerwise_output=True, force_last_chunk=True)
            f1[f"reuse@{r}"].append(max(compute_f1(dec(out), a, tok) for a in ans)); del out
            torch.cuda.empty_cache()

            for rr in RECOMP_RATIOS:
                n_doc = int(doc_mask.sum()); k_t = int(n_doc * rr)
                arm_cfg = {
                    "hkvd": dict(eligible_mask=doc_mask, recompute_ratio=rr),
                    "position": dict(selection_scores=pos_scores, eligible_mask=doc_mask, recompute_ratio=rr),
                }
                for g in GATES:
                    keep = doc_mask & (Biso_rank <= g)              # low-B_iso bottom g
                    nk = int(keep.sum())
                    rr_g = min(1.0, k_t / max(1, nk))               # equalize selected count to k_t
                    arm_cfg[f"expgate{int(g*100)}"] = dict(eligible_mask=keep, recompute_ratio=rr_g)

                hkvd_top = None
                for arm in ARMS:
                    out, top = fuse_selective(lw, blend_chunks, store, check_layer=CHECK_LAYER,
                                              return_layerwise_output=True, return_hkvd_indices=True,
                                              force_last_chunk=True, **arm_cfg[arm])
                    f1[f"{arm}@{r}_{rr}"].append(max(compute_f1(dec(out), a, tok) for a in ans))
                    if arm == "hkvd":
                        hkvd_top = [int(p) for p in top.tolist() if bool(doc_mask[int(p)])]
                    del out
                    torch.cuda.empty_cache()
                # diagnostic: B_iso percentile of HKVD-selected doc tokens
                if hkvd_top:
                    diag[f"hkvd_Biso_pctile@{r}_{rr}"].append(float(np.mean([float(Biso_rank[p]) for p in hkvd_top])))
                # corr(total-blend-deviation proxy, B_iso): use HKVD ranking ~ deviation; instead corr Biso vs being-picked
                dm = doc_mask.numpy().astype(bool)
                picked = np.zeros(total); picked[hkvd_top] = 1
                if dm.sum() > 2:
                    diag[f"corr_dev_Biso@{r}_{rr}"].append(float(np.corrcoef(picked[dm], Biso_rank.numpy()[dm])[0, 1]))
        if (qi + 1) % 10 == 0 or qi == 0:
            print(f"  [{qi+1}/{len(ds)}] full={np.mean(f1['full']):.3f}", flush=True)

    means = {k: float(np.mean(v)) for k, v in f1.items() if v}
    dmeans = {k: float(np.nanmean(v)) for k, v in diag.items() if v}
    rng = np.random.default_rng(0)

    def ci(a, b):
        d = np.array(f1[a]) - np.array(f1[b])
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        lo, hi = np.quantile(bs, [.025, .975]); return d.mean(), lo, hi, (lo > 0 or hi < 0)

    print(f"\n{'='*72}\n== MEANS (N={len(ds)}) full={means['full']:.4f} ==", flush=True)
    for r in KVZIP_RATIOS:
        print(f"  kv={r}: reuse={means[f'reuse@{r}']:.4f}", flush=True)
        for rr in RECOMP_RATIOS:
            vals = "  ".join(f"{a}={means[f'{a}@{r}_{rr}']:.4f}" for a in ARMS)
            print(f"    rc={rr}: {vals}", flush=True)
            for g in GATES:
                a = f"expgate{int(g*100)}"
                d1, lo1, hi1, s1 = ci(f"{a}@{r}_{rr}", f"hkvd@{r}_{rr}")
                d2, lo2, hi2, s2 = ci(f"{a}@{r}_{rr}", f"position@{r}_{rr}")
                print(f"      {a}: vs hkvd {d1:+.4f}[{lo1:+.3f},{hi1:+.3f}]{'★' if s1 else ''}  "
                      f"vs pos {d2:+.4f}[{lo2:+.3f},{hi2:+.3f}]{'★' if s2 else ''}", flush=True)
            print(f"      DIAG: HKVD-selected B_iso pctile={dmeans.get(f'hkvd_Biso_pctile@{r}_{rr}',float('nan')):.3f} "
                  f"(>0.5=고B_iso 선호)  corr(picked,B_iso)={dmeans.get(f'corr_dev_Biso@{r}_{rr}',float('nan')):+.3f}", flush=True)

    out_path = os.environ.get("CB_OUT", f"/tmp/expgate_{MODEL.split('/')[-1]}.json")
    with open(out_path, "w") as fh:
        json.dump({"model": MODEL, "N": len(ds), "kvzip_ratios": KVZIP_RATIOS, "recomp_ratios": RECOMP_RATIOS,
                   "gates": GATES, "means": means, "diag_means": dmeans, "f1": f1, "diag": diag}, fh)
    print(f"WROTE {out_path}\nEXPGATE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
