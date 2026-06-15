"""CacheBlend-faithful new_k: compute new_k over the FULL input context.

Root-cause fix. CacheBlend defines new_kv on the REAL full concatenated input
(user prompt + all chunk text as one sequence); old_kv = per-chunk precompute.
Our compressed blend wrongly computed new_k over the SURVIVOR-only sequence
(token_prune was applied UPSTREAM of the blend), so a survivor's recomputed K
attended over fewer keys → softmax renormalization → "deletion damage" (B_iso).
That likely manufactured the HKVD-collapse-under-compression finding.

This script tests both, head-to-head, same data / same survivors:

  CORRECTED (full-input new_k): blend sequence = FULL chunks (all tokens). Cached
    old_kv injected ONLY at survivor positions; pruned doc tokens are FORCED fresh
    (forced_extra_mask) — they have no cache. HKVD pool = survivor positions
    (eligible_mask). Because the forward is over the full sequence, every survivor
    recomputes with ALL original neighbors present → no deletion damage, AND
    survivors keep their TRUE positions → correct RoPE. Pruned tokens reconstructed
    fresh (compression = memory/cross-request, blend reconstructs).

  WRONG (survivors-only new_k): blend sequence = pruned survivors (our prior impl).
    Reproduces the collapse for comparison.

Arms per kv ratio: reuse_full, hkvd_full, position_full  (corrected)
                    reuse_surv, hkvd_surv                 (wrong, repro)
+ full ceiling (uncompressed full prefill).

KEY question: does hkvd_full >= reuse_full with NO collapse at kv=0.3/0.2
(where hkvd_surv fell BELOW reuse)? If yes → the compression-failure findings
were an artifact of survivors-only new_k.

Env: CACHEBLEND_MODEL, CB_N(100), CB_KVZIP_RATIOS("0.5,0.3,0.2"),
CB_RECOMP_RATIOS("0.15"), CACHEBLEND_CHECK_LAYER(1).
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
from cacheblend.chunker import Chunk, _stable_id
from cacheblend.fusor import fuse_selective, fuse_full_recompute
from cacheblend.compress import CompressionBudget, token_prune, to_blend_inputs, reduce_importance
from cacheblend.compress.kvzip import KVzipBackend, KVzipConfig

import importlib.util as _ilu
_us = _ilu.spec_from_file_location("_cbq_utils", str(HERE / "utils.py"))
_um = _ilu.module_from_spec(_us); _us.loader.exec_module(_um)
load_dataset, build_qa_prompt, compute_f1 = _um.load_dataset, _um.build_qa_prompt, _um.compute_f1

MODEL = os.environ["CACHEBLEND_MODEL"]
N = int(os.environ.get("CB_N", "100"))
KVZIP_RATIOS = [float(x) for x in os.environ.get("CB_KVZIP_RATIOS", "0.5,0.3,0.2").split(",")]
RECOMP_RATIOS = [float(x) for x in os.environ.get("CB_RECOMP_RATIOS", "0.15").split(",")]
CHECK_LAYER = int(os.environ.get("CACHEBLEND_CHECK_LAYER", "1"))
MAX_NEW_TOKENS = 32
ARMS_FULL = ("reuse_full", "hkvd_full", "position_full")
ARMS_SURV = ("reuse_surv", "hkvd_surv")

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


def _decode(model, tok, logits, pkv):
    eos = getattr(tok, "eos_token_id", None)
    nxt = logits[0, -1].argmax().view(1, 1); gen = [int(nxt)]
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS - 1):
            if eos is not None and gen[-1] == eos:
                break
            o = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values; nxt = o.logits[0, -1].argmax().view(1, 1); gen.append(int(nxt))
    return tok.decode(gen, skip_special_tokens=True)


def main() -> int:
    print(f"[fullnewk] model={MODEL} N={N} kvzip={KVZIP_RATIOS} rc={RECOMP_RATIOS} check_layer={CHECK_LAYER}", flush=True)
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"
    lw = LayerwiseModel(MODEL, dtype="float16", device="cuda", attn_implementation="sdpa")
    model, tok, device = lw.model, lw.tokenizer, lw.device
    user_open, asst_open = _resolve_wrapper(MODEL, tok)
    ds = load_dataset("inputs/musique_s.json")[:N]

    def dec(o):
        return _decode(model, tok, o.logits, o.past_key_values)

    f1 = {"full": []}
    diag = {}
    for r in KVZIP_RATIOS:
        for a in ("reuse_full", "hkvd_full", "position_full", "reuse_surv", "hkvd_surv"):
            f1[f"{a}@{r}"] = []
        diag[f"hkvd_full_dev_pctile@{r}"] = []   # HKVD-selected survivors' deviation pctile (corrected)

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
            full_chunks, surv_chunks = [], []
            keep_idx = {}                                   # ci -> sorted kept local indices
            for ci, c in enumerate(orig):
                cc = cmp_full[c.chunk_id]
                full_chunks.append(cc)
                if dslice.start <= ci < dslice.stop and r < 1.0:
                    k = bud.keep_k(cc.chunk_len)
                    imp = reduce_importance(cc.importance, "mean")
                    keep = torch.sort(torch.topk(imp, k).indices).values
                    keep_idx[ci] = keep
                    surv_chunks.append(token_prune(cc, bud, reduce="mean"))
                else:
                    keep_idx[ci] = torch.arange(cc.chunk_len)
                    surv_chunks.append(cc)

            # ── CORRECTED: full-input new_k ───────────────────────────────────
            blend_full, store_full = to_blend_inputs(full_chunks)
            lens_f = [len(c.token_ids) for c in blend_full]
            tot_f = sum(lens_f); starts_f = np.cumsum([0] + lens_f[:-1]).tolist()
            doc_surv = torch.zeros(tot_f, dtype=torch.bool)
            doc_pruned = torch.zeros(tot_f, dtype=torch.bool)
            pos_scores_f = torch.zeros(tot_f)
            for ci in range(dslice.start, dslice.stop):
                s, T = starts_f[ci], lens_f[ci]
                survset = set(int(x) for x in keep_idx[ci].tolist())
                pos_scores_f[s:s + T] = -torch.arange(T, dtype=torch.float32)
                for t in range(T):
                    if t in survset:
                        doc_surv[s + t] = True
                    else:
                        doc_pruned[s + t] = True

            cfg_full = {
                "reuse_full":    dict(recompute_ratio=0.0, eligible_mask=doc_surv, forced_extra_mask=doc_pruned),
                "hkvd_full":     dict(recompute_ratio=RECOMP_RATIOS[0], eligible_mask=doc_surv, forced_extra_mask=doc_pruned),
                "position_full": dict(recompute_ratio=RECOMP_RATIOS[0], selection_scores=pos_scores_f,
                                      eligible_mask=doc_surv, forced_extra_mask=doc_pruned),
            }
            hkvd_full_top = None
            for arm in ARMS_FULL:
                out, top = fuse_selective(lw, blend_full, store_full, check_layer=CHECK_LAYER,
                                          return_layerwise_output=True, return_hkvd_indices=True,
                                          force_last_chunk=True, **cfg_full[arm])
                f1[f"{arm}@{r}"].append(max(compute_f1(dec(out), a, tok) for a in ans))
                if arm == "hkvd_full":
                    hkvd_full_top = [int(p) for p in top.tolist() if bool(doc_surv[int(p)])]
                del out
                torch.cuda.empty_cache()

            # ── WRONG: survivors-only new_k (reproduce collapse) ──────────────
            blend_surv, store_surv = to_blend_inputs(surv_chunks)
            lens_s = [len(c.token_ids) for c in blend_surv]
            tot_s = sum(lens_s); starts_s = np.cumsum([0] + lens_s[:-1]).tolist()
            doc_mask_s = torch.zeros(tot_s, dtype=torch.bool)
            for ci in range(dslice.start, dslice.stop):
                doc_mask_s[starts_s[ci]:starts_s[ci] + lens_s[ci]] = True
            cfg_surv = {
                "reuse_surv": dict(recompute_ratio=0.0, eligible_mask=doc_mask_s),
                "hkvd_surv":  dict(recompute_ratio=RECOMP_RATIOS[0], eligible_mask=doc_mask_s),
            }
            for arm in ARMS_SURV:
                out = fuse_selective(lw, blend_surv, store_surv, check_layer=CHECK_LAYER,
                                     return_layerwise_output=True, force_last_chunk=True, **cfg_surv[arm])
                f1[f"{arm}@{r}"].append(max(compute_f1(dec(out), a, tok) for a in ans)); del out
                torch.cuda.empty_cache()

        if (qi + 1) % 10 == 0 or qi == 0:
            print(f"  [{qi+1}/{len(ds)}] full={np.mean(f1['full']):.3f}", flush=True)

    means = {k: float(np.mean(v)) for k, v in f1.items() if v}
    rng = np.random.default_rng(0)

    def ci(a, b):
        d = np.array(f1[a]) - np.array(f1[b])
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        lo, hi = np.quantile(bs, [.025, .975]); return d.mean(), lo, hi, (lo > 0 or hi < 0)

    print(f"\n{'='*72}\n== MEANS (N={len(ds)}) full={means['full']:.4f} ==", flush=True)
    for r in KVZIP_RATIOS:
        print(f"  kv={r}:", flush=True)
        for a in ("reuse_full", "hkvd_full", "position_full", "reuse_surv", "hkvd_surv"):
            print(f"    {a:16s}={means[f'{a}@{r}']:.4f}", flush=True)
        d1, l1, h1, s1 = ci(f"hkvd_full@{r}", f"reuse_full@{r}")
        d2, l2, h2, s2 = ci(f"hkvd_surv@{r}", f"reuse_surv@{r}")
        d3, l3, h3, s3 = ci(f"hkvd_full@{r}", f"hkvd_surv@{r}")
        print(f"    HKVD recovers? full: hkvd-reuse {d1:+.4f}[{l1:+.3f},{h1:+.3f}]{'★' if s1 else ''} | "
              f"surv: hkvd-reuse {d2:+.4f}[{l2:+.3f},{h2:+.3f}]{'★' if s2 else ''}", flush=True)
        print(f"    full vs surv new_k: hkvd_full-hkvd_surv {d3:+.4f}[{l3:+.3f},{h3:+.3f}]{'★' if s3 else ''}", flush=True)

    out_path = os.environ.get("CB_OUT", f"/tmp/fullnewk_{MODEL.split('/')[-1]}.json")
    with open(out_path, "w") as fh:
        json.dump({"model": MODEL, "N": len(ds), "kvzip_ratios": KVZIP_RATIOS,
                   "recomp_ratios": RECOMP_RATIOS, "check_layer": CHECK_LAYER,
                   "means": means, "f1": f1}, fh)
    print(f"WROTE {out_path}\nFULLNEWK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
