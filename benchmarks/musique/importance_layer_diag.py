"""Step-0 go/no-go diagnostic for importance-guided layer scheduling.

The scheduling idea (gradual-filtering with offline importance, "HKVD decides who,
importance decides how deep") only pays off if per-layer importance rankings
actually DIFFER across layers. If every layer ranks tokens the same, a flat
top-k already captures everything and per-layer scheduling cannot help.

For each doc chunk this script reduces importance [L, H_kv, T] over heads (mean)
to [L, T] and measures:
  adj_spearman    mean Spearman rank corr between ADJACENT layers (CacheBlend's
                  Insight-2 quantity, but for importance instead of deviation)
  layer_vs_mean   per-layer Spearman vs the all-layer-mean ranking (min / mean)
  deep_only_frac  |top15%(deep half) − top15%(shallow quarter)| / |top15%(deep)|
                  — fraction of deep-important tokens INVISIBLE to shallow layers
  shallow1_jacc   Jaccard(top15% at layer 1, top15% of all-layer mean) — how much
                  a single shallow layer misses of the full-depth signal

GO   (scheduling worth implementing): adj_spearman noticeably < 1 AND
     deep_only_frac substantially > 0 (deep-only tokens exist).
NO-GO: adj_spearman ≈ 1 and deep_only_frac ≈ 0 → flat top-k suffices; stop.

Needs GPU + KVzip (scoring only; no blending). Env: CACHEBLEND_MODEL,
CB_N (default 20), CB_OUT (default /tmp/imp_layer_diag_<model>.json).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
# Avoid shadowing KVzip's top-level `utils` package (see blend_compress_kvzip.py).
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != HERE]
os.chdir(HERE)
os.environ.setdefault("CACHEBLEND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

from cacheblend.chunker import Chunk, _stable_id
from cacheblend.compress.kvzip import KVzipBackend, KVzipConfig

import importlib.util as _ilu
_us = _ilu.spec_from_file_location("_cbq_utils", str(HERE / "utils.py"))
_um = _ilu.module_from_spec(_us); _us.loader.exec_module(_um)
load_dataset, build_qa_prompt = _um.load_dataset, _um.build_qa_prompt

MODEL = os.environ["CACHEBLEND_MODEL"]
N = int(os.environ.get("CB_N", "20"))
TOPF = 0.15                                    # top-15% sets (CacheBlend's regime)

PREFIX_PROMPT = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
QUERY_PROMPT = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
_WRAPPERS = {"llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                          "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
             "llama3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
             "qwen": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n")}


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def _topset(x: torch.Tensor, frac: float) -> set:
    k = max(1, int(round(x.numel() * frac)))
    return set(torch.topk(x, k).indices.tolist())


def main() -> int:
    print(f"[imp-layer-diag] model={MODEL} N={N} top_frac={TOPF}", flush=True)
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"
    tokenizer = backend.tokenizer
    mid = MODEL.lower()
    user_open = next((w[0] for k, w in _WRAPPERS.items() if k in mid), "")
    ds = load_dataset("inputs/musique_s.json")[:N]

    adj_sp, lvm_min, lvm_mean, deep_only, sh1_jacc, lens = [], [], [], [], [], []
    for qi, ex in enumerate(ds):
        doc_prompts, _q = build_qa_prompt(ex, QUERY_PROMPT)
        for d in doc_prompts:
            ids = tokenizer(d, add_special_tokens=False)["input_ids"]
            if len(ids) < 16:
                continue
            cc = backend.score(torch.tensor([ids], dtype=torch.long))
            M = cc.importance.float().mean(dim=1).cpu()        # [L, T] head-mean
            L, T = M.shape
            lens.append(T)
            # adjacent-layer Spearman
            adj_sp.append(float(np.mean([_spearman(M[l], M[l + 1]) for l in range(L - 1)])))
            # per-layer vs all-layer-mean ranking
            mean_rank = M.mean(dim=0)
            per = [_spearman(M[l], mean_rank) for l in range(L)]
            lvm_min.append(min(per)); lvm_mean.append(float(np.mean(per)))
            # deep-only fraction: deep-half top15% not visible in shallow-quarter top15%
            shallow = set().union(*[_topset(M[l], TOPF) for l in range(max(1, L // 4))])
            deep = set().union(*[_topset(M[l], TOPF) for l in range(L // 2, L)])
            deep_only.append(len(deep - shallow) / max(1, len(deep)))
            # single shallow layer (1) vs full-depth mean
            sh1_jacc.append(len(_topset(M[1], TOPF) & _topset(mean_rank, TOPF))
                            / max(1, len(_topset(M[1], TOPF) | _topset(mean_rank, TOPF))))
        if (qi + 1) % 5 == 0:
            print(f"  [{qi+1}/{N}] chunks={len(adj_sp)}", flush=True)

    def s(x):
        return {"mean": float(np.mean(x)), "std": float(np.std(x)), "n": len(x)}

    out = {"model": MODEL, "N": N, "top_frac": TOPF, "chunk_len": s(lens),
           "adj_spearman": s(adj_sp), "layer_vs_mean_min": s(lvm_min),
           "layer_vs_mean_mean": s(lvm_mean), "deep_only_frac": s(deep_only),
           "shallow1_vs_fulldepth_jaccard": s(sh1_jacc)}
    print(f"\n{'='*64}\n== importance layer-structure (chunks={len(adj_sp)}) ==", flush=True)
    for k in ("adj_spearman", "layer_vs_mean_min", "layer_vs_mean_mean",
              "deep_only_frac", "shallow1_vs_fulldepth_jaccard"):
        print(f"  {k:32s}: {out[k]['mean']:.4f} ± {out[k]['std']:.4f}", flush=True)
    go = out["adj_spearman"]["mean"] < 0.95 and out["deep_only_frac"]["mean"] > 0.05
    verdict = ("GO — per-layer structure exists; scheduling may pay off" if go
               else "NO-GO — rankings ~identical across layers; flat top-k suffices")
    print(f"\n  VERDICT: {verdict}", flush=True)
    out_path = os.environ.get("CB_OUT", f"/tmp/imp_layer_diag_{MODEL.split('/')[-1]}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    print(f"WROTE {out_path}\nIMP_LAYER_DIAG_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
