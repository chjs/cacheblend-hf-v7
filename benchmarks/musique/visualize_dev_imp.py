"""Dump per-token KV-deviation and KVzip-importance profiles for visualization.

For randomly chosen MuSiQue questions (seed=42; q1 ⊂ q5 ⊂ q100 nested):
  deviation[t] = mean over ALL layers of ||K_fresh_pre[l,t] − K_cached_pre[l,t]||²
                 (feature-sum already integrates heads; pre-RoPE K is
                 position-free so cached/fresh are directly comparable)
  importance[t] = mean over (layers, kv-heads) of KVzip importance

Outputs JSON (CB_OUT, default /tmp/devimp_<model>.json):
  q1   : full-resolution raw + within-chunk-rank arrays over the fused sequence,
         chunk boundaries, doc region — for the absolute-position figure
  q5   : per-question binned within-chunk curves (50 bins over relative pos 0→1,
         within-chunk rank-normalized, averaged over that question's doc chunks)
  q100 : same binned curves for 100 questions
kv ratio = 1.0 (no pruning — full signal). GPU + KVzip needed. Plotting is local.
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
from cacheblend.compress.kvzip import KVzipBackend, KVzipConfig

import importlib.util as _ilu
_us = _ilu.spec_from_file_location("_cbq_utils", str(HERE / "utils.py"))
_um = _ilu.module_from_spec(_us); _us.loader.exec_module(_um)
load_dataset, build_qa_prompt = _um.load_dataset, _um.build_qa_prompt

MODEL = os.environ["CACHEBLEND_MODEL"]
N_BIG = int(os.environ.get("CB_N_VIS", "100"))
BINS = 50

PREFIX_PROMPT = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
QUERY_PROMPT = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
_WRAPPERS = {"llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                          "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
             "llama3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
             "qwen": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n")}


def _rank01(x: np.ndarray) -> np.ndarray:
    n = len(x)
    r = np.empty(n); r[np.argsort(x)] = np.arange(n)
    return r / max(1, n - 1)


def _binned(vals: np.ndarray, bins: int = BINS) -> np.ndarray:
    """Average `vals` (one chunk) into `bins` equal relative-position bins."""
    n = len(vals)
    idx = np.minimum((np.arange(n) / max(1, n) * bins).astype(int), bins - 1)
    out = np.zeros(bins); cnt = np.zeros(bins)
    np.add.at(out, idx, vals); np.add.at(cnt, idx, 1)
    return out / np.maximum(cnt, 1)


def main() -> int:
    print(f"[devimp-vis] model={MODEL} N={N_BIG} bins={BINS}", flush=True)
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"
    lw = LayerwiseModel(MODEL, dtype="float16", device="cuda", attn_implementation="sdpa")
    tokenizer, device = lw.tokenizer, lw.device
    mid = MODEL.lower()
    user_open, assistant_open = next((w for k, w in _WRAPPERS.items() if k in mid),
                                     ("", ""))
    ds = load_dataset("inputs/musique_s.json")
    order = np.random.default_rng(42).permutation(len(ds))[:N_BIG].tolist()
    print(f"  selected (seed=42): q1={order[0]}  q5={order[:5]}", flush=True)

    def profile(ex):
        """→ (dev_raw[S], imp_raw[S], bounds, doc_slice) for one question."""
        doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)
        ctexts = [user_open + PREFIX_PROMPT] + list(doc_prompts) + [q_prompt + assistant_open]
        bos = tokenizer.bos_token_id
        chunks = []
        for i, text in enumerate(ctexts):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if i == 0 and bos is not None:
                ids = [bos] + ids
            chunks.append(Chunk(text=text, token_ids=ids, chunk_id=_stable_id(text, ids)))
        lens = [len(c.token_ids) for c in chunks]
        bounds = np.cumsum([0] + lens).tolist()
        S = bounds[-1]
        n_layers = lw.num_layers

        # cached pre-RoPE K + importance per chunk (KVzip scoring, no prune)
        K_cached = None; imp_LT = None
        col = 0
        for c in chunks:
            cc = backend.score(torch.tensor([c.token_ids], dtype=torch.long, device=device))
            T = cc.chunk_len
            if K_cached is None:
                K_cached = [np.zeros((S, cc.key_cache[0].shape[-1]), dtype=np.float32)
                            for _ in range(n_layers)]
                imp_LT = np.zeros((n_layers, S), dtype=np.float32)
            for li in range(n_layers):
                K_cached[li][col:col + T] = cc.key_cache[li][0].float().cpu().numpy()
            imp_LT[:, col:col + T] = cc.importance.float().mean(dim=1).cpu().numpy()
            col += T
            del cc
            torch.cuda.empty_cache()

        # fresh pre-RoPE K at every layer: one full forward (k_proj hooks capture)
        with torch.inference_mode():
            lw.forward_layerwise(fused_input_ids(chunks, device=device), use_cache=True)
        dev_L = np.zeros((n_layers, S), dtype=np.float32)
        for li in range(n_layers):
            fresh = lw.get_pre_rope_k(li)[0].float().cpu().numpy()
            dev_L[li] = ((fresh - K_cached[li]) ** 2).sum(-1)
        torch.cuda.empty_cache()

        dev_raw = dev_L.mean(0)            # layer-mean (heads inside feature sum)
        imp_raw = imp_LT.mean(0)           # (L,H)-mean
        return dev_raw, imp_raw, bounds, (1, 1 + len(doc_prompts))

    def binned_curves(dev_raw, imp_raw, bounds, doc_slice):
        """Within-chunk rank → 50-bin relative-position curves, avg over doc chunks."""
        dcs, ics = [], []
        for ci in range(doc_slice[0], doc_slice[1]):
            s, e = bounds[ci], bounds[ci + 1]
            if e - s < 8:
                continue
            dcs.append(_binned(_rank01(dev_raw[s:e])))
            ics.append(_binned(_rank01(imp_raw[s:e])))
        return np.mean(dcs, axis=0), np.mean(ics, axis=0)

    out = {"model": MODEL, "seed": 42, "order": order, "bins": BINS,
           "q5": {}, "q100": {"dev": [], "imp": []}}
    for j, qi in enumerate(order):
        dev_raw, imp_raw, bounds, dsl = profile(ds[qi])
        db, ib = binned_curves(dev_raw, imp_raw, bounds, dsl)
        out["q100"]["dev"].append(db.tolist())
        out["q100"]["imp"].append(ib.tolist())
        if j == 0:
            # full-resolution for the single-question absolute-position figure
            dev_rank = np.zeros_like(dev_raw); imp_rank = np.zeros_like(imp_raw)
            for ci in range(len(bounds) - 1):
                s, e = bounds[ci], bounds[ci + 1]
                dev_rank[s:e] = _rank01(dev_raw[s:e]); imp_rank[s:e] = _rank01(imp_raw[s:e])
            out["q1"] = {"qidx": int(qi), "dev_raw": dev_raw.tolist(),
                         "imp_raw": imp_raw.tolist(), "dev_rank": dev_rank.tolist(),
                         "imp_rank": imp_rank.tolist(), "bounds": bounds,
                         "doc_slice": list(dsl)}
        if j < 5:
            out["q5"][str(int(qi))] = {"dev": db.tolist(), "imp": ib.tolist()}
        if (j + 1) % 10 == 0 or j == 0:
            print(f"  [{j+1}/{len(order)}]", flush=True)

    out_path = os.environ.get("CB_OUT", f"/tmp/devimp_{MODEL.split('/')[-1]}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    print(f"WROTE {out_path}\nDEVIMP_VIS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
