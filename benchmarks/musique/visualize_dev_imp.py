"""Dump per-token KV-deviation and KVzip-importance profiles for visualization.

For randomly chosen MuSiQue questions (seed=42; q1 ⊂ q5 ⊂ q100 nested), at each
KV ratio in CB_KVZIP_VIS (default "1.0,0.5" — uncompressed AND pruned survivors):
  deviation[t] = mean over ALL layers of ||K_fresh_pre[l,t] − K_cached_pre[l,t]||²
                 (feature-sum already integrates heads; pre-RoPE K is
                 position-free so cached/fresh are directly comparable)
  importance[t] = mean over (layers, kv-heads) of KVzip importance

Outputs JSON (CB_OUT): {"ratios": {"<r>": {"q1": full-resolution arrays +
chunk bounds (absolute-position figure), "q5"/"q100": per-question 50-bin
within-chunk-rank curves (relative position 0→1, averaged over doc chunks)}}}.
Chunks are scored ONCE per question; pruning per ratio reuses the scored chunk.
GPU + KVzip needed. Plotting is local (plot_dev_imp.py).
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
from cacheblend.compress import CompressionBudget, token_prune
from cacheblend.compress.kvzip import KVzipBackend, KVzipConfig

import importlib.util as _ilu
_us = _ilu.spec_from_file_location("_cbq_utils", str(HERE / "utils.py"))
_um = _ilu.module_from_spec(_us); _us.loader.exec_module(_um)
load_dataset, build_qa_prompt = _um.load_dataset, _um.build_qa_prompt

MODEL = os.environ["CACHEBLEND_MODEL"]
N_BIG = int(os.environ.get("CB_N_VIS", "100"))
RATIOS = [float(x) for x in os.environ.get("CB_KVZIP_VIS", "1.0,0.5").split(",")]
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
    n = len(vals)
    idx = np.minimum((np.arange(n) / max(1, n) * bins).astype(int), bins - 1)
    out = np.zeros(bins); cnt = np.zeros(bins)
    np.add.at(out, idx, vals); np.add.at(cnt, idx, 1)
    return out / np.maximum(cnt, 1)


def main() -> int:
    print(f"[devimp-vis] model={MODEL} N={N_BIG} ratios={RATIOS} bins={BINS}", flush=True)
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"
    lw = LayerwiseModel(MODEL, dtype="float16", device="cuda", attn_implementation="sdpa")
    tokenizer, device = lw.tokenizer, lw.device
    n_layers = lw.num_layers
    mid = MODEL.lower()
    user_open, assistant_open = next((w for k, w in _WRAPPERS.items() if k in mid), ("", ""))
    ds = load_dataset("inputs/musique_s.json")
    order = np.random.default_rng(42).permutation(len(ds))[:N_BIG].tolist()
    print(f"  selected (seed=42): q1={order[0]}  q5={order[:5]}", flush=True)

    def build_chunks(ex):
        doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)
        ctexts = [user_open + PREFIX_PROMPT] + list(doc_prompts) + [q_prompt + assistant_open]
        bos = tokenizer.bos_token_id
        chunks = []
        for i, text in enumerate(ctexts):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if i == 0 and bos is not None:
                ids = [bos] + ids
            chunks.append(Chunk(text=text, token_ids=ids, chunk_id=_stable_id(text, ids)))
        return chunks, (1, 1 + len(doc_prompts))

    def profile(surv_cmps):
        """Survivor chunks → (dev_raw[S], imp_raw[S], bounds)."""
        lens = [cc.chunk_len for cc in surv_cmps]
        bounds = np.cumsum([0] + lens).tolist()
        S = bounds[-1]
        K_cached = [np.zeros((S, surv_cmps[0].key_cache[0].shape[-1]), dtype=np.float32)
                    for _ in range(n_layers)]
        imp_LT = np.zeros((n_layers, S), dtype=np.float32)
        for ci, cc in enumerate(surv_cmps):
            s = bounds[ci]
            for li in range(n_layers):
                K_cached[li][s:s + cc.chunk_len] = cc.key_cache[li][0].float().cpu().numpy()
            imp_LT[:, s:s + cc.chunk_len] = cc.importance.float().mean(dim=1).cpu().numpy()
        surv_chunks = [Chunk(text="", token_ids=list(cc.token_ids), chunk_id=f"v{ci}")
                       for ci, cc in enumerate(surv_cmps)]
        with torch.inference_mode():
            lw.forward_layerwise(fused_input_ids(surv_chunks, device=device), use_cache=True)
        dev_L = np.zeros((n_layers, S), dtype=np.float32)
        for li in range(n_layers):
            fresh = lw.get_pre_rope_k(li)[0].float().cpu().numpy()
            dev_L[li] = ((fresh - K_cached[li]) ** 2).sum(-1)
        torch.cuda.empty_cache()
        return dev_L.mean(0), imp_LT.mean(0), bounds

    def binned_curves(dev_raw, imp_raw, bounds, dsl):
        dcs, ics = [], []
        for ci in range(dsl[0], dsl[1]):
            s, e = bounds[ci], bounds[ci + 1]
            if e - s < 8:
                continue
            dcs.append(_binned(_rank01(dev_raw[s:e])))
            ics.append(_binned(_rank01(imp_raw[s:e])))
        return np.mean(dcs, axis=0), np.mean(ics, axis=0)

    out = {"model": MODEL, "seed": 42, "order": order, "bins": BINS,
           "ratios": {str(r): {"q5": {}, "q100": {"dev": [], "imp": []}} for r in RATIOS}}
    for j, qi in enumerate(order):
        chunks, dsl = build_chunks(ds[qi])
        cmps = [backend.score(torch.tensor([c.token_ids], dtype=torch.long, device=device)).to(device)
                for c in chunks]                       # score ONCE per question
        for r in RATIOS:
            surv = [token_prune(cc, CompressionBudget(ratio=r), reduce="mean")
                    if (dsl[0] <= ci < dsl[1] and r < 1.0) else cc
                    for ci, cc in enumerate(cmps)]
            dev_raw, imp_raw, bounds = profile(surv)
            db, ib = binned_curves(dev_raw, imp_raw, bounds, dsl)
            slot = out["ratios"][str(r)]
            slot["q100"]["dev"].append(db.tolist())
            slot["q100"]["imp"].append(ib.tolist())
            if j == 0:
                dev_rank = np.zeros_like(dev_raw); imp_rank = np.zeros_like(imp_raw)
                for ci in range(len(bounds) - 1):
                    s, e = bounds[ci], bounds[ci + 1]
                    dev_rank[s:e] = _rank01(dev_raw[s:e]); imp_rank[s:e] = _rank01(imp_raw[s:e])
                slot["q1"] = {"qidx": int(qi), "dev_raw": dev_raw.tolist(),
                              "imp_raw": imp_raw.tolist(), "dev_rank": dev_rank.tolist(),
                              "imp_rank": imp_rank.tolist(), "bounds": bounds,
                              "doc_slice": list(dsl)}
            if j < 5:
                slot["q5"][str(int(qi))] = {"dev": db.tolist(), "imp": ib.tolist()}
        del cmps
        torch.cuda.empty_cache()
        if (j + 1) % 10 == 0 or j == 0:
            print(f"  [{j+1}/{len(order)}]", flush=True)

    out_path = os.environ.get("CB_OUT", f"/tmp/devimp_{MODEL.split('/')[-1]}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    print(f"WROTE {out_path}\nDEVIMP_VIS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
