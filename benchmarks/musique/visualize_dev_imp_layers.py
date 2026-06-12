"""Per-LAYER KV-deviation + importance profiles for ONE question (32 panels).

Scenario matches the blend pipeline's selection stage: the fused sequence is
[prefix + doc chunks] (the query is NOT part of blending — it is prefilled later
on top of the blended cache). For each layer l:
  dev[l, t] = ||K_fresh_pre[l, t] − K_cached_pre[l, t]||²   (heads inside the sum)
  imp[l, t] = head-mean of KVzip importance at layer l

Collected at each KV ratio in CB_KVZIP_VIS (default "1.0,0.5") — chunks scored
ONCE, doc pruning per ratio reuses the scored chunks. Question = seed-42 pick
(same as previous figures, q#70 on MuSiQue-150). Output JSON: per-ratio
{"dev": [L][S], "imp": [L][S], "bounds": [...]}. Plotting: plot_dev_imp_layers.py.
GPU + KVzip needed.
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
RATIOS = [float(x) for x in os.environ.get("CB_KVZIP_VIS", "1.0,0.5").split(",")]
QSEED = int(os.environ.get("CB_QSEED", "42"))

PREFIX_PROMPT = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
QUERY_PROMPT = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
_WRAPPERS = {"llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n", ""),
             "llama3": ("<|start_header_id|>user<|end_header_id|>\n\n", ""),
             "qwen": ("<|im_start|>user\n", "")}


def main() -> int:
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"
    lw = LayerwiseModel(MODEL, dtype="float16", device="cuda", attn_implementation="sdpa")
    tokenizer, device = lw.tokenizer, lw.device
    n_layers = lw.num_layers
    mid = MODEL.lower()
    user_open = next((w[0] for k, w in _WRAPPERS.items() if k in mid), "")
    ds = load_dataset("inputs/musique_s.json")
    qidx = int(np.random.default_rng(QSEED).permutation(len(ds))[0])
    print(f"[devimp-layers] model={MODEL} q#{qidx} ratios={RATIOS} layers={n_layers}", flush=True)

    # fused sequence = [prefix] + docs (NO query — it is prefilled after blending)
    doc_prompts, _q = build_qa_prompt(ds[qidx], QUERY_PROMPT)
    ctexts = [user_open + PREFIX_PROMPT] + list(doc_prompts)
    bos = tokenizer.bos_token_id
    chunks = []
    for i, text in enumerate(ctexts):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if i == 0 and bos is not None:
            ids = [bos] + ids
        chunks.append(Chunk(text=text, token_ids=ids, chunk_id=_stable_id(text, ids)))
    doc_slice = (1, len(chunks))

    # score ONCE per chunk (full); prune per ratio
    cmps = [backend.score(torch.tensor([c.token_ids], dtype=torch.long, device=device)).to(device)
            for c in chunks]

    out = {"model": MODEL, "qidx": qidx, "n_layers": n_layers, "ratios": {}}
    for r in RATIOS:
        surv = [token_prune(cc, CompressionBudget(ratio=r), reduce="mean")
                if (doc_slice[0] <= ci < doc_slice[1] and r < 1.0) else cc
                for ci, cc in enumerate(cmps)]
        lens = [cc.chunk_len for cc in surv]
        bounds = np.cumsum([0] + lens).tolist()
        S = bounds[-1]
        # cached K per layer + per-layer importance
        K_cached = [np.zeros((S, surv[0].key_cache[0].shape[-1]), dtype=np.float32)
                    for _ in range(n_layers)]
        imp_L = np.zeros((n_layers, S), dtype=np.float32)
        for ci, cc in enumerate(surv):
            s = bounds[ci]
            for li in range(n_layers):
                K_cached[li][s:s + cc.chunk_len] = cc.key_cache[li][0].float().cpu().numpy()
            imp_L[:, s:s + cc.chunk_len] = cc.importance.float().mean(dim=1).cpu().numpy()
        # fresh pre-RoPE K (one full forward over the fused survivor ids)
        surv_chunks = [Chunk(text="", token_ids=list(cc.token_ids), chunk_id=f"v{ci}")
                       for ci, cc in enumerate(surv)]
        with torch.inference_mode():
            lw.forward_layerwise(fused_input_ids(surv_chunks, device=device), use_cache=True)
        dev_L = np.zeros((n_layers, S), dtype=np.float32)
        for li in range(n_layers):
            fresh = lw.get_pre_rope_k(li)[0].float().cpu().numpy()
            dev_L[li] = ((fresh - K_cached[li]) ** 2).sum(-1)
        torch.cuda.empty_cache()
        out["ratios"][str(r)] = {"dev": dev_L.tolist(), "imp": imp_L.tolist(),
                                 "bounds": bounds}
        print(f"  kv={r}: S={S} done", flush=True)

    out_path = os.environ.get("CB_OUT", f"/tmp/devimp_layers_{MODEL.split('/')[-1]}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    print(f"WROTE {out_path}\nDEVIMP_LAYERS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
