"""Multi-question check: within-chunk importance position pattern vs compression.

Importance only (KVzip scoring; NO blend forward needed — importance is frozen,
compression just selects survivors). For N questions, each doc chunk scored once;
at each KV ratio the chunk is importance-pruned, then on the SURVIVORS we measure:
  - sink            : importance of the first survivor token (the attention sink)
  - body fh / sh    : first-half vs second-half mean importance EXCLUDING the
                      first token (the visible chunk "body")
  - frac end-higher : fraction of chunks with sh > fh
  - 10-bin body curve (excl first token), averaged over all chunks/questions
Output JSON + summary. Env: CACHEBLEND_MODEL, CB_N (100), CB_KVZIP_VIS ("1.0,0.5,0.25").
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

from cacheblend.compress import CompressionBudget, token_prune, reduce_importance
from cacheblend.compress.kvzip import KVzipBackend, KVzipConfig

import importlib.util as _ilu
_us = _ilu.spec_from_file_location("_cbq_utils", str(HERE / "utils.py"))
_um = _ilu.module_from_spec(_us); _us.loader.exec_module(_um)
load_dataset, build_qa_prompt = _um.load_dataset, _um.build_qa_prompt

MODEL = os.environ["CACHEBLEND_MODEL"]
N = int(os.environ.get("CB_N", "100"))
RATIOS = [float(x) for x in os.environ.get("CB_KVZIP_VIS", "1.0,0.5,0.25").split(",")]
B = 10
QUERY_PROMPT = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"


def main() -> int:
    print(f"[imp-pos-multiq] model={MODEL} N={N} ratios={RATIOS}", flush=True)
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"
    tok = backend.tokenizer
    ds = load_dataset("inputs/musique_s.json")[:N]

    agg = {str(r): {"fh": [], "sh": [], "sink": [], "bins": np.zeros(B), "binc": np.zeros(B)}
           for r in RATIOS}
    for qi, ex in enumerate(ds):
        doc_prompts, _q = build_qa_prompt(ex, QUERY_PROMPT)
        for dp in doc_prompts:
            ids = tok(dp, add_special_tokens=False)["input_ids"]
            if len(ids) < 8:
                continue
            cc = backend.score(torch.tensor([ids], dtype=torch.long))
            for r in RATIOS:
                pc = token_prune(cc, CompressionBudget(ratio=r), reduce="mean") if r < 1.0 else cc
                imp = reduce_importance(pc.importance, "mean").cpu().numpy()  # [T] per-token
                T = len(imp)
                if T < 6:
                    continue
                a = agg[str(r)]
                a["sink"].append(float(imp[0]))
                body = imp[1:]                       # exclude first token (sink)
                h = len(body) // 2
                a["fh"].append(float(body[:h].mean())); a["sh"].append(float(body[h:].mean()))
                for j, v in enumerate(body):
                    k = min(int(j / len(body) * B), B - 1)
                    a["bins"][k] += v; a["binc"][k] += 1
            del cc
            torch.cuda.empty_cache()
        if (qi + 1) % 20 == 0:
            print(f"  [{qi+1}/{N}]", flush=True)

    out = {"model": MODEL, "N": N, "ratios": {}}
    print(f"\n{'='*64}", flush=True)
    for r in RATIOS:
        a = agg[str(r)]
        fh = np.array(a["fh"]); sh = np.array(a["sh"])
        curve = (a["bins"] / np.maximum(a["binc"], 1)).tolist()
        eh = float((sh > fh).mean())
        out["ratios"][str(r)] = {"n_chunks": len(fh), "frac_end_higher": eh,
                                 "fh_mean": float(fh.mean()), "sh_mean": float(sh.mean()),
                                 "sink_mean": float(np.mean(a["sink"])), "body_curve": curve}
        print(f"kv={r}: chunks={len(fh)}  sink={np.mean(a['sink']):.4f}  "
              f"body 전반={fh.mean():.4f} 후반={sh.mean():.4f}  후반>전반={eh*100:.0f}%", flush=True)
    out_path = os.environ.get("CB_OUT", f"/tmp/imp_pos_multiq_{MODEL.split('/')[-1]}.json")
    with open(out_path, "w") as fh_:
        json.dump(out, fh_)
    print(f"WROTE {out_path}\nIMP_POS_MULTIQ_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
