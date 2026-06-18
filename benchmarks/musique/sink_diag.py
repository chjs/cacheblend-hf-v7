"""Diagnostic: does KVzip importance COVER attention-sink tokens?

Attention sink (StreamingLLM): initial tokens absorb excess softmax mass —
positional, emerge at layer >=2. In PIC/blend (EPIC) each chunk's first tokens
become local sinks during isolated prefill. KVzip importance = attention-received,
so sinks (high attention-received) SHOULD be high-importance — but sinks live at
deep layers, so single-layer signals may miss them.

This measures, in the ISOLATED per-chunk prefill frame (matching importance):
  - attention_received[l,t] = sum over query positions of attn weight to token t,
    mean over heads (eager attention) → identifies empirical sinks.
  - importance[l,t] = KVzip per-(layer,head,token) score, head-reduced.
Sink positions = first 4 tokens of each chunk (positional, StreamingLLM/EPIC).

Reports, per layer (aggregated over chunks x questions):
  (1) attn-received percentile of the first-4 (sink) tokens  → are they sinks? when (layer>=2)?
  (2) importance percentile of the first-4 tokens            → does importance see them?
  (3) coverage: fraction of empirical top-attn-received tokens captured by top-importance.
  (4) Spearman(importance, attn_received) per layer.
Env: CACHEBLEND_MODEL, CB_N(100), CB_SINK_K(4), CB_REDUCE(mean), CB_TOPQ(0.15).
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

from cacheblend.compress.kvzip import KVzipBackend, KVzipConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import importlib.util as _ilu
_us = _ilu.spec_from_file_location("_cbq_utils", str(HERE / "utils.py"))
_um = _ilu.module_from_spec(_us); _us.loader.exec_module(_um)
load_dataset, build_qa_prompt = _um.load_dataset, _um.build_qa_prompt

MODEL = os.environ["CACHEBLEND_MODEL"]
N = int(os.environ.get("CB_N", "100"))
SINK_K = int(os.environ.get("CB_SINK_K", "4"))
REDUCE = os.environ.get("CB_REDUCE", "mean")
TOPQ = float(os.environ.get("CB_TOPQ", "0.15"))
PREFIX_PROMPT = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
QUERY_PROMPT = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"


def head_reduce(imp, mode):           # imp [L,H,T] -> [L,T]
    imp = imp.float()
    if mode == "mean":
        return imp.mean(1)
    if mode == "max":
        return imp.amax(1)
    if mode == "ranknorm_max":
        T = imp.shape[-1]
        r = imp.argsort(-1).argsort(-1).float() / max(1, T - 1)
        return r.amax(1)
    raise ValueError(mode)


def rank01(x):                        # [T] -> percentile rank in [0,1]
    T = x.shape[0]
    return x.argsort().argsort().float() / max(1, T - 1)


def main() -> int:
    print(f"[sink_diag] model={MODEL} N={N} sink_k={SINK_K} reduce={REDUCE} topq={TOPQ}", flush=True)
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="eager", device_map={"": 0})
    model.eval()
    device = model.device
    n_layers = model.config.num_hidden_layers
    ds = load_dataset("inputs/musique_s.json")[:N]

    # per-layer accumulators
    sink_attn_pct = [[] for _ in range(n_layers)]   # percentile of first-K tokens by attn-received
    sink_imp_pct = [[] for _ in range(n_layers)]    # percentile of first-K tokens by importance
    body_attn_pct = [[] for _ in range(n_layers)]   # percentile of non-sink tokens (baseline ~0.5)
    corr = [[] for _ in range(n_layers)]            # spearman(imp, attn) per layer
    cover = [[] for _ in range(n_layers)]           # frac of top-attn tokens in top-imp (per layer)
    cover_meanimp = []                              # coverage using mean-over-layer importance

    bos = tok.bos_token_id
    for qi, ex in enumerate(ds):
        docs, qp = build_qa_prompt(ex, QUERY_PROMPT)
        # per-chunk token id lists (doc chunks only — the compressed reusable context)
        chunk_ids = [tok(d, add_special_tokens=False)["input_ids"] for d in docs]
        for ids in chunk_ids:
            T = len(ids)
            if T < SINK_K + 4:
                continue
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)
            # importance (KVzip, isolated)
            imp = backend.score(ids_t).importance.to("cpu")     # [L,Hkv,T]
            imp_LT = head_reduce(imp, REDUCE)                    # [L,T]
            # attention-received (eager, isolated)
            with torch.inference_mode():
                out = model(input_ids=ids_t, output_attentions=True, use_cache=False)
            attns = out.attentions                              # tuple L of [1,H,T,T]
            sink = list(range(min(SINK_K, T)))
            imp_meanlayer = imp_LT.mean(0)                       # [T]
            for li in range(n_layers):
                a = attns[li][0].float()                        # [H,Tq,Tk]
                recv = a.sum(1).mean(0).cpu()                   # [Tk] attention received
                ra = rank01(recv); ri = rank01(imp_LT[li])
                sink_attn_pct[li].append(float(ra[sink].mean()))
                sink_imp_pct[li].append(float(ri[sink].mean()))
                body = [t for t in range(T) if t not in sink]
                body_attn_pct[li].append(float(ra[body].mean()))
                # spearman ~ pearson of ranks
                corr[li].append(float(np.corrcoef(ra.numpy(), ri.numpy())[0, 1]))
                # coverage: top-q by attn-received captured by top-q by importance
                k = max(1, int(T * TOPQ))
                top_a = set(torch.topk(recv, k).indices.tolist())
                top_i = set(torch.topk(imp_LT[li], k).indices.tolist())
                cover[li].append(len(top_a & top_i) / len(top_a))
            # coverage of sinks by mean-layer importance top-q
            k = max(1, int(T * TOPQ))
            top_i_mean = set(torch.topk(imp_meanlayer, k).indices.tolist())
            cover_meanimp.append(len(set(sink) & top_i_mean) / len(sink))
            del attns, out
            torch.cuda.empty_cache()
        if (qi + 1) % 10 == 0 or qi == 0:
            print(f"  [{qi+1}/{len(ds)}]", flush=True)

    def m(x):
        return float(np.mean(x)) if x else float("nan")

    print(f"\n{'='*78}\n== SINK DIAG (N={len(ds)}, sink=first {SINK_K}, reduce={REDUCE}) ==", flush=True)
    print(f"{'layer':>5} {'attn_pct(sink)':>14} {'attn_pct(body)':>14} {'imp_pct(sink)':>13} {'corr(imp,attn)':>14} {'cover@%d%%' % int(TOPQ*100):>9}", flush=True)
    summary = {}
    for li in range(n_layers):
        print(f"{li:>5} {m(sink_attn_pct[li]):>14.3f} {m(body_attn_pct[li]):>14.3f} {m(sink_imp_pct[li]):>13.3f} {m(corr[li]):>14.3f} {m(cover[li]):>9.3f}", flush=True)
        summary[li] = dict(attn_sink=m(sink_attn_pct[li]), attn_body=m(body_attn_pct[li]),
                           imp_sink=m(sink_imp_pct[li]), corr=m(corr[li]), cover=m(cover[li]))
    # layer bands
    def band(acc, lo, hi):
        return m([v for li in range(lo, hi) for v in acc[li]])
    print(f"\nBANDS  layers 0-1 vs 2+:", flush=True)
    print(f"  attn_pct(sink): L0-1={band(sink_attn_pct,0,2):.3f}  L2+={band(sink_attn_pct,2,n_layers):.3f}", flush=True)
    print(f"  imp_pct(sink):  L0-1={band(sink_imp_pct,0,2):.3f}  L2+={band(sink_imp_pct,2,n_layers):.3f}", flush=True)
    print(f"  cover@{int(TOPQ*100)}%:     L0-1={band(cover,0,2):.3f}  L2+={band(cover,2,n_layers):.3f}", flush=True)
    print(f"  sink covered by mean-layer-imp top-{int(TOPQ*100)}%: {m(cover_meanimp):.3f}", flush=True)

    out_path = os.environ.get("CB_OUT", f"/tmp/sinkdiag_{MODEL.split('/')[-1]}.json")
    with open(out_path, "w") as fh:
        json.dump({"model": MODEL, "N": len(ds), "sink_k": SINK_K, "reduce": REDUCE, "topq": TOPQ,
                   "per_layer": summary, "cover_meanimp": m(cover_meanimp)}, fh)
    print(f"WROTE {out_path}\nSINKDIAG_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
