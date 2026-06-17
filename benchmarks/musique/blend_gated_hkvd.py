"""Gated HKVD sweep (paper §3 two-stage selector) on the corrected full-token New KV.

Stage 1 (gate): C = top-g fraction of candidate (non-query) retained tokens by
compression-derived importance. Stage 2 (select): HKVD top-k within C. g=1.0 == no
gate == only-HKVD baseline; smaller g lets importance dominate. Finds the gate ratio
that maximizes F1, and whether Gated HKVD beats only-HKVD on the CLEAN deviation
(prior 'gate hurts' results were on the buggy depleted-context deviation).

New KV is computed over the full pre-eviction text (full_input_ids + retained_pos);
evicted tokens never enter the pool. Arms: full_prefill_all, full_reuse_kvzip@kv,
gated@kv_rc_g for each g. Env: CACHEBLEND_MODEL, CB_N(150),
CB_KVZIP_RATIOS("0.7,0.5,0.3"), CB_RECOMP_RATIOS("0.15"),
CB_GATE_RATIOS("1.0,0.7,0.5,0.35,0.25"), CB_REDUCE(mean).
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
from cacheblend.chunker import Chunk, _stable_id, chunk_offsets
from cacheblend.fusor import fuse_selective, fuse_full_recompute
from cacheblend.compress import CompressionBudget, token_prune, to_blend_inputs, reduce_importance
from cacheblend.compress.kvzip import KVzipBackend, KVzipConfig

import importlib.util as _ilu
_us = _ilu.spec_from_file_location("_cbq_utils", str(HERE / "utils.py"))
_um = _ilu.module_from_spec(_us); _us.loader.exec_module(_um)
load_dataset, build_qa_prompt, compute_f1 = _um.load_dataset, _um.build_qa_prompt, _um.compute_f1

MODEL = os.environ["CACHEBLEND_MODEL"]
N = int(os.environ.get("CB_N", "150"))
KVZIP_RATIOS = [float(x) for x in os.environ.get("CB_KVZIP_RATIOS", "0.7,0.5,0.3").split(",")]
RECOMP_RATIOS = [float(x) for x in os.environ.get("CB_RECOMP_RATIOS", "0.15").split(",")]
GATE_RATIOS = [float(x) for x in os.environ.get("CB_GATE_RATIOS", "1.0,0.7,0.5,0.35,0.25").split(",")]
REDUCE = os.environ.get("CB_REDUCE", "mean")
CHECK_LAYER = int(os.environ.get("CACHEBLEND_CHECK_LAYER", "1"))
MAX_NEW_TOKENS = 32

PREFIX_PROMPT = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
QUERY_PROMPT = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
_WRAPPERS = {
    "mistral": ("[INST]", "[/INST]"),
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
    print(f"[gated] model={MODEL} N={N} kv={KVZIP_RATIOS} rc={RECOMP_RATIOS} gates={GATE_RATIOS} reduce={REDUCE}", flush=True)
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"
    lw = LayerwiseModel(MODEL, dtype="float16", device="cuda", attn_implementation="sdpa")
    model, tok, device = lw.model, lw.tokenizer, lw.device
    user_open, asst_open = _resolve_wrapper(MODEL, tok)
    ds = load_dataset("inputs/musique_s.json")[:N]

    def dec(o):
        return _decode(model, tok, o.logits, o.past_key_values)

    f1 = {"full": []}
    for r in KVZIP_RATIOS:
        f1[f"reuse@{r}"] = []
        for rr in RECOMP_RATIOS:
            for g in GATE_RATIOS:
                f1[f"gated@{r}_{rr}_{g}"] = []

    for qi, ex in enumerate(ds):
        ans = ex["answers"]
        docs, qp = build_qa_prompt(ex, QUERY_PROMPT)
        ctexts = [user_open + PREFIX_PROMPT] + list(docs) + [qp + asst_open]
        orig = _build_chunks(tok, ctexts)
        dslice = slice(1, 1 + len(docs))

        out = fuse_full_recompute(lw, orig, return_layerwise_output=True)
        f1["full"].append(max(compute_f1(dec(out), a, tok) for a in ans)); del out
        torch.cuda.empty_cache()

        cmp_full = {c.chunk_id: backend.score(torch.tensor([c.token_ids], dtype=torch.long, device=device)).to(device) for c in orig}

        for r in KVZIP_RATIOS:
            budget = CompressionBudget(ratio=r)
            survivor_cmps = []
            full_ids_list, retained_pos_list = [], []
            full_off = 0
            for ci, c in enumerate(orig):
                cc = cmp_full[c.chunk_id]
                n_ci = cc.chunk_len
                if dslice.start <= ci < dslice.stop and r < 1.0:
                    k = budget.keep_k(n_ci)
                    imp_tok = reduce_importance(cc.importance, REDUCE)
                    keep = torch.sort(torch.topk(imp_tok, k).indices).values
                    cc = token_prune(cc, budget, reduce=REDUCE)
                else:
                    keep = torch.arange(n_ci)
                survivor_cmps.append(cc)
                full_ids_list.extend(int(t) for t in orig[ci].token_ids)
                retained_pos_list.extend(full_off + int(i) for i in keep.tolist())
                full_off += n_ci

            blend_chunks, store = to_blend_inputs(survivor_cmps)
            full_input_ids = torch.tensor(full_ids_list, dtype=torch.long, device=device)
            retained_pos = torch.tensor(retained_pos_list, dtype=torch.long, device=device)

            # retained-pool per-token importance + query-chunk span (forced, gate-bypass)
            offs = chunk_offsets(blend_chunks)
            total = offs[-1][1]
            imp_pool = torch.cat([reduce_importance(sc.importance, REDUCE) for sc in survivor_cmps]).to(device)
            q_start = offs[-1][0]                              # last chunk (query) start in retained pool
            cand = torch.zeros(total, dtype=torch.bool, device=device)
            cand[:q_start] = True                             # candidates = everything except query

            # reuse floor
            out = fuse_selective(lw, blend_chunks, store, recompute_ratio=0.0, check_layer=CHECK_LAYER,
                                 return_layerwise_output=True, force_last_chunk=True,
                                 full_input_ids=full_input_ids, retained_pos=retained_pos)
            f1[f"reuse@{r}"].append(max(compute_f1(dec(out), a, tok) for a in ans)); del out
            torch.cuda.empty_cache()

            for rr in RECOMP_RATIOS:
                n_cand = int(cand.sum().item())
                for g in GATE_RATIOS:
                    if g >= 1.0:
                        gate_mask = None                      # no gate = only-HKVD
                    else:
                        n_keep = max(1, int(round(g * n_cand)))
                        gate_mask = torch.zeros(total, dtype=torch.bool, device=device)
                        gate_mask[q_start:] = True            # query bypasses gate
                        sc = imp_pool.clone(); sc[~cand] = float("-inf")
                        topi = torch.topk(sc, min(n_keep, n_cand)).indices
                        gate_mask[topi] = True
                    out = fuse_selective(lw, blend_chunks, store, recompute_ratio=rr, check_layer=CHECK_LAYER,
                                         return_layerwise_output=True, force_last_chunk=True,
                                         full_input_ids=full_input_ids, retained_pos=retained_pos,
                                         gate_mask=gate_mask)
                    f1[f"gated@{r}_{rr}_{g}"].append(max(compute_f1(dec(out), a, tok) for a in ans)); del out
                    torch.cuda.empty_cache()
        if (qi + 1) % 10 == 0 or qi == 0:
            print(f"  [{qi+1}/{len(ds)}] full={np.mean(f1['full']):.3f}", flush=True)

    means = {k: float(np.mean(v)) for k, v in f1.items() if v}
    rng = np.random.default_rng(0)

    def ci(a, b):
        d = np.array(f1[a]) - np.array(f1[b])
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        lo, hi = np.quantile(bs, [.025, .975]); return d.mean(), lo, hi, (lo > 0 or hi < 0)

    print(f"\n{'='*78}\n== MEANS (N={len(ds)}) full={means['full']:.4f} ==", flush=True)
    for r in KVZIP_RATIOS:
        print(f"  kv={r}: reuse={means[f'reuse@{r}']:.4f}", flush=True)
        for rr in RECOMP_RATIOS:
            base = means[f"gated@{r}_{rr}_{GATE_RATIOS[0]}"] if GATE_RATIOS[0] >= 1.0 else None
            for g in GATE_RATIOS:
                v = means[f"gated@{r}_{rr}_{g}"]
                tag = "(only-HKVD)" if g >= 1.0 else ""
                line = f"    rc={rr} gate={g}: {v:.4f} {tag}"
                if base is not None and g < 1.0:
                    d, lo, hi, s = ci(f"gated@{r}_{rr}_{g}", f"gated@{r}_{rr}_{GATE_RATIOS[0]}")
                    line += f"  vs onlyHKVD {d:+.4f}[{lo:+.3f},{hi:+.3f}]{'★' if s else ''}"
                print(line, flush=True)

    out_path = os.environ.get("CB_OUT", f"/tmp/gated_{MODEL.split('/')[-1]}.json")
    with open(out_path, "w") as fh:
        json.dump({"model": MODEL, "N": len(ds), "kvzip_ratios": KVZIP_RATIOS, "recomp_ratios": RECOMP_RATIOS,
                   "gate_ratios": GATE_RATIOS, "means": means, "f1": f1}, fh)
    print(f"WROTE {out_path}\nGATED_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
