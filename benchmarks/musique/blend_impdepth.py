"""Importance-only STATIC gradual filtering (per-layer importance depth schedule).

Idea (static analogue of CacheBlend §4.3 gradual filtering): KVzip importance is
known for ALL layers offline, so we need no runtime deviation. Each candidate
token t gets a recompute DEPTH d_t and is recomputed at layers [check_layer..d_t]
(a prefix — to recompute at a layer it must be recomputed at all earlier layers,
the nested-set constraint). A token is recomputed deep iff it stays important deep.
Within the SAME token-layer budget as flat rc, recompute as MANY tokens as
possible (broad shallow coverage, narrowing with depth).

Depth assignment: per-layer importance score R[l,t] (head-reduce → per-layer
rank-normalize for cross-layer comparability); threshold tau (binary-searched to
hit budget B = rc * n_cand * n_stages); d_t = deepest layer l>=check with
R[l,t] >= tau, then +offset. Realized via fuse_selective's layer_scores (= d_t)
+ layer_budgets (= #{d_t >= layer} per layer) nested schedule.

Ablations (theory checks): head-reduce in {mean, max, ranknorm_max} (per-head
criticality / sink preservation), depth offset in {0 (=l), 1 (=l+1)}.
Arms: full, reuse, hkvd (only-HKVD, equal budget), impdepth@{reduce}_off{off}.
Env: CACHEBLEND_MODEL, CB_N(150), CB_KVZIP_RATIOS("0.7,0.5,0.3"), CB_RC("0.15").
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
RC = float(os.environ.get("CB_RC", "0.15"))
REDUCES = os.environ.get("CB_REDUCES", "mean,max,ranknorm_max").split(",")
OFFSETS = [int(x) for x in os.environ.get("CB_OFFSETS", "0,1").split(",")]
CHECK_LAYER = int(os.environ.get("CACHEBLEND_CHECK_LAYER", "1"))
MAX_NEW_TOKENS = 32

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


def per_layer_R(imp, mode):
    """imp [L,H,T] → R [L,T] in [0,1] (head-reduce, then per-layer rank-normalize)."""
    imp = imp.float()
    if mode == "mean":
        M = imp.mean(1)
    elif mode == "max":
        M = imp.amax(1)
    elif mode == "ranknorm_max":
        L, H, T = imp.shape
        r = imp.argsort(-1).argsort(-1).float() / max(1, T - 1)
        M = r.amax(1)
    else:
        raise ValueError(mode)
    T = M.shape[1]
    return M.argsort(-1).argsort(-1).float() / max(1, T - 1)      # [L,T] per-layer rank


def assign_depth(R_full, cand_mask, n_layers, check, budget, off):
    """R_full [L,total]; return d [total] long: recompute depth (0 = not recomputed,
    else layer index in [check..L-1]). Binary-search tau so Σ(d-check+1)≈budget."""
    total = R_full.shape[1]
    layers = torch.arange(check, n_layers)                       # [n_stages]
    Rc = R_full[check:n_layers]                                  # [n_stages, total]
    cand = cand_mask
    lay_idx = layers.view(-1, 1)                                 # [n_stages,1]

    def depth_for_tau(tau):
        above = (Rc >= tau) & cand.view(1, -1)                   # [n_stages,total]
        # deepest layer above tau per token (else <check → not recomputed)
        masked_layer = torch.where(above, lay_idx, torch.full_like(lay_idx, check - 1))
        dl = masked_layer.amax(0)                                # [total]
        d = torch.where(dl >= check, torch.clamp(dl + off, max=n_layers - 1),
                        torch.zeros_like(dl))
        d = torch.where(cand, d, torch.zeros_like(d))
        return d

    lo, hi = 0.0, 1.0
    d = depth_for_tau(0.0)
    for _ in range(24):
        mid = (lo + hi) / 2
        d = depth_for_tau(mid)
        cost = int(torch.clamp(d - (check - 1), min=0)[cand].sum().item())
        if cost > budget:
            lo = mid          # too much recompute → raise threshold
        else:
            hi = mid
    return depth_for_tau(hi).long()


def main() -> int:
    print(f"[impdepth] model={MODEL} N={N} kv={KVZIP_RATIOS} rc={RC} reduces={REDUCES} offsets={OFFSETS}", flush=True)
    backend = KVzipBackend(MODEL, KVzipConfig(kv_type="retain"))
    os.environ["COMPBLEND_KVZIP_NO_SYS_PROMPT"] = "1"
    lw = LayerwiseModel(MODEL, dtype="float16", device="cuda", attn_implementation="sdpa")
    model, tok, device = lw.model, lw.tokenizer, lw.device
    n_layers = lw.num_layers
    user_open, asst_open = _resolve_wrapper(MODEL, tok)
    ds = load_dataset("inputs/musique_s.json")[:N]

    def dec(o):
        return _decode(model, tok, o.logits, o.past_key_values)

    f1 = {"full": []}
    for r in KVZIP_RATIOS:
        f1[f"reuse@{r}"] = []
        f1[f"hkvd@{r}"] = []
        for red in REDUCES:
            for off in OFFSETS:
                f1[f"impdepth@{r}_{red}_off{off}"] = []

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
            survivor_cmps, full_ids_list, retained_pos_list, full_off = [], [], [], 0
            for ci, c in enumerate(orig):
                cc = cmp_full[c.chunk_id]; n_ci = cc.chunk_len
                if dslice.start <= ci < dslice.stop and r < 1.0:
                    k = budget.keep_k(n_ci)
                    keep = torch.sort(torch.topk(reduce_importance(cc.importance, "mean"), k).indices).values
                    cc = token_prune(cc, budget, reduce="mean")
                else:
                    keep = torch.arange(n_ci)
                survivor_cmps.append(cc)
                full_ids_list.extend(int(t) for t in orig[ci].token_ids)
                retained_pos_list.extend(full_off + int(i) for i in keep.tolist())
                full_off += n_ci

            blend_chunks, store = to_blend_inputs(survivor_cmps)
            full_input_ids = torch.tensor(full_ids_list, dtype=torch.long, device=device)
            retained_pos = torch.tensor(retained_pos_list, dtype=torch.long, device=device)
            offs = chunk_offsets(blend_chunks); total = offs[-1][1]
            q_start = offs[-1][0]                                  # query chunk start (forced)
            cand_mask = torch.zeros(total, dtype=torch.bool); cand_mask[:q_start] = True
            n_cand = int(cand_mask.sum()); n_forced = total - q_start
            n_stages = n_layers - CHECK_LAYER
            B = int(round(RC * n_cand * n_stages))

            # per-layer importance R [L,total] for each reduce mode (doc+prefix; query=0)
            R_by = {}
            for red in REDUCES:
                Rf = torch.zeros(n_layers, total)
                for ci in range(len(blend_chunks)):
                    s, T = offs[ci][0], offs[ci][1] - offs[ci][0]
                    if s < q_start:                                # candidate chunk
                        Rf[:, s:s + T] = per_layer_R(survivor_cmps[ci].importance, red)
                R_by[red] = Rf

            # reuse floor
            out = fuse_selective(lw, blend_chunks, store, recompute_ratio=0.0, check_layer=CHECK_LAYER,
                                 return_layerwise_output=True, force_last_chunk=True,
                                 full_input_ids=full_input_ids, retained_pos=retained_pos)
            f1[f"reuse@{r}"].append(max(compute_f1(dec(out), a, tok) for a in ans)); del out; torch.cuda.empty_cache()

            # only-HKVD baseline (flat rc, equal budget)
            out = fuse_selective(lw, blend_chunks, store, recompute_ratio=RC, check_layer=CHECK_LAYER,
                                 return_layerwise_output=True, force_last_chunk=True,
                                 full_input_ids=full_input_ids, retained_pos=retained_pos)
            f1[f"hkvd@{r}"].append(max(compute_f1(dec(out), a, tok) for a in ans)); del out; torch.cuda.empty_cache()

            for red in REDUCES:
                for off in OFFSETS:
                    d = assign_depth(R_by[red], cand_mask, n_layers, CHECK_LAYER, B, off).to(device)
                    # layer_scores[l,t] = d_t (broadcast); forced query → large (always kept)
                    big = float(n_layers + 1)
                    dvec = d.float().clone(); dvec[q_start:] = big
                    layer_scores = dvec.view(1, -1).expand(n_layers, -1).contiguous()
                    selection_scores = dvec.clone()
                    # layer_budgets[j] = #{d >= check+j among cand} + n_forced
                    budgets = []
                    for j in range(n_stages):
                        lay = CHECK_LAYER + j
                        cnt = int((d[:q_start] >= lay).sum().item())
                        budgets.append(cnt + n_forced)
                    out = fuse_selective(lw, blend_chunks, store, check_layer=CHECK_LAYER,
                                         return_layerwise_output=True, force_last_chunk=True,
                                         full_input_ids=full_input_ids, retained_pos=retained_pos,
                                         selection_scores=selection_scores, layer_scores=layer_scores,
                                         layer_budgets=budgets)
                    f1[f"impdepth@{r}_{red}_off{off}"].append(max(compute_f1(dec(out), a, tok) for a in ans))
                    del out; torch.cuda.empty_cache()
        if (qi + 1) % 10 == 0 or qi == 0:
            print(f"  [{qi+1}/{len(ds)}] full={np.mean(f1['full']):.3f}", flush=True)

    means = {k: float(np.mean(v)) for k, v in f1.items() if v}
    rng = np.random.default_rng(0)

    def ci(a, b):
        d = np.array(f1[a]) - np.array(f1[b])
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        lo, hi = np.quantile(bs, [.025, .975]); return d.mean(), lo, hi, (lo > 0 or hi < 0)

    print(f"\n{'='*80}\n== MEANS (N={len(ds)}) full={means['full']:.4f} ==", flush=True)
    for r in KVZIP_RATIOS:
        print(f"  kv={r}: reuse={means[f'reuse@{r}']:.4f}  hkvd(only)={means[f'hkvd@{r}']:.4f}", flush=True)
        for red in REDUCES:
            for off in OFFSETS:
                v = means[f"impdepth@{r}_{red}_off{off}"]
                dd, lo, hi, s = ci(f"impdepth@{r}_{red}_off{off}", f"hkvd@{r}")
                print(f"    impdepth {red:12s} off{off}: {v:.4f}  vs hkvd {dd:+.4f}[{lo:+.3f},{hi:+.3f}]{'★' if s else ''}", flush=True)

    out_path = os.environ.get("CB_OUT", f"/tmp/impdepth_{MODEL.split('/')[-1]}.json")
    with open(out_path, "w") as fh:
        json.dump({"model": MODEL, "N": len(ds), "kvzip_ratios": KVZIP_RATIOS, "rc": RC,
                   "reduces": REDUCES, "offsets": OFFSETS, "means": means, "f1": f1}, fh)
    print(f"WROTE {out_path}\nIMPDEPTH_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
