"""Comprehensive GPU verification of the compblend-branch fixes (C2/H1/H2/H3 + N1-3).

Loads Mistral-7B ONCE and checks every fixed feature end-to-end on a real model
(transformers 4.51.3). Run from benchmarks/musique/ so `utils` + the benchmark
helpers import. Env: CACHEBLEND_MODEL (default Mistral-7B-Instruct-v0.2),
VERIFY_N (F1 examples, default 12).

Checks:
  F2 (C2)  fuse_selective populates cache to full length + coherent decode.
  F3 (H2)  fused sequence has BOS at pos 0; full_recompute & selective same tokens.
  F4       ratio=1.0: fuse_selective ≡ fuse_full_recompute (bit-identical logits).
  F5 (H1)  fuse_full_reuse decodes against its OWN reuse cache (≠ full-recompute).
  F6 (H3)  force_last_chunk=True recomputes ALL query positions vs ~1 when False;
           F1 over N examples for full / selective(False) / selective(True).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
os.environ.setdefault("CACHEBLEND_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

from cacheblend import LayerwiseModel
from cacheblend.chunker import fused_input_ids
from cacheblend.kv_store import KVStore
from cacheblend.precompute import precompute_chunk_kv
from cacheblend.fusor import fuse_selective, fuse_full_recompute, fuse_full_reuse
from utils import load_dataset, build_qa_prompt, compute_f1
from blend_musique_generic import (
    _resolve_wrapper, _build_chunks, _greedy_decode, PREFIX_PROMPT, QUERY_PROMPT,
)

MODEL = os.environ["CACHEBLEND_MODEL"]
N = int(os.environ.get("VERIFY_N", "12"))
RATIO = 0.15
P = "✅"; X = "❌"
fails = []


def check(name, ok, detail=""):
    print(f"  {P if ok else X} {name}: {detail}", flush=True)
    if not ok:
        fails.append(name)


print(f"[verify_all] model={MODEL} N={N} ratio={RATIO}", flush=True)
lw = LayerwiseModel(MODEL, dtype="float16", device="cuda", attn_implementation="sdpa")
model, tok, device = lw.model, lw.tokenizer, lw.device
user_open, assistant_open = _resolve_wrapper(MODEL, tok)
ds = load_dataset("inputs/musique_s.json")[:N]


def build(ex):
    doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)
    ctexts = [user_open + PREFIX_PROMPT] + list(doc_prompts) + [q_prompt + assistant_open]
    return _build_chunks(tok, ctexts)


def precompute(chunks):
    st = KVStore()
    for c in chunks:
        K, V = precompute_chunk_kv(lw, c)
        st.put(c.chunk_id, K, V)
    return st


def decode(out):
    return _greedy_decode(model, tok, out.logits, out.past_key_values, device, time.perf_counter())[0]


# ───────────────────────── structural checks (example 0) ─────────────────────
print("\n── Structural checks (example 0) ──", flush=True)
chunks0 = build(ds[0])
store0 = precompute(chunks0)
total0 = sum(len(c.token_ids) for c in chunks0)
qstart0 = sum(len(c.token_ids) for c in chunks0[:-1])

# F3 (H2): BOS at position 0 of the fused sequence
fused0 = fused_input_ids(chunks0)
check("F3 H2: fused[0]==BOS", int(fused0[0, 0]) == tok.bos_token_id,
      f"fused[0]={int(fused0[0,0])} bos={tok.bos_token_id} total_tokens={total0}")

# F2 (C2): selective populates cache full length + finite logits + coherent decode
out_sel = fuse_selective(lw, chunks0, store0, recompute_ratio=RATIO, check_layer=1,
                         return_layerwise_output=True)
seq_before = out_sel.past_key_values.get_seq_length()   # capture BEFORE decode mutates it
fin_ok = bool(torch.isfinite(out_sel.logits[0, -1]).all())
check("F2 C2: cache full-length + finite logits", seq_before == total0 and fin_ok,
      f"cache_seq={seq_before}/{total0} finite={fin_ok}")
txt_sel = decode(out_sel)
check("F2 C2: coherent decode", len(txt_sel.strip()) > 0, f"answer={txt_sel!r}")

# F4: ratio=1.0 selective ≡ full_recompute (bit-identical logits path)
out_sel1 = fuse_selective(lw, chunks0, store0, recompute_ratio=1.0, check_layer=1,
                          return_layerwise_output=True)
out_full0 = fuse_full_recompute(lw, chunks0, return_layerwise_output=True)
maxdiff = (out_sel1.logits.float() - out_full0.logits.float()).abs().max().item()
check("F4: ratio=1.0 selective ≡ full_recompute", maxdiff == 0.0, f"max|Δlogits|={maxdiff:.2e}")

# F6a (H3): force_last_chunk=True recomputes ALL query positions; False ~1
_, top_T = fuse_selective(lw, chunks0, store0, recompute_ratio=RATIO, check_layer=1,
                          return_layerwise_output=True, return_hkvd_indices=True,
                          force_last_chunk=True)
_, top_F = fuse_selective(lw, chunks0, store0, recompute_ratio=RATIO, check_layer=1,
                          return_layerwise_output=True, return_hkvd_indices=True,
                          force_last_chunk=False)
qlen = total0 - qstart0
qT = sum(1 for p in range(qstart0, total0) if p in set(top_T.tolist()))
qF = sum(1 for p in range(qstart0, total0) if p in set(top_F.tolist()))
check("F6a H3: force_last_chunk=True recomputes ALL query", qT == qlen and qT > qF,
      f"query_recomputed True={qT}/{qlen} False={qF}/{qlen}")

# F5 (H1): full_reuse decodes against its OWN cache, which differs from full-recompute.
# Capture caches BEFORE any decode (greedy decode appends to past_key_values).
out_reuse = fuse_full_reuse(lw, chunks0, store0, return_layerwise_output=True)
li = lw.num_layers - 1
k_reuse = out_reuse.past_key_values.key_cache[li].clone()   # pre-decode
k_full = out_full0.past_key_values.key_cache[li].clone()    # out_full0 was never decoded
kdiff = (k_reuse.float() - k_full.float()).abs().max().item()
check("F5 H1: reuse cache ≠ full-recompute cache", kdiff > 1e-3, f"last-layer max|ΔK|={kdiff:.3f}")
txt_reuse = decode(out_reuse)
check("F5 H1: full_reuse coherent decode", len(txt_reuse.strip()) > 0, f"answer={txt_reuse!r}")

# ───────────────────────── F1 over N examples ────────────────────────────────
print(f"\n── F1 over {N} examples (full / selective F={RATIO} legacy / realistic) ──", flush=True)
f1_full, f1_legacy, f1_real = [], [], []
for i, ex in enumerate(ds):
    chunks = build(ex)
    store = precompute(chunks)
    ans = ex["answers"]
    o = fuse_full_recompute(lw, chunks, return_layerwise_output=True)
    f1_full.append(max(compute_f1(decode(o), a, tok) for a in ans))
    o = fuse_selective(lw, chunks, store, recompute_ratio=RATIO, check_layer=1,
                       return_layerwise_output=True, force_last_chunk=False)
    f1_legacy.append(max(compute_f1(decode(o), a, tok) for a in ans))
    o = fuse_selective(lw, chunks, store, recompute_ratio=RATIO, check_layer=1,
                       return_layerwise_output=True, force_last_chunk=True)
    f1_real.append(max(compute_f1(decode(o), a, tok) for a in ans))
    print(f"  [{i+1}/{N}] full={f1_full[-1]:.3f} legacy={f1_legacy[-1]:.3f} realistic={f1_real[-1]:.3f}", flush=True)

print(f"\n── F1 means (N={N}) ──", flush=True)
print(f"  full_recompute            : {np.mean(f1_full):.4f}", flush=True)
print(f"  selective legacy   (F=0)  : {np.mean(f1_legacy):.4f}", flush=True)
print(f"  selective realistic(F=1)  : {np.mean(f1_real):.4f}", flush=True)

print(f"\n{'='*52}", flush=True)
if fails:
    print(f"VERIFY_ALL: {len(fails)} CHECK(S) FAILED: {fails}", flush=True)
    sys.exit(1)
print("VERIFY_ALL: ALL STRUCTURAL CHECKS PASSED", flush=True)
