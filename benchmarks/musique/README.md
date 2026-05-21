# benchmarks/musique/ — Run the original YaoJiayi/CacheBlend `blend_musique.py` unmodified

`blend_musique.py` is a **verbatim copy** of
[`example/blend_musique.py`](https://github.com/YaoJiayi/CacheBlend/blob/main/example/blend_musique.py)
from the YaoJiayi/CacheBlend repo.

> **Rule**: do not modify `blend_musique.py`. Any plumbing changes go in
> wrappers, adapters, import paths, or environment — never in the reference
> file itself.

The original script targets the YaoJiayi vLLM fork (custom `cache_fuse_metadata`
+ `hack_kv` + `old_kvs` hooks). This directory provides the surrounding
scaffolding that lets the same file run unmodified against the HF-Transformers
based CacheBlend impl in `src/cacheblend/`:

```
benchmarks/musique/
├── blend_musique.py            # ORIGINAL, untouched (do not edit) — Mistral-7B, needs the shim
├── blend_musique_generic.py    # OUR model-agnostic version (editable) — any model, shim-free
├── utils.py                    # hard copy of YaoJiayi/CacheBlend/example/utils.py
├── inputs/musique_s.json       # hard copy of YaoJiayi/CacheBlend/inputs/musique_s.json
├── _shim/vllm/__init__.py      # vllm.LLM adapter — ONLY needed by blend_musique.py
├── run_blend_musique.py        # wrapper for blend_musique.py (sets up shim + cwd, runpy)
└── README.md
```

The hard-copied files (`blend_musique.py`, `utils.py`, `musique_s.json`) are
self-contained — no `external/` dependency.

### Two workloads — verbatim original (shim) + model-agnostic generic (direct)

| File | Models | How it runs | Status |
|---|---|---|---|
| `blend_musique.py` | Mistral-7B-Instruct-v0.2 only | via `run_blend_musique.py` + `_shim/vllm/` | **verbatim** YaoJiayi original — do NOT edit |
| `blend_musique_generic.py` | **any** HF instruction model | **standalone**, calls `cacheblend` directly | **our** generalization (editable) |

`blend_musique.py` targets the YaoJiayi vLLM fork (`cache_fuse_metadata` /
`hack_kv` hooks). To run it unmodified, `_shim/vllm/` provides a fake `vllm`
that routes those calls to our HF code. **The shim exists only for the
verbatim original.**

`blend_musique_generic.py` does NOT use the shim. It calls
`precompute_chunk_kv` / `fuse_selective` / `fuse_full_recompute` directly —
no `vllm` import, no `cache_fuse_metadata`, no runner needed.

The musique experiment has exactly **one** model-dependent part — the
instruction wrapper:

| Model | wrapper |
|---|---|
| Mistral | `[INST] ... [/INST]` |
| Llama-3.1 | `<\|start_header_id\|>user<\|end_header_id\|> ... <\|eot_id\|>...assistant...` |
| Qwen | `<\|im_start\|>user ... <\|im_end\|>...` |

Everything else (dataset, prompts, `build_qa_prompt`, chunking, `compute_f1`,
CacheBlend-vs-full comparison) is model-independent. So **there is no need for
a new script per model.** `blend_musique_generic.py` picks the wrapper from a
small per-family table — and for an unknown family derives it from the
tokenizer's chat template. A new model is just `CACHEBLEND_MODEL=...`.

**Tokenization consistency** — `blend_musique_generic.py` tokenizes each chunk
once and feeds the *same* `fused_input_ids(chunks)` to both `fuse_selective`
and `fuse_full_recompute`. So at `recompute_ratio=1.0` the two paths are
**bit-identical** (verified: 20/20 examples). The shim path, by contrast,
re-encoded prompt strings — `encode(A)+encode(B) ≠ encode(A+B)` — so it
diverged on a few examples even at ratio=1.0. The direct path removes that
tokenization confound.

```bash
# Mistral, verbatim original — via the shim + runner:
python benchmarks/musique/run_blend_musique.py

# Llama-3.1-8B — generic workload, standalone (no shim, no runner):
CACHEBLEND_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    python benchmarks/musique/blend_musique_generic.py

# Qwen2.5-7B — same generic workload, no new file:
CACHEBLEND_MODEL=Qwen/Qwen2.5-7B-Instruct \
    python benchmarks/musique/blend_musique_generic.py
```

Note: gated models (Llama, Mistral) require `huggingface-cli login` with a
token that has accepted the model license.

## How it works

The original file relies on `vllm.LLM` plus a deeply-nested attribute chain
that lets user code mutate the model's per-forward behavior:

```python
cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
cache_fuse_metadata['collect'] = True    # capture per-chunk K/V into layers[j].self_attn.hack_kv
cache_fuse_metadata['check']   = True    # selective recompute against ...model.model.old_kvs
```

Our shim exposes the same attribute chain but routes the underlying compute
to `cacheblend`:

| `cache_fuse_metadata` flag | Original (vLLM fork) | Shim (HF) |
|---|---|---|
| `collect=True` | per-chunk standalone forward, hack_kv = post-RoPE K/V | per-chunk `precompute_chunk_kv` → KVStore (pre-RoPE K + V); hack_kv = zero-placeholder so user slicing doesn't crash |
| `check=True` | selective recompute using accumulated `old_kvs` | `fuse_selective(tracked_chunks, kv_store, recomp_ratio)` → greedy decode |
| both False | normal full prefill | HF model forward + greedy decode |

`old_kvs` assignments from user code are recorded but unused — the real KV
cache lives in the shim's internal `KVStore`. `hack_kv` placeholders are
sized just-enough that user-code slicing (`[:s_start_len]`,
`[s_start_1_len : len(...)+1]`) does not raise.

**Chunk-boundary parity**: chunk 0 (the `[INST]` + prefix prompt) is stored
with the BOS token included (it sits at fused position 0). Chunks 1..N
(documents and the trailing `[/INST]` query) are stored with BOS stripped,
matching the original's `[s_start_1_len : len(doc_chunk_ids[i])+1]` slice
which drops BOS for non-first chunks.

## Usage

From repo root.

### Install

```bash
pip install -e .
pip install -r requirements.txt   # on a GPU pod with torch preinstalled, strip the torch line first
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### `blend_musique_generic.py` — recommended (shim-free, any model)

Standalone — no runner, no shim:

```bash
# Mistral-7B, full 150-example sweep:
CACHEBLEND_MODEL=mistralai/Mistral-7B-Instruct-v0.2 \
    python benchmarks/musique/blend_musique_generic.py

# Llama-3.1-8B:
CACHEBLEND_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    python benchmarks/musique/blend_musique_generic.py

# Truncate to N examples for a quick check:
CACHEBLEND_MUSIQUE_N=20 CACHEBLEND_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    python benchmarks/musique/blend_musique_generic.py
```

First run downloads the model (~14-16GB, ~8 min on HF Hub). Requires a GPU
with ≥16GB (Mistral-7B) / ≥20GB (Llama-3.1-8B) VRAM.

### `blend_musique.py` — the YaoJiayi verbatim original (Mistral only, via shim)

Runs the unmodified original through the vLLM shim + runner:

```bash
# CPU scaffolding smoke (no model load):
CACHEBLEND_MOCK_MODEL=1 CACHEBLEND_MUSIQUE_N=2 python benchmarks/musique/run_blend_musique.py

# Real run on GPU:
python benchmarks/musique/run_blend_musique.py
```

`CACHEBLEND_MOCK_MODEL` and the runner exist only for this verbatim-original
path — `blend_musique_generic.py` does not use them.

## Environment variables

`blend_musique_generic.py` reads these directly:

| Variable | Default | Effect |
|---|---|---|
| `CACHEBLEND_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | HF model id. |
| `CACHEBLEND_DTYPE` | `float16` | Model dtype. |
| `CACHEBLEND_ATTN_IMPL` | `sdpa` | `attn_implementation`. musique prompts reach ~7K tokens; eager OOMs on 24GB GPUs. |
| `CACHEBLEND_CHECK_LAYER` | `1` | `check_layer` arg to `fuse_selective`. |
| `CACHEBLEND_RECOMP_RATIO` | `0.15` | Recompute ratio (musique default). |
| `CACHEBLEND_MUSIQUE_N` | (unset = all 150) | Run only the first N examples. |

`blend_musique.py` (verbatim original, via the shim) additionally honours
`CACHEBLEND_MOCK_MODEL`, `CACHEBLEND_WORKLOAD`, `CACHEBLEND_DEVICE` — these are
shim/runner concepts and do not apply to the generic workload.

## Original ↔ shim mapping

| Original line | Original op | Shim handling |
|---|---|---|
| `from vllm import LLM, SamplingParams` | import vllm | `_shim/vllm/` injected onto sys.path |
| `LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.5)` | spin up vLLM engine | `LayerwiseModel(...)` + build fake attribute chain |
| `llm.set_tokenizer(tokenizer)` | install tokenizer | mock mode: adopt user-passed; real mode: use LayerwiseModel's own |
| `llm.llm_engine...cache_fuse_metadata` | mutable per-forward flags | dict in `_FakeInnerModel` |
| `cache_fuse_metadata['collect']=True; llm.generate([chunk_text], max_tokens=1)` | per-chunk K/V capture | re-encode, `precompute_chunk_kv` → KVStore; populate `hack_kv` placeholder |
| `llm_layers[j].self_attn.hack_kv` | per-layer (K, V) post-forward | zero-tensor placeholder; real K/V in KVStore |
| `model.old_kvs = chunk_past_key_values` | install fused KV cache | recorded but unused |
| `cache_fuse_metadata['check']=True; llm.generate([input_prompt], max_tokens=32)` | selective recompute + decode | `fuse_selective(...)` + greedy decode |
| `cache_fuse_metadata['check']=False; llm.generate(...)` | full prefill + decode | HF `model(...)` + greedy decode |
| `output[0].outputs[0].text` | generated text | `_Completion(text=...)` |
| `output[0].metrics.first_token_time - .first_scheduled_time` | TTFT | `_Metrics(first_scheduled_time, first_token_time)` from `time.perf_counter()` brackets |

## Known differences from the original vLLM fork

1. **TTFT semantics**: original measures inside vLLM's request scheduler;
   shim uses `time.perf_counter()` brackets around prefill → first decoded
   token. Approximation, not identical.
2. **`old_kvs` is unused**: user-code accumulator is ignored. Real cache
   flows through the shim's internal `KVStore`, keyed by `chunker._stable_id`.
3. **`hack_kv` is zero-filled**: user-code reads are not consumed for our
   compute path. To expose real post-RoPE K/V (e.g. for byte-level cross-check
   with vLLM), modify `_populate_hack_kv` in the shim.
4. **HF eager/SDPA vs vLLM PagedAttention**: differences at the last few bits
   of `softmax(QK^T)`. Should not affect F1 at the token level.

See `_shim/vllm/__init__.py` for shim-internal notes.

## Reproduced results (N=150 each)

### Mistral-7B-Instruct-v0.2 — `blend_musique.py` (verbatim original), RTX 3090 24GB

| Metric | CacheBlend | Full Prefill | Delta |
|---|---|---|---|
| TTFT (mean) | **0.566 s** | 2.271 s | **4.01x speedup** |
| F1 (mean) | 0.2576 | 0.2758 | -0.018 (**93.4% retention**) |

### Llama-3.1-8B-Instruct — `blend_musique_generic.py`, A100 80GB

| Metric | CacheBlend | Full Prefill | Delta |
|---|---|---|---|
| TTFT (mean) | **0.171 s** | 0.555 s | **3.24x speedup** |
| F1 (mean) | 0.2944 | 0.3111 | -0.017 (**94.6% retention**) |

Both match the YaoJiayi paper's musique claims (3-4x TTFT, 90-95% F1
retention). Absolute TTFT is not comparable across the two rows (different
GPUs); the speedup ratio and F1 retention are the model-comparable signals.
The Llama run validates that the model-agnostic `blend_musique_generic.py`
reproduces the experiment correctly on a non-Mistral model.
