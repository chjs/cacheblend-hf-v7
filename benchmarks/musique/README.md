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
├── blend_musique.py            # ORIGINAL, untouched (do not edit) — Mistral-7B
├── blend_musique_generic.py    # OUR model-agnostic version (editable) — any model
├── utils.py                    # hard copy of YaoJiayi/CacheBlend/example/utils.py
├── inputs/musique_s.json       # hard copy of YaoJiayi/CacheBlend/inputs/musique_s.json
├── _shim/vllm/__init__.py      # vllm.LLM + SamplingParams adapter — routes
│                               # generate() to LayerwiseModel + fuse_selective
├── run_blend_musique.py        # wrapper: sets sys.path + cwd, runpy-executes the workload
└── README.md
```

The hard-copied files (`blend_musique.py`, `utils.py`, `musique_s.json`) are
self-contained — no `external/` dependency.

### Two workloads — verbatim original + model-agnostic generic

| File | Models | Status |
|---|---|---|
| `blend_musique.py` | Mistral-7B-Instruct-v0.2 only | **verbatim** copy of YaoJiayi original — do NOT edit |
| `blend_musique_generic.py` | **any** HF instruction model | **our** generalization (editable) |

The YaoJiayi/CacheBlend repo ships only Mistral-7B experiment scripts (it
hard-codes Mistral's model id and `[INST]` token ids). The musique experiment
has exactly **one** model-dependent part — the instruction wrapper:

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

The runner picks the workload via `CACHEBLEND_WORKLOAD`:

```bash
# Mistral, via the verbatim original (default):
python benchmarks/musique/run_blend_musique.py

# Llama-3.1-8B, via the generic workload:
CACHEBLEND_WORKLOAD=blend_musique_generic.py \
CACHEBLEND_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    python benchmarks/musique/run_blend_musique.py

# Qwen2.5-7B, same generic workload — no new file:
CACHEBLEND_WORKLOAD=blend_musique_generic.py \
CACHEBLEND_MODEL=Qwen/Qwen2.5-7B-Instruct \
    python benchmarks/musique/run_blend_musique.py
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

From repo root:

### CPU smoke test (no GPU, no model load)

Verifies the entire scaffolding — sys.path injection, attribute chain access,
collect/check/normal dispatch, output plumbing — without loading Mistral-7B:

```bash
pip install -e .
pip install -r requirements.txt
CACHEBLEND_MOCK_MODEL=1 CACHEBLEND_MUSIQUE_N=2 python benchmarks/musique/run_blend_musique.py
```

Expected (truncated):

```
[run_blend_musique] mock model: 1
Loading dataset: inputs/musique_s.json
[run_blend_musique] CACHEBLEND_MUSIQUE_N=2 → slicing dataset to first 2 examples
Cached generation: [mock check 12 chunks]
TTFT with cache: 3.83e-06
Normal generation: [mock normal generation]
TTFT with full prefill: 2.92e-07
------------
...
---------------Result Summary---------------------
F1 with cache: 0.0
F1 with full prefill: 0.0
```

F1 = 0 in mock mode is expected — mock generation returns stub text.

### Real run on GPU

Requires a GPU with ≥16GB VRAM (Mistral-7B FP16 ≈ 14GB + activations + KV).

```bash
# On pod (pytorch:2.4.1-cuda12.4 image — torch already at correct version):
grep -v -E '^torch(\s|=|$)' requirements.txt > /tmp/reqs-no-torch.txt
pip install -r /tmp/reqs-no-torch.txt
pip install -e .
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Full 150-example sweep:
python benchmarks/musique/run_blend_musique.py

# Truncated to N examples:
CACHEBLEND_MUSIQUE_N=20 python benchmarks/musique/run_blend_musique.py
```

First run downloads Mistral-7B-Instruct-v0.2 (~14GB, ~8 min on HF Hub).

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `CACHEBLEND_MOCK_MODEL` | `0` | If `1`: skip model load; `generate()` returns stub text. Scaffolding-only test. |
| `CACHEBLEND_WORKLOAD` | `blend_musique.py` | Which workload to run: `blend_musique.py` (Mistral verbatim) or `blend_musique_generic.py` (any model). |
| `CACHEBLEND_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Model id used by `blend_musique_generic.py`. |
| `CACHEBLEND_DEVICE` | auto (`cuda` if available else `cpu`) | Device to load the model on. |
| `CACHEBLEND_DTYPE` | `float16` | Model dtype. |
| `CACHEBLEND_CHECK_LAYER` | `1` | `check_layer` arg to `fuse_selective`. |
| `CACHEBLEND_RECOMP_RATIO` | `0.15` | Default `recomp_ratio` (musique default). |
| `CACHEBLEND_ATTN_IMPL` | `sdpa` | `attn_implementation` for HF model. musique prompts reach ~7K tokens; eager OOMs on 24GB GPUs. |
| `CACHEBLEND_MUSIQUE_N` | (unset = all 150) | Slice `utils.load_dataset()[:N]`. Workload file untouched; truncation done via `utils` rebinding. |

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
