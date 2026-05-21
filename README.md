# cacheblend-hf-v7

Slim, self-contained distribution of the HuggingFace-Transformers
reimplementation of **CacheBlend** (Liu et al., EuroSys '25, §4 selective
KV recompute), pre-wired to run the original YaoJiayi/CacheBlend
`example/blend_musique.py` workload unmodified.

This repo is a curated subset of the larger `cacheblend-hf-v4` development
tree. It contains only:

```
cacheblend-hf-v7/
├── src/cacheblend/                       # The reimplementation (~1.7K LOC)
│   ├── chunker.py                          chunk_texts, _stable_id, Chunk
│   ├── kv_store.py                         in-memory LRU + async prefetch
│   ├── model.py                            LayerwiseModel — HF wrap + k_proj hooks
│   ├── precompute.py                       precompute_chunk_kv / from_cache_prompt
│   ├── rope.py                             apply_rope_shift
│   ├── hkvd.py                             kv_deviation + top-K
│   ├── fusor.py                            fuse_full_recompute / full_reuse /
│   │                                       selective / prefix_cache
│   ├── runners.py                          mydata-harness-compatible runners
│   ├── controller.py                       LoadingController + StorageProfile
│   ├── tolerance.py                        Tolerance + assert_logits_close
│   └── __init__.py                         public API
├── benchmarks/musique/                   # Reproduces YaoJiayi's musique workload
│   ├── blend_musique.py                    ORIGINAL — verbatim copy, do NOT edit (Mistral-7B)
│   ├── blend_musique_generic.py            our model-agnostic version — any HF model
│   ├── utils.py                            hard copy of YaoJiayi/CacheBlend/example/utils.py
│   ├── inputs/musique_s.json               hard copy of YaoJiayi/CacheBlend/inputs/musique_s.json
│   ├── _shim/vllm/__init__.py              vllm.LLM adapter → routes to fuse_selective
│   ├── run_blend_musique.py                wrapper: sys.path + cwd + runpy
│   └── README.md
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

No `external/` clones required. No `tests/`, no `docs/`, no `scripts/` —
this is the production-ready slice for reproducing CacheBlend's musique
result.

## Quickstart

### CPU smoke (verify scaffolding, no GPU needed)

```bash
pip install torch==2.4.1
pip install -r requirements.txt
pip install -e .
CACHEBLEND_MOCK_MODEL=1 CACHEBLEND_MUSIQUE_N=2 python benchmarks/musique/run_blend_musique.py
```

## Running on an NVIDIA A100 server — step by step

Tested target hardware: 1× **NVIDIA A100 80GB** (musique fits in 40GB; 80GB
gives comfortable headroom for KV cache + long contexts). Steps below also
work on **A100 40GB**, **H100 PCIe**, **RTX 4090 24GB**, and **RTX 3090 24GB**
— for 24GB cards, `CACHEBLEND_ATTN_IMPL=sdpa` (the default) is mandatory
because eager attention OOMs at ~7K-token musique prompts.

The instructions assume the server already has:
- NVIDIA driver compatible with CUDA 12.4 (`nvidia-smi` works)
- Python 3.11 (3.10 or 3.12 also work, but 3.11 matches the locked stack)
- ~80GB free disk (Mistral-7B ≈14GB + dependencies)
- Outbound network (HuggingFace Hub + GitHub)

### 1. Connect and prepare the workspace

```bash
ssh <user>@<a100-server>
mkdir -p ~/work && cd ~/work
```

### 2. Clone this repo

```bash
git clone https://github.com/chjs/cacheblend-hf-v7.git
cd cacheblend-hf-v7
```

(If you received this repo as a tarball, expand it with
`tar -xzf cacheblend-hf-v7.tgz && cd cacheblend-hf-v7` instead.)

Sanity-check that the dataset is in place — it ships with the repo and is
**not** a symlink, so this should print `150`:

```bash
python -c "import json; print(len(json.load(open('benchmarks/musique/inputs/musique_s.json'))))"
```

### 3. Create an isolated Python environment

Pick **one** of the two options.

**Option A — `venv` (lightest, recommended):**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**Option B — `conda`:**

```bash
conda create -n cacheblend python=3.11 -y
conda activate cacheblend
```

### 4. Install PyTorch with CUDA 12.4

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

Verify torch sees the A100:

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
PY
```

Expected output (memory size will match your card):

```
torch=2.4.1+cu124 cuda=12.4 device=NVIDIA A100-SXM4-80GB
```

### 5. Install the rest of the dependencies + this package

```bash
# Skip torch in requirements.txt — we just installed it in step 4.
grep -v -E '^torch(\s|=|$)' requirements.txt > /tmp/reqs-no-torch.txt
pip install -r /tmp/reqs-no-torch.txt
pip install -e .
```

Verify `cacheblend` imports:

```bash
python -c "from cacheblend import LayerwiseModel, fuse_selective; print('cacheblend OK')"
```

### 6. Authenticate with Hugging Face (Mistral-7B-Instruct-v0.2 is gated)

You need a HuggingFace account that has accepted the Mistral-7B-Instruct-v0.2
license at <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2>. Then
create an access token at <https://huggingface.co/settings/tokens> with
`read` scope and log in:

```bash
huggingface-cli login --token <your_hf_token>
```

(Alternatively, `export HF_TOKEN=<your_hf_token>` works for the duration of
the shell.)

### 7. (Recommended) Place the HF cache on the larger volume

```bash
export HF_HOME=$PWD/.hf_home
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p $HF_HUB_CACHE
```

Without this, HF caches to `~/.cache/huggingface/` which may be on a small
root volume.

### 8. Reduce fragmentation (recommended for 24GB cards; safe on A100)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 9. Sanity check — CPU mock run (no model download, ~5 seconds)

```bash
CACHEBLEND_MOCK_MODEL=1 CACHEBLEND_MUSIQUE_N=2 python benchmarks/musique/run_blend_musique.py
```

Expected last 4 lines:

```
---------------Result Summary---------------------
TTFT with cache: <microseconds>
TTFT with full prefill: <microseconds>
F1 with cache: 0.0
F1 with full prefill: 0.0
```

F1=0 in mock mode is expected (no real generation).

### 10. Quick GPU smoke — 5 examples (~1 min after model download)

```bash
CACHEBLEND_MUSIQUE_N=5 python benchmarks/musique/run_blend_musique.py
```

The **first** run downloads Mistral-7B-Instruct-v0.2 (~14GB, 3 shards). Time
depends on HF Hub speed — typically 3–10 minutes. Subsequent runs reuse the
cache.

Expected last 4 lines (numbers vary slightly with hardware):

```
---------------Result Summary---------------------
TTFT with cache: ~0.5 s
TTFT with full prefill: ~2.0 s
F1 with cache: ~0.5
F1 with full prefill: ~0.7
```

If you see this, the pipeline works end-to-end.

### 11. Full musique sweep — 150 examples (~10 min wall time after model load)

```bash
mkdir -p logs
python benchmarks/musique/run_blend_musique.py 2>&1 | tee logs/run-n150.log
```

If you want to keep your SSH session free, run detached:

```bash
nohup python benchmarks/musique/run_blend_musique.py \
      > logs/run-n150.log 2>&1 &
echo "PID=$!"
tail -f logs/run-n150.log   # Ctrl-C just stops following; run keeps going
```

Expected final summary in `logs/run-n150.log`:

```
---------------Result Summary---------------------
TTFT with cache: 0.56 s
TTFT with full prefill: 2.27 s
F1 with cache: 0.258
F1 with full prefill: 0.276
```

That's the **4.01x TTFT speedup, 93.4% F1 retention** result matching the
paper.

### 12. Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.OutOfMemoryError: Tried to allocate 5.21 GiB` | Eager attention on 7K tokens | `export CACHEBLEND_ATTN_IMPL=sdpa` (default since v7) |
| `OSError: You are trying to access a gated repo` | Mistral license not accepted | Visit the model page on HF and accept terms |
| `ModuleNotFoundError: No module named 'cacheblend'` | Forgot `pip install -e .` | Re-run step 5 |
| Model download stalls | HF Hub network issue | Retry; consider `HF_HUB_ENABLE_HF_TRANSFER=1 pip install hf_transfer` and re-export |
| `KeyError: 'HF_TOKEN'` | Token not exported | Run `huggingface-cli login` (step 6) |
| Long tokenizer warning about chat templates | Cosmetic only | Ignore |

### 13. Tunable knobs

All read from the environment by `_shim/vllm/__init__.py` at import time.

| Variable | Default | Effect |
|---|---|---|
| `CACHEBLEND_MUSIQUE_N` | (all 150) | Run only first N examples — useful for quick checks. |
| `CACHEBLEND_RECOMP_RATIO` | `0.15` | Fraction of tokens to recompute at `check_layer`. Lower = faster, more quality drop. |
| `CACHEBLEND_CHECK_LAYER` | `1` | Layer at which to perform HKVD selection. |
| `CACHEBLEND_DTYPE` | `float16` | Model dtype. `bfloat16` works on A100/H100; `float32` is sanity-check only. |
| `CACHEBLEND_DEVICE` | auto | `cuda` (default) or `cpu` (won't fit Mistral-7B on most CPUs). |
| `CACHEBLEND_ATTN_IMPL` | `sdpa` | Use `eager` only on 40GB+ GPUs for debugging — OOMs on 24GB. |
| `CACHEBLEND_MOCK_MODEL` | `0` | If `1`: scaffolding-only mode (no model loaded). |

### 14. Multi-instance / pod considerations

On ephemeral pods (vast.ai, Lambda, etc.):

- `--disk 60` (60GB) is enough for the repo + Mistral-7B + HF cache.
- Use `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime` (or `-devel`) — that
  image already has CUDA-correct torch installed, so you can `pip install -r
  requirements.txt` after stripping the torch line and skip the wheels
  download.
- Save your `.hf_home` to a persistent volume if you spin pods up/down often
  — that's the 14GB Mistral cache.
- The full N=150 sweep on RTX 3090 takes ~43 min wall (incl. model download
  on first run). On A100 80GB, expect ~10–15 min total.

### Full GPU run (Mistral-7B, ≥16GB VRAM) — short version

For experienced users who don't need the step-by-step above, on any
vast.ai-style pod with `pytorch:2.4.1-cuda12.4` already provisioned:

```bash
grep -v -E '^torch(\s|=|$)' requirements.txt > /tmp/reqs-no-torch.txt
pip install -r /tmp/reqs-no-torch.txt
pip install -e .
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Mistral-7B via the verbatim YaoJiayi original:
python benchmarks/musique/run_blend_musique.py

# Or any other model (Llama-3.1-8B, Qwen2.5-7B, ...) via the generic workload:
CACHEBLEND_WORKLOAD=blend_musique_generic.py \
CACHEBLEND_MODEL=meta-llama/Llama-3.1-8B-Instruct \
    python benchmarks/musique/run_blend_musique.py
```

First run downloads the model (~14-16GB), ~8 min on HF Hub.

## What CacheBlend does

For multi-document RAG-style prompts (e.g. musique's 10 passages + question),
CacheBlend:

1. Pre-computes per-document KV cache standalone (positions 0..L-1).
2. At inference, runs a full forward only up to `check_layer` (e.g. layer 1).
3. At `check_layer`, selects the top-K most "diverged" tokens (HKVD —
   high-KV-deviation, paper §4.2) via `||K_fresh - K_cached||²` per position.
4. For layers ≥ `check_layer`, runs a sparse forward only over the top-K
   tokens — non-HKVD positions reuse cached KV (RoPE-shifted to fused positions).
5. Decodes from the last position normally.

The result: ~4x prefill speedup with near-zero quality loss (93-95% F1
retention vs full prefill) on multi-doc QA.

## Reproduction status

| Dataset | Model | Hardware | TTFT speedup | F1 retention | N |
|---|---|---|---|---|---|
| musique_s | Mistral-7B-Instruct-v0.2 | RTX 3090 24GB | 4.01x | 93.4% | 150 |

Matches the paper's musique claims. See `benchmarks/musique/README.md` for
details on shim semantics, known differences, and reproduction commands.

## Provenance

Derived from `cacheblend-hf-v4` (internal dev tree), commit `3f3808a` on
branch `step/musique-original-runnable` (musique experiment) + commit
`a043c23` on branch `chore/rename-cacheblend-runner` (V4 suffix removed
from `CacheBlendRunner`).

`blend_musique.py`, `utils.py`, and `musique_s.json` are unmodified copies
from <https://github.com/YaoJiayi/CacheBlend>. The rest of the code is
this project's HF reimplementation.
