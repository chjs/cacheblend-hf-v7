#!/usr/bin/env python
"""Runner for original YaoJiayi/CacheBlend `blend_musique.py` — UNMODIFIED.

This wrapper:
  1. Inserts `_shim/` onto sys.path so `import vllm` finds our adapter shim
     (which routes vLLM API calls to our HF-based cacheblend-hf-v4 impl).
  2. Inserts the script's own directory onto sys.path so `from utils import ...`
     finds the symlinked utils.py (sister file from original example/).
  3. chdir into the script's directory so the relative `inputs/musique_s.json`
     path in the original code resolves to our symlinked dataset.
  4. runpy.run_path('blend_musique.py', run_name='__main__') — runs the
     original file exactly as if it had been invoked directly.

Env vars (forwarded to the shim):
  CACHEBLEND_MOCK_MODEL=1     skip model load; .generate() returns stub text.
                              Useful for CPU smoke tests without GPU/14GB VRAM.
  CACHEBLEND_DEVICE           'cuda' or 'cpu' (default: auto)
  CACHEBLEND_DTYPE            'float16' or 'float32' (default: float16)
  CACHEBLEND_CHECK_LAYER      check_layer for fuse_selective (default 1)
  CACHEBLEND_RECOMP_RATIO     default recompute ratio (default 0.15)
  CACHEBLEND_ATTN_IMPL        attn_implementation for HF model (default 'sdpa';
                              musique prompts hit ~7K tokens — eager OOMs on
                              24GB GPUs)
  CACHEBLEND_MUSIQUE_N        if set, monkey-patch utils.load_dataset to slice
                              [:N] before blend_musique.py imports it. The
                              original file is untouched; we only intercept the
                              dataset-loading helper at import time.

Usage:
  # CPU dry-run of the scaffolding (no model load), 2 examples:
  CACHEBLEND_MOCK_MODEL=1 CACHEBLEND_MUSIQUE_N=2 python benchmarks/musique/run_blend_musique.py

  # Real run on GPU (all 150 examples):
  python benchmarks/musique/run_blend_musique.py
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    shim_root = here / "_shim"
    blend_script = here / "blend_musique.py"

    if not blend_script.exists():
        print(f"ERROR: blend_musique.py not found at {blend_script}", file=sys.stderr)
        return 2
    if not shim_root.is_dir():
        print(f"ERROR: vllm shim not found at {shim_root}", file=sys.stderr)
        return 2
    if not (here / "utils.py").exists():
        print(f"ERROR: utils.py symlink missing at {here / 'utils.py'}", file=sys.stderr)
        print("Set up with: ln -s ../../external/CacheBlend/example/utils.py utils.py", file=sys.stderr)
        return 2
    if not (here / "inputs" / "musique_s.json").exists():
        print(f"ERROR: dataset symlink missing at {here / 'inputs' / 'musique_s.json'}", file=sys.stderr)
        print("Set up with: ln -s ../../../external/CacheBlend/inputs/musique_s.json inputs/musique_s.json", file=sys.stderr)
        return 2

    # Push shim and script dir to the front of sys.path so they outrank any
    # real vllm install in the venv.
    sys.path.insert(0, str(shim_root))
    sys.path.insert(0, str(here))

    # Original script uses relative `inputs/musique_s.json` — must chdir.
    os.chdir(here)

    # Optional truncation hook — wrap utils.load_dataset before blend_musique
    # imports it. Original file untouched; we just rebind the helper.
    limit = os.environ.get('CACHEBLEND_MUSIQUE_N')
    if limit:
        n = int(limit)
        import utils as _orig_utils
        _orig_load = _orig_utils.load_dataset

        def _limited_load_dataset(path):
            data = _orig_load(path)
            print(f"[run_blend_musique] CACHEBLEND_MUSIQUE_N={n} → slicing dataset to first {n} examples", flush=True)
            return data[:n]

        _orig_utils.load_dataset = _limited_load_dataset

    print(f"[run_blend_musique] cwd: {here}", flush=True)
    print(f"[run_blend_musique] vllm shim: {shim_root}/vllm", flush=True)
    print(f"[run_blend_musique] mock model: {os.environ.get('CACHEBLEND_MOCK_MODEL', '0')}", flush=True)
    print(f"[run_blend_musique] device: {os.environ.get('CACHEBLEND_DEVICE', 'auto')}", flush=True)
    print(f"[run_blend_musique] dtype: {os.environ.get('CACHEBLEND_DTYPE', 'float16')}", flush=True)
    print(f"[run_blend_musique] check_layer: {os.environ.get('CACHEBLEND_CHECK_LAYER', '1')}", flush=True)
    print(f"[run_blend_musique] recomp_ratio: {os.environ.get('CACHEBLEND_RECOMP_RATIO', '0.15')}", flush=True)
    print(f"[run_blend_musique] running: {blend_script}", flush=True)
    print("─" * 70, flush=True)

    runpy.run_path(str(blend_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
