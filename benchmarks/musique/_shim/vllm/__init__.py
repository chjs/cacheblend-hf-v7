"""vllm shim — routes original YaoJiayi/CacheBlend `blend_musique.py` calls
to our HF-based cacheblend-hf-v4 implementation.

The original script does (in this order, per example):

    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.5)
    llm.set_tokenizer(tokenizer)
    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    cache_fuse_metadata['collect'] = True;  cache_fuse_metadata['check'] = False
    for each chunk_text:
        llm.generate([chunk_text], SamplingParams(temperature=0, max_tokens=1))
        # then reads layers[j].self_attn.hack_kv (per-layer K/V capture)
    # builds chunk_past_key_values and assigns to ...model.model.old_kvs

    cache_fuse_metadata['check'] = True;  cache_fuse_metadata['collect'] = False
    out = llm.generate([input_prompt], SamplingParams(temperature=0, max_tokens=32))
    # reads out[0].outputs[0].text, out[0].metrics.first_token_time/first_scheduled_time

    cache_fuse_metadata['check'] = False;  cache_fuse_metadata['collect'] = False
    out = llm.generate([input_prompt], SamplingParams(...))   # full prefill baseline

The shim:
  - collect=True: precompute pre-RoPE K + V per chunk via our LayerwiseModel and
    store into an internal KVStore. hack_kv is filled with a zero placeholder
    just large enough that user-code slicing does not crash.
  - check=True:  call our fuse_selective on the tracked chunks + KVStore, then
    greedy-decode max_tokens.
  - default:     full forward via HF + greedy decode.

old_kvs assignment from user code is recorded but unused — real KV cache lives
in the shim's internal KVStore.

Env vars:
  CACHEBLEND_MOCK_MODEL=1    skip model load; .generate() returns stub text.
                             Useful for CPU smoke tests of the scaffolding.
  CACHEBLEND_RECOMP_RATIO    default recompute ratio for check mode if user
                             code doesn't set cache_fuse_metadata['recomp_ratio']
                             (musique default 0.15).
  CACHEBLEND_DEVICE          'cuda' or 'cpu' (default: auto)
  CACHEBLEND_DTYPE           'float16' or 'float32' (default: float16)
  CACHEBLEND_CHECK_LAYER     check_layer for fuse_selective (default 1)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Public vLLM-compatible types
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 16
    # Accept and ignore other vllm sampling kwargs.
    top_p: float = 1.0
    top_k: int = -1


@dataclass
class _Metrics:
    first_scheduled_time: float
    first_token_time: float


@dataclass
class _Completion:
    text: str


@dataclass
class _RequestOutput:
    outputs: list[_Completion]
    metrics: _Metrics


# ──────────────────────────────────────────────────────────────────────────────
# Fake attribute chain used by user code: llm.llm_engine.model_executor.driver_worker.model_runner.model.model
# ──────────────────────────────────────────────────────────────────────────────


class _FakeSelfAttn:
    """Placeholder for `layer.self_attn` — exposes `hack_kv` for user code."""
    def __init__(self):
        self.hack_kv: list[torch.Tensor] | None = None


class _FakeLayer:
    def __init__(self):
        self.self_attn = _FakeSelfAttn()


class _FakeInnerModel:
    """The deeply-nested object user code reaches via the long attribute chain.

    Fields the original script touches:
      - cache_fuse_metadata: dict of mode flags (collect, check, recomp_ratio, ...)
      - layers: list of layers, each with `.self_attn.hack_kv`
      - old_kvs: assigned by user code; recorded but not consumed by shim.
    """
    def __init__(self, num_layers: int):
        self.cache_fuse_metadata: dict[str, Any] = {
            'collect': False,
            'check': False,
            'attn_bias': None,
            'recomp_ratio': float(os.environ.get('CACHEBLEND_RECOMP_RATIO', '0.15')),
            'fast_attention': True,
            'suffix_len': 0,
        }
        self.layers = [_FakeLayer() for _ in range(num_layers)]
        self.old_kvs: Any = None


class _FakeModel:
    def __init__(self, num_layers: int):
        self.model = _FakeInnerModel(num_layers)


class _FakeModelRunner:
    def __init__(self, num_layers: int):
        self.model = _FakeModel(num_layers)


class _FakeDriverWorker:
    def __init__(self, num_layers: int):
        self.model_runner = _FakeModelRunner(num_layers)


class _FakeModelExecutor:
    def __init__(self, num_layers: int):
        self.driver_worker = _FakeDriverWorker(num_layers)


class _FakeLLMEngine:
    def __init__(self, num_layers: int):
        self.model_executor = _FakeModelExecutor(num_layers)


# ──────────────────────────────────────────────────────────────────────────────
# Main shim: LLM
# ──────────────────────────────────────────────────────────────────────────────


_MOCK = os.environ.get('CACHEBLEND_MOCK_MODEL') == '1'
_DEVICE = os.environ.get('CACHEBLEND_DEVICE') or (
    'cuda' if torch.cuda.is_available() else 'cpu'
)
_DTYPE = os.environ.get('CACHEBLEND_DTYPE', 'float16')
_CHECK_LAYER = int(os.environ.get('CACHEBLEND_CHECK_LAYER', '1'))
# eager attention materializes full (heads, S, S) score tensor — OOMs at
# S >= 6000 on 24GB GPUs. SDPA dispatches to memory-efficient kernels.
# musique prompts run ~6-7K tokens (10 docs × ~600 + prefix/query), so default
# to SDPA. Override via CACHEBLEND_ATTN_IMPL=eager only for short-context tests.
_ATTN_IMPL = os.environ.get('CACHEBLEND_ATTN_IMPL', 'sdpa')

# Mistral-7B has 32 layers — used when mock model is enabled (no real model loaded).
_MOCK_NUM_LAYERS = 32
_MOCK_HIDDEN_KV = 1024   # 8 KV heads × 128 head_dim


class LLM:
    """Drop-in replacement for vllm.LLM, routing to our HF CacheBlend impl.

    User code creates one LLM at process startup and reuses it across many
    examples — so per-example state (chunks collected, KVStore) is reset
    automatically when a fresh collect cycle begins.
    """

    def __init__(self, model: str, gpu_memory_utilization: float = 0.5, **kwargs):
        self._model_name = model
        self._mock = _MOCK

        if self._mock:
            self._lw = None
            self._tokenizer = None
            num_layers = _MOCK_NUM_LAYERS
            self._hidden_kv = _MOCK_HIDDEN_KV
            self._dtype = torch.float32
            self._device = torch.device('cpu')
        else:
            from cacheblend import LayerwiseModel
            self._lw = LayerwiseModel(model, dtype=_DTYPE, attn_implementation=_ATTN_IMPL)
            self._tokenizer = self._lw.tokenizer
            num_layers = self._lw.num_layers
            attn0 = self._lw._inner.layers[0].self_attn
            self._hidden_kv = attn0.config.num_key_value_heads * attn0.head_dim
            self._dtype = self._lw.dtype
            self._device = self._lw.device

        # Build the deep attribute chain that user code traverses.
        self.llm_engine = _FakeLLMEngine(num_layers)
        self._inner = self.llm_engine.model_executor.driver_worker.model_runner.model.model

        # Per-example state.
        from cacheblend.kv_store import KVStore
        self._tracked_chunks: list = []
        self._kv_store = KVStore()
        self._last_mode: str | None = None  # 'collect' | 'check' | 'normal'
        self._KVStore = KVStore  # save class for reset

    # ── public vLLM-compatible API ─────────────────────────────────────────

    def set_tokenizer(self, tokenizer):
        """User code passes a HF tokenizer; in mock mode we adopt it."""
        if self._mock:
            self._tokenizer = tokenizer

    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[_RequestOutput]:
        assert len(prompts) == 1, "shim supports single-prompt calls only (matches musique workload)"
        prompt = prompts[0]
        meta = self._inner.cache_fuse_metadata

        if bool(meta.get('collect')):
            return self._do_collect(prompt, sampling_params)
        if bool(meta.get('check')):
            return self._do_check(prompt, sampling_params, meta)
        return self._do_normal(prompt, sampling_params)

    # ── internal: per-mode dispatch ────────────────────────────────────────

    def _maybe_reset_for_new_example(self, target_mode: str):
        """If we just finished an example (last_mode in {check, normal}) and now
        entering collect, reset per-example state."""
        if target_mode == 'collect' and self._last_mode in (None, 'check', 'normal') and self._last_mode is not None:
            self._tracked_chunks = []
            self._kv_store = self._KVStore()
        # Also reset on the very first collect after init (last_mode=None means just init)
        if target_mode == 'collect' and self._last_mode is None and self._tracked_chunks:
            self._tracked_chunks = []
            self._kv_store = self._KVStore()

    def _do_collect(self, prompt: str, sampling_params: SamplingParams) -> list[_RequestOutput]:
        self._maybe_reset_for_new_example('collect')
        t_start = time.perf_counter()

        if self._mock:
            # Mock mode: no real K/V compute. Fake hack_kv with a tiny tensor.
            L = max(2, len(prompt) // 4)  # rough placeholder length
            self._populate_hack_kv(L)
            self._tracked_chunks.append({'prompt': prompt, 'L': L})
        else:
            self._collect_real(prompt)

        self._last_mode = 'collect'
        t_first = time.perf_counter()
        return [self._make_output('', t_start, t_first)]

    def _collect_real(self, prompt: str):
        """Re-encode prompt (BOS prepended), precompute pre-RoPE K + V, populate
        KVStore and hack_kv.

        Chunk-boundary parity with original:
          - chunk 0 (first collect call in this example): use FULL re-encoded
            token_ids including BOS — chunk 0 is the prefix [INST]+prompt at
            global position 0 in the fused sequence.
          - chunks i>0: strip BOS from re-encoded token_ids — these chunks
            appear later in the fused sequence (not at position 0), and the
            original code's slicing `[s_start_1_len : len(doc_chunk_ids[i])+1]`
            (=`[1 : L+1]`) drops BOS from captured K/V.
        """
        from cacheblend.chunker import Chunk, _stable_id
        from cacheblend.precompute import precompute_chunk_kv

        token_ids = self._tokenizer.encode(prompt)  # includes BOS
        L = len(token_ids)
        is_first = len(self._tracked_chunks) == 0

        if is_first:
            chunk_ids = token_ids
        else:
            chunk_ids = token_ids[1:]  # drop BOS for non-first chunks

        # Stable id for KVStore keying.
        chunk_text = self._tokenizer.decode(chunk_ids, skip_special_tokens=False)
        chunk = Chunk(text=chunk_text, token_ids=list(chunk_ids), chunk_id=_stable_id(chunk_text, list(chunk_ids)))

        K, V = precompute_chunk_kv(self._lw, chunk)
        self._kv_store.put(chunk.chunk_id, K, V)
        self._tracked_chunks.append(chunk)

        # hack_kv shape MUST allow user-code slicing without crash. User does
        # `past_key_values[0][:s_start_len].clone()` or
        # `past_key_values[0][s_start_1_len : len(doc_chunk_ids[i])+1].clone()`.
        # Provide a (L, hidden_kv) zero tensor per layer (positions = original
        # re-encoded length including BOS).
        self._populate_hack_kv(L)

    def _populate_hack_kv(self, L: int):
        for j in range(len(self._inner.layers)):
            K = torch.zeros((L, self._hidden_kv), dtype=self._dtype, device=self._device)
            V = torch.zeros((L, self._hidden_kv), dtype=self._dtype, device=self._device)
            self._inner.layers[j].self_attn.hack_kv = [K, V]

    def _do_check(self, prompt: str, sampling_params: SamplingParams, meta: dict) -> list[_RequestOutput]:
        t_start = time.perf_counter()

        if self._mock:
            text = f"[mock check {len(self._tracked_chunks)} chunks]"
            t_first = time.perf_counter()
            self._last_mode = 'check'
            return [self._make_output(text, t_start, t_first)]

        from cacheblend.fusor import fuse_selective
        ratio = float(meta.get('recomp_ratio', 0.15))
        check_layer = _CHECK_LAYER

        if self._device.type == 'cuda':
            torch.cuda.synchronize()
        t_prefill_start = time.perf_counter()
        prefill_out = fuse_selective(
            self._lw, self._tracked_chunks, self._kv_store,
            recompute_ratio=ratio, check_layer=check_layer,
            return_layerwise_output=True,
        )
        text, t_first = self._greedy_decode(
            prefill_out.logits, prefill_out.past_key_values,
            sampling_params.max_tokens, t_prefill_start,
        )

        self._last_mode = 'check'
        return [self._make_output(text, t_start, t_first)]

    def _do_normal(self, prompt: str, sampling_params: SamplingParams) -> list[_RequestOutput]:
        t_start = time.perf_counter()

        if self._mock:
            text = "[mock normal generation]"
            t_first = time.perf_counter()
            self._last_mode = 'normal'
            return [self._make_output(text, t_start, t_first)]

        ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(self._device)
        if self._device.type == 'cuda':
            torch.cuda.synchronize()
        t_prefill_start = time.perf_counter()
        with torch.inference_mode():
            out = self._lw.model(input_ids=ids, use_cache=True)
        text, t_first = self._greedy_decode(
            out.logits, out.past_key_values,
            sampling_params.max_tokens, t_prefill_start,
        )

        self._last_mode = 'normal'
        return [self._make_output(text, t_start, t_first)]

    # ── helpers ────────────────────────────────────────────────────────────

    def _greedy_decode(self, prefill_logits, past_key_values, max_new_tokens: int,
                       t_prefill_start: float):
        """Greedy decode, returns (text, t_first_token_perf_counter)."""
        eos = getattr(self._tokenizer, 'eos_token_id', None)
        next_id = prefill_logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        if self._device.type == 'cuda':
            torch.cuda.synchronize()
        t_first = time.perf_counter()
        generated = [int(next_id.item())]
        with torch.inference_mode():
            for _ in range(max_new_tokens - 1):
                if eos is not None and generated[-1] == eos:
                    break
                out = self._lw.model(input_ids=next_id, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                generated.append(int(next_id.item()))
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return text, t_first

    def _make_output(self, text: str, t_start: float, t_first: float) -> _RequestOutput:
        return _RequestOutput(
            outputs=[_Completion(text=text)],
            metrics=_Metrics(first_scheduled_time=t_start, first_token_time=t_first),
        )


__all__ = ['LLM', 'SamplingParams']
