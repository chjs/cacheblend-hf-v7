"""Runner subclasses — wrap v4 fusor functions in the mydata harness CacheBlendRunner ABC.

Phase 0: stub classes (instantiate-only, no model required).
Phase 5: real `prepare/generate` implementations + CPU stub mode (model is None
         → returns a dummy GenerationResult; used by Phase 5 dry-run plumbing
         and tests/test_runners.py without GPU).
Phase 6+: GPU evaluation actually invokes `prepare/generate` with a real model.

The mydata harness (external/mydata/cacheblend_fig12/harness/) provides:
  - CacheBlendRunner ABC with `prepare(system, docs, question)` + `generate(max_new_tokens)`
  - FullPrefillRunner (reference baseline)
  - F1/Rouge-L metrics

We add 4 runners on top:
  - FullRecomputeRunner: standard prefill, no KV reuse (Phase 6 baseline)
  - FullReuseRunner: cache reuse without selective recompute (Phase 2)
  - PrefixCacheRunner: only first chunk reused (Phase 4)
  - CacheBlendRunner: selective recompute (paper §4, sparse forward)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


def _try_import_harness():
    harness_path = Path(__file__).resolve().parents[2] / "external" / "mydata" / "cacheblend_fig12"
    if harness_path.exists():
        if str(harness_path) not in sys.path:
            sys.path.insert(0, str(harness_path))
        try:
            from harness.runner import CacheBlendRunner, GenerationResult
            return CacheBlendRunner, GenerationResult, True
        except ImportError:
            pass
    return object, None, False


_BaseRunner, _GenerationResult, _HARNESS_AVAILABLE = _try_import_harness()


def _stub_generation():
    """CPU stub mode: dummy result for plumbing tests without a real model."""
    if _GenerationResult is not None:
        return _GenerationResult(text="", ttft_seconds=0.0, total_seconds=0.0, n_generated_tokens=0)
    # Harness not available — return a duck-typed object.
    class _Stub:
        text = ""
        ttft_seconds = 0.0
        total_seconds = 0.0
        n_generated_tokens = 0
    return _Stub()


class _RunnerBase(_BaseRunner):
    """Common base — provides CPU stub-mode plumbing.

    Subclasses must override `_run_prefill_and_generate`. The dispatch
    structure (prepare/generate) is identical for all runners; only the prefill
    fusion strategy varies.
    """

    def __init__(self, model=None, tokenizer=None):
        # L31: skip super().__init__ when model is None (harness ABC dereferences
        # next(model.parameters()).device, which crashes on None). Phase 5 stub
        # mode and Phase 0 smoke test go through this branch.
        if _HARNESS_AVAILABLE and model is not None:
            super().__init__(model=model, tokenizer=tokenizer)
        else:
            self.model = model
            self.tokenizer = tokenizer
            self.device = (
                next(model.parameters()).device if model is not None
                else torch.device("cpu")
            )
        # Per-example state (reset by prepare()).
        self._system: str | None = None
        self._docs: list[str] | None = None
        self._question: str | None = None

    def prepare(self, system: str, docs: list[str], question: str) -> None:
        """Default: store inputs; subclasses may pre-tokenize / pre-cache."""
        self._system = system
        self._docs = docs
        self._question = question

    def _build_prompt_text(self) -> str:
        doc_block = "\n\n".join(f"Document {i + 1}:\n{d}" for i, d in enumerate(self._docs or []))
        return f"{self._system}\n\n{doc_block}\n\nQuestion: {self._question}\nAnswer:"

    def generate(self, max_new_tokens: int = 32):
        # CPU stub mode: no model → no generation; harness still gets a valid result.
        if self.model is None:
            return _stub_generation()
        return self._run_prefill_and_generate(max_new_tokens=max_new_tokens)

    # Subclasses override.
    def _run_prefill_and_generate(self, max_new_tokens: int):
        raise NotImplementedError(f"{type(self).__name__}: _run_prefill_and_generate not implemented")

    # ── Shared autoregressive decode helper ─────────────────────────────────

    def _greedy_decode_from_prefill(
        self,
        prefill_logits: torch.Tensor,
        past_key_values,
        max_new_tokens: int,
        t_start: float,
        t_first_token: float | None = None,
    ):
        """Step-decode after the prefill pass produced past_key_values.

        - Picks argmax of the LAST prefill position to get token #1.
        - Then loops calling self.model(input_ids=last_token, past_key_values=...).
        """
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        # First decoded token from prefill's last position.
        next_id = prefill_logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        if t_first_token is None:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t_first_token = time.perf_counter()

        generated = [int(next_id.item())]

        for _ in range(max_new_tokens - 1):
            if eos_id is not None and generated[-1] == eos_id:
                break
            out = self.model(
                input_ids=next_id,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            next_id = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            generated.append(int(next_id.item()))

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return _GenerationResult(
            text=text,
            ttft_seconds=t_first_token - t_start,
            total_seconds=t_end - t_start,
            n_generated_tokens=len(generated),
        )

    # ── Helper: chunkify prompt parts (used by reuse-style runners) ─────────

    def _build_chunks(self):
        """Split (system, docs, question+answer-tag) into Chunks via mydata's
        prompt format. Each chunk is tokenized independently for KVStore keying.
        """
        from cacheblend.chunker import chunk_texts
        # The harness format is: system + "\n\n" + Doc1 + "\n\n" + ... + "\n\nQuestion: ...\nAnswer:"
        # We chunk at: [system], [Doc i for each i], [question+answer-tag].
        parts = []
        parts.append(self._system or "")
        for i, d in enumerate(self._docs or []):
            parts.append(f"\n\nDocument {i + 1}:\n{d}")
        parts.append(f"\n\nQuestion: {self._question}\nAnswer:")
        return chunk_texts(self.tokenizer, parts)


class FullRecomputeRunner(_RunnerBase):
    """Standard full prefill — same as harness FullPrefillRunner reference."""

    def _run_prefill_and_generate(self, max_new_tokens: int):
        prompt = self._build_prompt_text()
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.inference_mode():
            out = self.model(input_ids=enc["input_ids"], use_cache=True)
        return self._greedy_decode_from_prefill(
            prefill_logits=out.logits,
            past_key_values=out.past_key_values,
            max_new_tokens=max_new_tokens,
            t_start=t_start,
        )


class FullReuseRunner(_RunnerBase):
    """All chunks reused via cached pre-RoPE K + V (Phase 2)."""

    def __init__(self, model=None, tokenizer=None):
        super().__init__(model=model, tokenizer=tokenizer)
        self._lw_model = None
        self._kv_store = None

    def _ensure_lw_and_store(self):
        from cacheblend import LayerwiseModel
        from cacheblend.kv_store import KVStore
        if self._lw_model is None:
            # Build a thin LayerwiseModel view that shares weights with self.model
            # by wrapping the same model + tokenizer. Cheap (no reload).
            self._lw_model = LayerwiseModel.__new__(LayerwiseModel)
            self._lw_model.model = self.model
            self._lw_model.tokenizer = self.tokenizer
            self._lw_model.device = self.device
            self._lw_model.dtype = next(self.model.parameters()).dtype
            self._lw_model._inner = self.model.model
            self._lw_model.num_layers = len(self._lw_model._inner.layers)
            self._lw_model._pre_rope_k = {}
            self._lw_model._hook_handles = []
            self._lw_model._install_k_proj_hooks()
        if self._kv_store is None:
            self._kv_store = KVStore()

    def _run_prefill_and_generate(self, max_new_tokens: int):
        from cacheblend.precompute import precompute_chunk_kv
        from cacheblend.fusor import fuse_full_reuse

        self._ensure_lw_and_store()
        chunks = self._build_chunks()
        for c in chunks:
            if not self._kv_store.has(c.chunk_id):
                K, V = precompute_chunk_kv(self._lw_model, c)
                self._kv_store.put(c.chunk_id, K, V)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        prefill_logits = fuse_full_reuse(self._lw_model, chunks, self._kv_store)
        # KV cache wasn't saved through DynamicCache here; for generation, fall
        # back to a full forward to materialize past_key_values.
        from cacheblend.chunker import fused_input_ids
        input_ids = fused_input_ids(chunks, device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, use_cache=True)
        return self._greedy_decode_from_prefill(
            prefill_logits=prefill_logits,
            past_key_values=out.past_key_values,
            max_new_tokens=max_new_tokens,
            t_start=t_start,
        )


class PrefixCacheRunner(_RunnerBase):
    """Only first chunk reused (Phase 4 baseline)."""

    def __init__(self, model=None, tokenizer=None):
        super().__init__(model=model, tokenizer=tokenizer)
        self._lw_model = None
        self._kv_store = None

    def _ensure_lw_and_store(self):
        FullReuseRunner._ensure_lw_and_store(self)

    def _run_prefill_and_generate(self, max_new_tokens: int):
        from cacheblend.precompute import precompute_chunk_kv
        from cacheblend.fusor import fuse_prefix_cache
        from cacheblend.chunker import fused_input_ids

        self._ensure_lw_and_store()
        chunks = self._build_chunks()
        # Only need the first chunk cached.
        first = chunks[0]
        if not self._kv_store.has(first.chunk_id):
            K, V = precompute_chunk_kv(self._lw_model, first)
            self._kv_store.put(first.chunk_id, K, V)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        prefill_logits = fuse_prefix_cache(self._lw_model, chunks, self._kv_store)
        input_ids = fused_input_ids(chunks, device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, use_cache=True)
        return self._greedy_decode_from_prefill(
            prefill_logits=prefill_logits,
            past_key_values=out.past_key_values,
            max_new_tokens=max_new_tokens,
            t_start=t_start,
        )


class CacheBlendRunner(_RunnerBase):
    """Selective recompute (paper §4, sparse-forward implementation)."""

    def __init__(self, model=None, tokenizer=None, recompute_ratio: float = 0.15, check_layer: int = 1):
        super().__init__(model=model, tokenizer=tokenizer)
        self.recompute_ratio = recompute_ratio
        self.check_layer = check_layer
        self._lw_model = None
        self._kv_store = None

    def _ensure_lw_and_store(self):
        FullReuseRunner._ensure_lw_and_store(self)

    def _run_prefill_and_generate(self, max_new_tokens: int):
        from cacheblend.precompute import precompute_chunk_kv
        from cacheblend.fusor import fuse_selective

        self._ensure_lw_and_store()
        chunks = self._build_chunks()
        for c in chunks:
            if not self._kv_store.has(c.chunk_id):
                K, V = precompute_chunk_kv(self._lw_model, c)
                self._kv_store.put(c.chunk_id, K, V)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        # Paper §4: decode uses the selectively-recomputed past_key_values
        # directly (no second full forward). DynamicCache built inside
        # fuse_selective has full-length K/V at every layer (cached@non-HKVD +
        # fresh@HKVD), which is exactly what paper §4 specifies.
        prefill_out = fuse_selective(
            self._lw_model, chunks, self._kv_store,
            recompute_ratio=self.recompute_ratio,
            check_layer=self.check_layer,
            return_layerwise_output=True,
        )
        return self._greedy_decode_from_prefill(
            prefill_logits=prefill_out.logits,
            past_key_values=prefill_out.past_key_values,
            max_new_tokens=max_new_tokens,
            t_start=t_start,
        )


__all__ = [
    "FullRecomputeRunner",
    "FullReuseRunner",
    "PrefixCacheRunner",
    "CacheBlendRunner",
]
