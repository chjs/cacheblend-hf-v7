"""KVzip backend — score() captures pre-RoPE K/V + per-token importance.

Heavy deps (flash_attn + the KVzip repo on PYTHONPATH) are lazy-imported, so the
dep-light blend path (cacheblend.compress.base) never pulls them in.

score() runs KVzip's `ModelKVzip.prefill(do_score=True)` (the "repeat the previous
context" reconstruction scoring) ONCE and returns a FULL CompressedChunk
(compression_rate == 0). Token-pruning is a separate cheap step (base.token_prune),
so the same scored chunk serves many compression ratios (B안 / score-once).

Pre-RoPE K is captured via a forward-hook on each layer's k_proj (its output is
pre-RoPE by construction — RoPE is applied later inside attention). At blend time
the fusor re-applies RoPE at the fused position.

Isolated per-chunk scoring: COMPBLEND_KVZIP_NO_SYS_PROMPT=1 (default) clears KVzip's
system prompt so the chunk is prefilled ALONE (sink=0). This is the reusability
choice (see docs + attention-sink analysis); the doc-beginning sink artifact it
creates is handled at blend time (force_chunk_starts / ranknorm_max prune).
Ported from compblend7 backends/kvzip.py (variant C dropped).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import torch

from cacheblend.compress.base import CompressedChunk, CompressionBackend


@dataclass
class KVzipConfig:
    kv_type: str = "retain"          # "retain" keeps key_cache as a regular 4D tensor
    chunk_id_prefix: str = "kvzip"


def kvzip_chunk_id(
    token_ids, prefix: str = "kvzip", *,
    model_id: str | None = None, tokenizer_id: str | None = None,
    dtype: str | None = None, n_layers: int | None = None,
    n_kv_heads: int | None = None, head_dim: int | None = None,
    algo_version: str = "v1",
) -> str:
    """Deterministic chunk_id from token_ids + model identity (cross-process stable).

    Model/tokenizer/dtype/shape are in the key so a corpus compressed under one
    model can't collide with another's cache. NOTE: ratio is intentionally NOT in
    the key — score() produces the FULL (unpruned) chunk; token_prune appends its
    own ":keep{N}" suffix.
    """
    parts = [",".join(str(t) for t in token_ids), f"algo={algo_version}"]
    if model_id is not None:     parts.append(f"model={model_id}")
    if tokenizer_id is not None: parts.append(f"tok={tokenizer_id}")
    if dtype is not None:        parts.append(f"dtype={dtype}")
    if n_layers is not None:     parts.append(f"L={n_layers}")
    if n_kv_heads is not None:   parts.append(f"Hkv={n_kv_heads}")
    if head_dim is not None:     parts.append(f"D={head_dim}")
    return f"{prefix}:{hashlib.sha256('|'.join(parts).encode()).hexdigest()[:16]}"


class KVzipBackend(CompressionBackend):
    """KVzip → CompressedChunk adapter (forward-hook pre-RoPE K capture).

    Not re-entrant: score() installs hooks on the shared model and removes them in
    a finally — do not call from multiple threads on one instance.
    """

    algo_id = "kvzip"

    def __init__(self, model_id: str, kvzip_config: KVzipConfig | None = None) -> None:
        self.model_id = model_id
        self.kvzip_config = kvzip_config or KVzipConfig()
        self._mk: Any = None

    # ── shared model handles (blend reuses these exact weights) ──────────────
    @property
    def hf_model(self) -> Any:
        return self._get_model_kvzip().model

    @property
    def tokenizer(self) -> Any:
        return self._get_model_kvzip().tokenizer

    def _get_model_kvzip(self) -> Any:
        if self._mk is None:
            try:
                from model import ModelKVzip  # KVzip repo's `model` package
            except ImportError as e:  # pragma: no cover — GPU pod has it
                raise ImportError(
                    "KVzipBackend needs the KVzip repo on PYTHONPATH "
                    "(git clone https://github.com/snu-mllab/KVzip; "
                    "export PYTHONPATH=$PYTHONPATH:/path/to/KVzip)."
                ) from e
            self._mk = ModelKVzip(self.model_id, kv_type=self.kvzip_config.kv_type)
        return self._mk

    # ── public API ───────────────────────────────────────────────────────────
    def score(self, input_ids: torch.Tensor, model: Any = None) -> CompressedChunk:
        """Run KVzip scoring over one chunk → FULL CompressedChunk (rate=0)."""
        del model  # KVzip carries its own ModelKVzip wrapper.
        mk = self._get_model_kvzip()

        pre_k_all: dict[int, list[torch.Tensor]] = {}
        pre_v_all: dict[int, list[torch.Tensor]] = {}
        handles = self._install_hooks(mk, pre_k_all, pre_v_all)

        # Isolated per-chunk prefill (sink=0): clear KVzip's sys prompt so the
        # captured K/V are not tied to KVzip's "assistant" context (which would
        # mismatch the CacheBlend fused prefix). See module docstring.
        no_sys = os.environ.get("COMPBLEND_KVZIP_NO_SYS_PROMPT", "1") == "1"
        original_sys = None
        if no_sys:
            original_sys = mk.sys_prompt_ids
            mk.sys_prompt_ids = torch.zeros((1, 0), dtype=original_sys.dtype,
                                            device=original_sys.device)
        try:
            ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
            ids = ids.to(device=mk.device, dtype=torch.long)
            kv = mk.prefill(ids, load_score=False, do_score=True)
        finally:
            for h in handles:
                h.remove()
            if original_sys is not None:
                mk.sys_prompt_ids = original_sys

        sink_ctx = int(kv.sink) + int(kv.ctx_len)
        pre_k = self._select_prefill_capture(pre_k_all, sink_ctx, "k_proj")
        pre_v = self._select_prefill_capture(pre_v_all, sink_ctx, "v_proj")
        return self._build_chunk(mk, kv, pre_k, pre_v, input_ids)

    # ── internals ──────────────────────────────────────────────────────────
    @staticmethod
    def _install_hooks(mk, pre_k_all, pre_v_all) -> list:
        """Hook k_proj/v_proj on every layer, APPENDING each fire (prefill +
        scoring task both fire; we pick the prefill capture by shape later)."""
        base = getattr(mk.model, "model", mk.model)
        handles = []
        for idx, layer in enumerate(base.layers):
            attn = layer.self_attn

            def mk_k(i):
                def hook(_m, _i, out): pre_k_all.setdefault(i, []).append(out.detach())
                return hook

            def mk_v(i):
                def hook(_m, _i, out): pre_v_all.setdefault(i, []).append(out.detach())
                return hook

            handles.append(attn.k_proj.register_forward_hook(mk_k(idx)))
            handles.append(attn.v_proj.register_forward_hook(mk_v(idx)))
        return handles

    @staticmethod
    def _select_prefill_capture(captures_all, sink_ctx, name) -> dict[int, torch.Tensor]:
        """Pick the prefill K/V per layer (shape[1] == sink+ctx_len), handling
        single and chunked (long-context) prefill."""
        out: dict[int, torch.Tensor] = {}
        for li, caps in captures_all.items():
            single = [t for t in caps if t.shape[1] == sink_ctx]
            if single:
                out[li] = single[0]
                continue
            running, chunks = 0, []
            for t in caps:
                chunks.append(t); running += int(t.shape[1])
                if running == sink_ctx:
                    break
                if running > sink_ctx:
                    chunks = []; break
            if running == sink_ctx and chunks:
                out[li] = torch.cat(chunks, dim=1).contiguous()
                continue
            raise RuntimeError(
                f"could not reconstruct {name} prefill at layer {li}: "
                f"target shape[1]={sink_ctx}, captured {[tuple(t.shape) for t in caps]}"
            )
        return out

    def _build_chunk(self, mk, kv, pre_k, pre_v, input_ids) -> CompressedChunk:
        n_layers, n_kv = kv.n_layers, kv.n_heads_kv
        sink, ctx_len = int(kv.sink), int(kv.ctx_len)
        missing = [li for li in range(n_layers) if li not in pre_k or li not in pre_v]
        if missing:
            raise RuntimeError(f"hooks did not fire for layers {missing}")

        key_ctx = [pre_k[li][:, sink:sink + ctx_len, :].contiguous() for li in range(n_layers)]
        val_ctx = [pre_v[li][:, sink:sink + ctx_len, :].contiguous() for li in range(n_layers)]
        importance = torch.stack(
            [kv.score[li].squeeze(0) for li in range(n_layers)], dim=0
        ).to(torch.float32)                                   # [L, H_kv, ctx_len]
        if importance.shape != (n_layers, n_kv, ctx_len):
            raise RuntimeError(
                f"importance shape {tuple(importance.shape)} != (L={n_layers},H={n_kv},T={ctx_len})"
            )

        token_ids = (input_ids.squeeze(0) if input_ids.dim() > 1 else input_ids).detach().cpu().tolist()
        tokenizer_id = getattr(mk.tokenizer, "name_or_path", None) or self.model_id
        head_dim = key_ctx[0].shape[-1] // n_kv
        chunk_id = kvzip_chunk_id(
            token_ids, prefix=self.kvzip_config.chunk_id_prefix, model_id=self.model_id,
            tokenizer_id=tokenizer_id, dtype=str(key_ctx[0].dtype),
            n_layers=n_layers, n_kv_heads=n_kv, head_dim=head_dim,
        )
        return CompressedChunk(
            algo_id=self.algo_id, chunk_id=chunk_id, model_id=self.model_id,
            tokenizer_id=tokenizer_id, token_ids=token_ids,
            key_cache=key_ctx, value_cache=val_ctx, importance=importance,
            compression_rate=0.0,
            backend_state={"kvzip_kv_type": self.kvzip_config.kv_type,
                           "sys_prompt_len": sink, "ctx_len": ctx_len},
        )
