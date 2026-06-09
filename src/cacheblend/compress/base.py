"""Compressed-KV blending — backend adapter + CompressedChunk + token-prune + blend glue.

Goal 1 (cacheblend + kvzip, ONLY-HKVD): an offline compression backend produces a
CompressedChunk (pre-RoPE K/V + per-token importance) per text chunk. `token_prune`
drops low-importance tokens (per-chunk budget). `to_blend_inputs` turns the pruned
chunks into the `(list[Chunk], KVStore)` consumed by the EXISTING `fuse_selective`
(pure HKVD selective recompute) — no new fuse function.

Two-stage, honest API (B안):
  * `backend.score(input_ids)` runs the EXPENSIVE importance pass once and returns a
    FULL (unpruned) CompressedChunk with `compression_rate == 0`.
  * `token_prune(chunk, budget)` is the CHEAP, reusable eviction step → a genuinely
    shorter CompressedChunk. Score-once / prune-to-many ratios without re-scoring.

Importance is used ONLY here (to decide which tokens survive). The blend itself
(`fuse_selective`) sees only deviation — importance is never carried into it.

K/V storage convention matches `cacheblend.kv_store.KVStore` exactly:
  key_cache[layer], value_cache[layer] : Tensor [1, chunk_len, num_kv_heads*head_dim]
  PRE-RoPE (k_proj / v_proj output). RoPE is applied at blend time over the fused
  (contiguous) positions, so a pruned chunk's survivors are re-compacted to
  contiguous positions — matching compblend7's scenario.
"""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from cacheblend.chunker import Chunk
from cacheblend.kv_store import KVStore


@dataclass
class CompressionBudget:
    """How many tokens a chunk keeps. Exactly one of ratio / absolute."""

    ratio: float | None = None          # fraction of tokens to KEEP, e.g. 0.30
    absolute: int | None = None         # absolute tokens to keep (overrides ratio)
    min_kept: int = 1

    def __post_init__(self) -> None:
        if self.ratio is None and self.absolute is None:
            raise ValueError("CompressionBudget requires one of: ratio, absolute")
        if self.ratio is not None and not (0.0 < self.ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {self.ratio}")
        if self.absolute is not None and self.absolute < 1:
            raise ValueError(f"absolute must be >= 1, got {self.absolute}")

    def keep_k(self, n: int) -> int:
        """Resolve the number of tokens to keep for a chunk of length n."""
        k = self.absolute if self.absolute is not None else int(round(n * self.ratio))
        return max(self.min_kept, min(n, k))


@dataclass
class CompressedChunk:
    """One text chunk's compressed KV cache (pre-RoPE K/V + per-token importance).

    Carries the FULL chunk after `score()` (compression_rate == 0); becomes a
    genuinely shorter chunk after `token_prune()`. `token_ids` are exactly the
    surviving tokens; positions are implicit [0, chunk_len) and re-RoPE'd to the
    fused position at blend time.
    """

    # identity
    algo_id: str
    chunk_id: str
    model_id: str
    tokenizer_id: str
    # geometry — chunk_len == len(token_ids)
    token_ids: list[int]
    # pre-RoPE K/V, one tensor per layer: [1, chunk_len, num_kv_heads*head_dim]
    key_cache: list[torch.Tensor]
    value_cache: list[torch.Tensor]
    # per-(layer, head, position) importance, fp32: [num_layers, H_kv, chunk_len]
    importance: torch.Tensor
    # diagnostics (no semantic effect on the blend)
    compression_rate: float = 0.0       # fraction of tokens DROPPED (0.0 = scored-full)
    backend_state: dict[str, Any] = field(default_factory=dict)

    SERIALIZATION_VERSION: int = 1

    @property
    def chunk_len(self) -> int:
        return len(self.token_ids)

    @property
    def num_layers(self) -> int:
        return len(self.key_cache)

    # ── disk serialization (offline RAG: compress corpus once, load per query) ──
    def save(self, path: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": self.SERIALIZATION_VERSION,
            "algo_id": self.algo_id, "chunk_id": self.chunk_id,
            "model_id": self.model_id, "tokenizer_id": self.tokenizer_id,
            "token_ids": list(self.token_ids),
            "key_cache": [k.detach().cpu().contiguous() for k in self.key_cache],
            "value_cache": [v.detach().cpu().contiguous() for v in self.value_cache],
            "importance": self.importance.detach().cpu().contiguous(),
            "compression_rate": float(self.compression_rate),
            "backend_state": dict(self.backend_state),
        }, p)

    @classmethod
    def load(cls, path: Any, map_location: Any = "cpu") -> "CompressedChunk":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        got = payload.get("version")
        if got != cls.SERIALIZATION_VERSION:
            raise ValueError(
                f"CompressedChunk version mismatch: file={got}, code={cls.SERIALIZATION_VERSION}"
            )
        return cls(
            algo_id=payload["algo_id"], chunk_id=payload["chunk_id"],
            model_id=payload["model_id"], tokenizer_id=payload["tokenizer_id"],
            token_ids=list(payload["token_ids"]),
            key_cache=list(payload["key_cache"]), value_cache=list(payload["value_cache"]),
            importance=payload["importance"],
            compression_rate=float(payload["compression_rate"]),
            backend_state=dict(payload["backend_state"]),
        )

    def to(self, device: Any, dtype: Any = None) -> "CompressedChunk":
        def mv(t: torch.Tensor, dt: Any = None) -> torch.Tensor:
            if dt is not None and t.is_floating_point():
                return t.to(device=device, dtype=dt).contiguous()
            return t.to(device=device).contiguous()
        return dataclasses.replace(
            self,
            key_cache=[mv(k, dtype) for k in self.key_cache],
            value_cache=[mv(v, dtype) for v in self.value_cache],
            importance=mv(self.importance, torch.float32),
        )


# ──────────────────────────────────────────────────────────────────────────────
# token pruning (the actual compression step) — per-chunk importance budget
# ──────────────────────────────────────────────────────────────────────────────


def _subset(chunk: CompressedChunk, keep: torch.Tensor) -> CompressedChunk:
    """Restrict a chunk to sorted token indices `keep` (head-uniform token-prune)."""
    keep = torch.as_tensor(keep, dtype=torch.long)
    kl = keep.tolist()
    nl = chunk.num_layers
    return dataclasses.replace(
        chunk,
        chunk_id=f"{chunk.chunk_id}:keep{len(kl)}",
        token_ids=[chunk.token_ids[i] for i in kl],
        key_cache=[chunk.key_cache[li][:, keep, :].contiguous() for li in range(nl)],
        value_cache=[chunk.value_cache[li][:, keep, :].contiguous() for li in range(nl)],
        importance=chunk.importance[:, :, keep].contiguous(),
        compression_rate=1.0 - (len(kl) / max(1, chunk.chunk_len)),
    )


def token_prune(
    chunk: CompressedChunk,
    budget: CompressionBudget,
    *,
    reduce: str = "mean",
) -> CompressedChunk:
    """Per-chunk budget: keep the top-k tokens of THIS chunk by importance.

    reduce: how the [L, H_kv, T] importance is reduced to a per-token score.
      "mean" (default, compblend7) = mean over all layers AND heads.
      "max"  (KVzip-paper-faithful) = max over all layers AND heads
             ("critical in ANY head/layer").
    Returns a genuinely shorter CompressedChunk (survivors only).
    """
    n = chunk.chunk_len
    k = budget.keep_k(n)
    imp = chunk.importance.float()
    if reduce == "max":
        imp_tok = imp.amax(dim=(0, 1))
    elif reduce == "mean":
        imp_tok = imp.mean(dim=(0, 1))
    else:
        raise ValueError(f"reduce must be 'mean' or 'max', got {reduce!r}")
    keep = torch.sort(torch.topk(imp_tok, k).indices).values
    return _subset(chunk, keep)


# ──────────────────────────────────────────────────────────────────────────────
# blend glue — compressed chunks → (Chunk list, KVStore) for fuse_selective
# ──────────────────────────────────────────────────────────────────────────────


def to_blend_inputs(
    chunks: list[CompressedChunk],
    kv_store: KVStore | None = None,
) -> tuple[list[Chunk], KVStore]:
    """Convert (possibly pruned) CompressedChunks into fuse_selective inputs.

    Returns `(blend_chunks, kv_store)` where blend_chunks preserve input order
    and each chunk's pre-RoPE K/V is loaded into the store under its chunk_id.
    `importance` is intentionally NOT carried — the only-HKVD blend uses
    deviation alone. Pass an existing `kv_store` to append (e.g. the caller adds
    the fresh query chunk afterwards, keeping it LAST for force_last_chunk).
    """
    if kv_store is None:
        kv_store = KVStore()
    blend_chunks: list[Chunk] = []
    for c in chunks:
        blend_chunks.append(Chunk(text="", token_ids=list(c.token_ids), chunk_id=c.chunk_id))
        kv_store.put(c.chunk_id, c.key_cache, c.value_cache)
    return blend_chunks, kv_store


# ──────────────────────────────────────────────────────────────────────────────
# backend protocol
# ──────────────────────────────────────────────────────────────────────────────


class CompressionBackend(ABC):
    """Adapter for an offline compression algorithm.

    `score()` runs the expensive importance pass and returns a FULL CompressedChunk
    (compression_rate == 0). Pruning to a budget is done separately by
    `token_prune()`, so the same scored chunk serves many compression ratios.
    """

    algo_id: str = "abstract"
    model_id: str = ""

    @abstractmethod
    def score(self, input_ids: torch.Tensor, model: Any = None) -> CompressedChunk:
        ...
