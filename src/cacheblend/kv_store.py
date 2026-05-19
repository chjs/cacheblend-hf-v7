"""KVStore — chunk-level pre-RoPE K + V cache with LRU eviction + async prefetch.

Phase 2: sync put/get/has/evict_lru.
Phase 4: + prefetch_chunk(chunk_id, loader_fn) submits to ThreadPoolExecutor.
         get(chunk_id) blocks on outstanding Future if needed.

Storage shape per layer per chunk:
    K_pre_rope[chunk_id][layer_idx]  — (1, chunk_seq_len, num_kv_heads * head_dim)
    V[chunk_id][layer_idx]           — (1, chunk_seq_len, num_kv_heads * head_dim)

Key is `chunk_id`. Eviction is LRU at chunk granularity.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional

import torch


class KVStore:
    """Chunk-level KV cache (pre-RoPE K, V), OrderedDict-backed LRU + optional prefetch.

    capacity   — max number of chunks (per-layer storage is implicit).
    max_workers — size of the thread pool used by prefetch_chunk; lazily started.
    """

    def __init__(self, capacity: int = 1024, max_workers: int = 4):
        self.capacity = capacity
        self.max_workers = max_workers
        # chunk_id → {"K": list[Tensor], "V": list[Tensor]} (one entry per layer)
        self._cache: OrderedDict[str, dict[str, list[torch.Tensor]]] = OrderedDict()
        # chunk_id → Future returning {"K": ..., "V": ...} (Phase 4 prefetch)
        self._inflight: dict[str, Future] = {}
        self._executor: Optional[ThreadPoolExecutor] = None  # lazy

    # ── Phase 2 sync API ─────────────────────────────────────────────────────

    def has(self, chunk_id: str) -> bool:
        # Either fully cached or inflight counts as "has" (caller can get()).
        return chunk_id in self._cache or chunk_id in self._inflight

    def get(self, chunk_id: str) -> dict[str, list[torch.Tensor]]:
        # If a prefetch is inflight, block on it before reading from _cache.
        if chunk_id in self._inflight:
            fut = self._inflight.pop(chunk_id)
            entry = fut.result()  # blocks; loader_fn must return {"K": ..., "V": ...}
            self._put_entry(chunk_id, entry)
        if chunk_id not in self._cache:
            raise KeyError(f"KVStore miss: {chunk_id}")
        self._cache.move_to_end(chunk_id)
        return self._cache[chunk_id]

    def put(
        self,
        chunk_id: str,
        K_per_layer: list[torch.Tensor],
        V_per_layer: list[torch.Tensor],
    ) -> None:
        """Store one chunk's per-layer pre-RoPE K + V."""
        self._put_entry(chunk_id, {"K": K_per_layer, "V": V_per_layer})

    def _put_entry(self, chunk_id: str, entry: dict[str, list[torch.Tensor]]) -> None:
        if chunk_id in self._cache:
            self._cache.move_to_end(chunk_id)
            self._cache[chunk_id] = entry
            return
        if len(self._cache) >= self.capacity:
            self.evict_lru()
        self._cache[chunk_id] = entry

    def evict_lru(self) -> Optional[str]:
        if not self._cache:
            return None
        evicted_id, _ = self._cache.popitem(last=False)
        return evicted_id

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        # Cancel still-pending prefetches; running ones we can't cancel safely.
        for fut in self._inflight.values():
            fut.cancel()
        self._inflight.clear()

    # ── Phase 4 async API ────────────────────────────────────────────────────

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="kvstore-prefetch",
            )
        return self._executor

    def prefetch_chunk(
        self,
        chunk_id: str,
        loader_fn: Callable[[], dict[str, list[torch.Tensor]]],
    ) -> Future:
        """Schedule an async load of `chunk_id`. Idempotent.

        loader_fn must return {"K": list[Tensor], "V": list[Tensor]}.

        - If already cached, returns a completed Future with the cached entry.
        - If already inflight, returns the existing Future.
        - Otherwise submits loader_fn to the thread pool.
        """
        if chunk_id in self._cache:
            f: Future = Future()
            f.set_result(self._cache[chunk_id])
            return f
        if chunk_id in self._inflight:
            return self._inflight[chunk_id]
        executor = self._ensure_executor()
        fut = executor.submit(loader_fn)
        self._inflight[chunk_id] = fut
        return fut

    def shutdown(self, wait: bool = True) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
