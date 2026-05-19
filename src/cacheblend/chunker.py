"""Chunk dataclass + chunking utilities (Phase 2).

A `Chunk` is the unit of KV reuse. Each chunk is a contiguous run of token IDs
plus a stable `chunk_id` that the KVStore uses as its key.

Phase 2 scope:
- `Chunk` dataclass
- `chunk_texts(tokenizer, texts)` — tokenize a list of strings into Chunks
- `fused_input_ids(chunks)` — concatenate chunk token_ids into a single
  `(1, total_seq)` tensor for layerwise forward.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch


@dataclass
class Chunk:
    text: str
    token_ids: list[int]
    chunk_id: str
    is_cached: bool = False

    @property
    def length(self) -> int:
        return len(self.token_ids)


def _stable_id(text: str, token_ids: list[int]) -> str:
    """Deterministic chunk_id from (text, tokenization). 16-char hex prefix."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(b"|")
    h.update(",".join(str(t) for t in token_ids).encode("utf-8"))
    return h.hexdigest()[:16]


def chunk_texts(tokenizer, texts: list[str]) -> list[Chunk]:
    """Tokenize each text independently and wrap as Chunks.

    No special tokens added. Uses tokenizer's default config.
    """
    chunks: list[Chunk] = []
    for text in texts:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        chunks.append(Chunk(
            text=text,
            token_ids=token_ids,
            chunk_id=_stable_id(text, token_ids),
        ))
    return chunks


def fused_input_ids(chunks: list[Chunk], device: torch.device | None = None) -> torch.Tensor:
    """Concat chunk token_ids → (1, total_seq) tensor."""
    flat: list[int] = []
    for c in chunks:
        flat.extend(c.token_ids)
    t = torch.tensor([flat], dtype=torch.long)
    if device is not None:
        t = t.to(device)
    return t


def chunk_offsets(chunks: list[Chunk]) -> list[tuple[int, int]]:
    """[(start, end) ...] absolute positions of each chunk in fused sequence."""
    offsets: list[tuple[int, int]] = []
    pos = 0
    for c in chunks:
        offsets.append((pos, pos + c.length))
        pos += c.length
    return offsets
