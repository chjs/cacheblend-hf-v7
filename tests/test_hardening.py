"""Hardening regression tests for issues found in the adversarial re-review.

N1: precompute_chunk_kv rejects a 0-token chunk (else a (1,0) forward crashes
    inside HF and the C2 seq_len guard passes trivially as 0==0).
N2: call_decoder_layer fails loud when a layer's forward exposes no cache
    parameter (signature wrapped/erased) instead of silently dropping the cache.
"""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cacheblend.chunker import Chunk, _stable_id  # noqa: E402
from cacheblend.precompute import precompute_chunk_kv  # noqa: E402
from cacheblend.model import call_decoder_layer, _layer_spec  # noqa: E402


def test_N1_empty_chunk_rejected():
    empty = Chunk(text="", token_ids=[], chunk_id=_stable_id("", []))
    with pytest.raises(ValueError, match="0 tokens"):
        precompute_chunk_kv(None, empty)  # guard fires before model is touched
    print("[N1] empty chunk rejected by precompute_chunk_kv")


class _ErasedLayer(nn.Module):
    """A layer whose forward signature is erased to (*args, **kwargs)."""
    def forward(self, *args, **kwargs):
        return (args[0] if args else kwargs["hidden_states"],)


def test_N2_no_cache_param_fails_loud():
    layer = _ErasedLayer()
    cache_name, _, has_var_kw = _layer_spec(layer)
    assert cache_name is None and has_var_kw, "fixture should erase the cache param"
    with pytest.raises(RuntimeError, match="no\\s+past_key_value"):
        call_decoder_layer(
            layer, torch.zeros(1, 3, 8),
            attention_mask=None, position_ids=None,
            past_key_values=DynamicCache(), use_cache=True,
            cache_position=None, position_embeddings=None,
        )
    print("[N2] cache_name=None raises (no silent drop)")


if __name__ == "__main__":
    test_N1_empty_chunk_rejected()
    print("PASS N1")
    test_N2_no_cache_param_fails_loud()
    print("PASS N2")
    print("\nALL HARDENING TESTS PASSED")
