"""RoPE shift — apply rotary embedding to pre-RoPE K at target positions.

Pre-RoPE K is stored as `k_proj` output (shape: B, S, num_kv_heads*head_dim).
At retrieval time we re-shape to (B, num_kv_heads, S, head_dim) and apply RoPE
at the *blended* global positions, exactly as HF's `apply_rotary_pos_emb` does.

This is the core of CacheBlend's pre-RoPE storage design (paper §4) — and the
key divergence from LMCache, which stores post-RoPE K at chunk-local positions
without any shift (docs/lmcache-analysis.md §Q3, design-decisions.md §11).
"""
from __future__ import annotations

import torch
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb


def _kproj_to_heads(K_pre_rope: torch.Tensor, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """(B, S, num_kv_heads*head_dim) → (B, num_kv_heads, S, head_dim).

    Matches the reshape that MistralAttention.forward does on k_proj output:
        key_states = self.k_proj(...).view(*input_shape, -1, head_dim).transpose(1, 2)
    """
    B, S, _ = K_pre_rope.shape
    return K_pre_rope.view(B, S, num_kv_heads, head_dim).transpose(1, 2)


def _heads_to_kproj(K_heads: torch.Tensor) -> torch.Tensor:
    """(B, num_kv_heads, S, head_dim) → (B, S, num_kv_heads*head_dim) (inverse)."""
    B, H, S, D = K_heads.shape
    return K_heads.transpose(1, 2).reshape(B, S, H * D)


def apply_rope_shift(
    K_pre_rope: torch.Tensor,
    target_positions: torch.Tensor,
    layerwise_model,
) -> torch.Tensor:
    """Apply RoPE at `target_positions` to pre-RoPE K.

    Args:
        K_pre_rope: (B, S, num_kv_heads * head_dim) — k_proj output (no RoPE).
        target_positions: (B, S) — absolute positions to encode at.
        layerwise_model: cacheblend.LayerwiseModel — provides rotary_emb + head dims.

    Returns:
        K_post_rope: same shape as K_pre_rope, after RoPE at target_positions.
    """
    inner = layerwise_model._inner
    # Use any layer's attention spec (all layers share the same head shape).
    attn0 = inner.layers[0].self_attn
    num_kv_heads = attn0.config.num_key_value_heads
    head_dim = attn0.head_dim

    # (B, S, hidden) → (B, num_kv_heads, S, head_dim)
    K_heads = _kproj_to_heads(K_pre_rope, num_kv_heads, head_dim)

    # rotary_emb wants (x, position_ids); returns (cos, sin) of shape (B, S, head_dim).
    cos, sin = inner.rotary_emb(K_heads, target_positions)

    # apply_rotary_pos_emb expects (q, k, cos, sin). We don't need q rotated — pass zeros
    # of matching shape and discard. Minor overhead, simplest correct path.
    dummy_q = torch.zeros_like(K_heads)
    _q, K_rot = apply_rotary_pos_emb(dummy_q, K_heads, cos, sin)

    return _heads_to_kproj(K_rot)
