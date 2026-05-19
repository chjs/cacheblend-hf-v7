"""Fusion strategies — full_recompute / full_reuse / selective / prefix_cache.

- `fuse_full_recompute`: baseline. Runs forward over fused input_ids end-to-end.
- `fuse_full_reuse`: KV reuse for *all* chunk tokens via forward-hook injection.
  k_proj output → replaced with stored pre-RoPE K (then HF's apply_rotary_pos_emb
  re-applies RoPE at the *blended* global positions, exactly what we want).
  v_proj output → replaced with stored V.
- `fuse_selective`: paper §4 selective recompute — LMCache `blender.process_qkv`
  1:1 port. Sparse Q × full mixed-K/V forward from check_layer onward. See
  docs/notes/lmcache-1to1-comparison.md for the formal 1:1 mapping.
- `fuse_prefix_cache`: only the first chunk reuses cached K/V (Phase 4 baseline).

Boundary safe-shortcut [L13]: ratio=0 → full_reuse, ratio≥1 → full_recompute,
single-chunk full_reuse → full_recompute (single prefix is identical path).
"""
from __future__ import annotations

import torch

from cacheblend.chunker import Chunk, chunk_offsets, fused_input_ids
from cacheblend.kv_store import KVStore
from cacheblend.model import LayerwiseOutput


def fuse_full_recompute(
    layerwise_model,
    chunks: list[Chunk],
    return_layerwise_output: bool = False,
):
    """Standard prefill over the fused sequence — no KV reuse.

    Returns logits of shape (1, total_seq, vocab_size). When `return_layerwise_output=True`,
    returns the full `LayerwiseOutput(logits, past_key_values)` for greedy decode.
    """
    input_ids = fused_input_ids(chunks, device=layerwise_model.device)
    with torch.inference_mode():
        out = layerwise_model.forward_layerwise(input_ids=input_ids, use_cache=True)
    return out if return_layerwise_output else out.logits


def fuse_full_reuse(
    layerwise_model,
    chunks: list[Chunk],
    kv_store: KVStore,
    return_layerwise_output: bool = False,
):
    """Reuse stored pre-RoPE K + V for every chunk token via forward-hook injection.

    Boundary safe-shortcut [L13]: with a single chunk, the fused sequence is
    just that chunk at positions 0..L-1 — same as `fuse_full_recompute`. We
    dispatch directly to keep max_diff = 0 for the single-prefix case.

    For multi-chunk, hooks override `k_proj` and `v_proj` outputs at the cached
    chunk positions. HF's `apply_rotary_pos_emb` then runs *on top* of our
    pre-RoPE K with the blended global positions — equivalent to RoPE-shifting
    chunk-local pre-RoPE K to global positions (paper §4).
    """
    # Boundary: single chunk = single prefix → identical path to full_recompute.
    if len(chunks) <= 1:
        return fuse_full_recompute(layerwise_model, chunks, return_layerwise_output=return_layerwise_output)

    offsets = chunk_offsets(chunks)
    total_seq = offsets[-1][1]

    # Build per-layer K/V override tensors of shape (1, total_seq, hidden_kv).
    # Slice in stored chunk K/V at each chunk's [start:end) range.
    n_layers = layerwise_model.num_layers
    inner = layerwise_model._inner
    attn0 = inner.layers[0].self_attn
    hidden_kv = attn0.config.num_key_value_heads * attn0.head_dim

    device = layerwise_model.device
    dtype = layerwise_model.dtype

    K_override = [
        torch.zeros((1, total_seq, hidden_kv), dtype=dtype, device=device)
        for _ in range(n_layers)
    ]
    V_override = [
        torch.zeros((1, total_seq, hidden_kv), dtype=dtype, device=device)
        for _ in range(n_layers)
    ]
    for chunk, (start, end) in zip(chunks, offsets):
        if not kv_store.has(chunk.chunk_id):
            raise KeyError(
                f"fuse_full_reuse: chunk {chunk.chunk_id!r} not in KVStore. "
                f"Did you precompute all chunks first?"
            )
        entry = kv_store.get(chunk.chunk_id)
        for li in range(n_layers):
            K_override[li][:, start:end, :] = entry["K"][li]
            V_override[li][:, start:end, :] = entry["V"][li]

    # Install hooks: replace k_proj/v_proj output for each layer.
    handles = []
    for li, layer in enumerate(inner.layers):
        def make_k_hook(idx):
            def hook(_m, _inp, _out):
                return K_override[idx]
            return hook

        def make_v_hook(idx):
            def hook(_m, _inp, _out):
                return V_override[idx]
            return hook

        handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(li)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(li)))

    try:
        input_ids = fused_input_ids(chunks, device=device)
        with torch.inference_mode():
            out = layerwise_model.forward_layerwise(input_ids=input_ids, use_cache=True)
        return out if return_layerwise_output else out.logits
    finally:
        for h in handles:
            h.remove()


def fuse_prefix_cache(
    layerwise_model,
    chunks: list[Chunk],
    kv_store: KVStore,
    return_layerwise_output: bool = False,
):
    """Baseline: only the first chunk reuses cached K/V; remaining chunks fresh.

    Boundary safe-shortcut [L13]: with a single chunk this is identical to
    fuse_full_recompute (the "first chunk" is the entire input, and reusing
    it at positions 0..L-1 gives the same result).

    For multi-chunk: hooks fire only for the first chunk's K/V positions
    (range [0, L_first)) at every layer. Position-correct because the first
    chunk's standalone-prefill positions == its fused-prefill positions
    (both 0..L_first-1). HF's apply_rotary_pos_emb runs on top normally.
    """
    if len(chunks) <= 1:
        return fuse_full_recompute(layerwise_model, chunks, return_layerwise_output=return_layerwise_output)

    offsets = chunk_offsets(chunks)
    total_seq = offsets[-1][1]
    first_chunk = chunks[0]
    first_start, first_end = offsets[0]

    if not kv_store.has(first_chunk.chunk_id):
        raise KeyError(
            f"fuse_prefix_cache: first chunk {first_chunk.chunk_id!r} not in KVStore"
        )

    n_layers = layerwise_model.num_layers
    inner = layerwise_model._inner
    entry = kv_store.get(first_chunk.chunk_id)

    # Per-layer: replace ONLY positions [first_start, first_end) of fresh K/V
    # with cached values. Other positions pass through unchanged.
    handles = []
    for li, layer in enumerate(inner.layers):
        K_cached_first = entry["K"][li]   # (1, L_first, hidden_kv)
        V_cached_first = entry["V"][li]

        def make_k_hook(cached, start=first_start, end=first_end):
            def hook(_m, _inp, output):
                result = output.clone()
                result[:, start:end, :] = cached
                return result
            return hook

        def make_v_hook(cached, start=first_start, end=first_end):
            def hook(_m, _inp, output):
                result = output.clone()
                result[:, start:end, :] = cached
                return result
            return hook

        handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(K_cached_first)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(V_cached_first)))

    try:
        input_ids = fused_input_ids(chunks, device=layerwise_model.device)
        with torch.inference_mode():
            out = layerwise_model.forward_layerwise(input_ids=input_ids, use_cache=True)
        return out if return_layerwise_output else out.logits
    finally:
        for h in handles:
            h.remove()


# ───────────────────────────────────────────────────────────────────────────────
# fuse_selective — paper §4 selective recompute (LMCache process_qkv 1:1 port)
# ───────────────────────────────────────────────────────────────────────────────


def fuse_selective(
    layerwise_model,
    chunks: list[Chunk],
    kv_store: KVStore,
    recompute_ratio: float = 0.15,
    check_layer: int = 1,
    return_layerwise_output: bool = False,
    return_hkvd_indices: bool = False,
):
    """Paper §4 selective recompute — LMCache `blender.process_qkv` 1:1 port.

    Algorithm:
      Layers 0..check_layer-1: full fresh forward (matches LMCache).
      Layer check_layer: select top-K HKVD via kv_deviation(pre-RoPE K_fresh,
        pre-RoPE K_cached). Slice Q (and residual) to top_indices — from here
        hidden_state stays SPARSE [topk_num] while K/V cache stays FULL
        (cached@non-HKVD scattered with fresh@HKVD).
      Layers check_layer+1..end: sparse Q × full mixed-K/V attention.

    The end past_key_values (DynamicCache) has full-length K/V at every layer:
      - Layers 0..check_layer-1: K/V from full fresh forward
      - Layer check_layer..end:  K cached(RoPE-shifted) at non-HKVD positions +
                                 K fresh-from-sparse-forward at HKVD positions
      - V same pattern (no RoPE)

    For greedy decoding to work, the last position MUST be in top_indices
    (we enforce by force-include if absent — minor LMCache deviation).

    Boundary safe-shortcuts [L13] (bit-exact edges):
      - ratio == 0     → fuse_full_reuse
      - ratio >= 1     → fuse_full_recompute
      - len(chunks) <= 1 → fuse_full_recompute

    See `docs/notes/lmcache-1to1-comparison.md` for the formal 1:1 mapping.
    """
    # Boundary safe-shortcuts.
    if recompute_ratio == 0:
        return fuse_full_reuse(layerwise_model, chunks, kv_store,
                               return_layerwise_output=return_layerwise_output)
    if recompute_ratio >= 1:
        return fuse_full_recompute(layerwise_model, chunks,
                                   return_layerwise_output=return_layerwise_output)
    if len(chunks) <= 1:
        return fuse_full_recompute(layerwise_model, chunks,
                                   return_layerwise_output=return_layerwise_output)

    from transformers.cache_utils import DynamicCache
    from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
    from cacheblend.hkvd import kv_deviation, select_top_k
    # SDPA is memory-efficient for sparse-Q × full-K attention at long context.
    # eager_attention_forward materializes the (num_heads, topk_num, S) score
    # tensor explicitly — OOM at S>=20k. SDPA dispatches to flash / memefficient
    # kernels which compute attention without materializing the full score matrix.
    from torch.nn.functional import scaled_dot_product_attention as _sdpa

    inner = layerwise_model._inner
    n_layers = layerwise_model.num_layers
    device = layerwise_model.device
    dtype = layerwise_model.dtype
    attn0 = inner.layers[0].self_attn
    num_kv_heads = attn0.config.num_key_value_heads
    num_heads = attn0.config.num_attention_heads
    head_dim = attn0.head_dim
    hidden_kv = num_kv_heads * head_dim

    # ── Build full-length cached K_pre + V from KVStore ─────────────────────
    offsets = chunk_offsets(chunks)
    total_seq = offsets[-1][1]

    K_stored_pre = [torch.zeros((1, total_seq, hidden_kv), dtype=dtype, device=device)
                    for _ in range(n_layers)]
    V_stored = [torch.zeros((1, total_seq, hidden_kv), dtype=dtype, device=device)
                for _ in range(n_layers)]
    for chunk, (start, end) in zip(chunks, offsets):
        if not kv_store.has(chunk.chunk_id):
            raise KeyError(f"fuse_selective: chunk {chunk.chunk_id!r} not in KVStore")
        entry = kv_store.get(chunk.chunk_id)
        for li in range(n_layers):
            K_stored_pre[li][:, start:end, :] = entry["K"][li]
            V_stored[li][:, start:end, :] = entry["V"][li]

    # ── Forward setup ───────────────────────────────────────────────────────
    input_ids = fused_input_ids(chunks, device=device)
    position_ids_full = torch.arange(total_seq, device=device).unsqueeze(0)
    cache_position_full = torch.arange(total_seq, device=device)

    past_key_values = DynamicCache()
    # Reset pre-RoPE K capture (LayerwiseModel's k_proj hooks fire during full forward).
    layerwise_model._pre_rope_k = {}

    with torch.inference_mode():
        hidden_states = inner.embed_tokens(input_ids)
        cos_full, sin_full = inner.rotary_emb(hidden_states, position_ids_full)
        position_embeddings_full = (cos_full, sin_full)

        causal_mask_full = inner._update_causal_mask(
            attention_mask=None,
            input_tensor=hidden_states,
            cache_position=cache_position_full,
            past_key_values=past_key_values,
            output_attentions=False,
        )
        # SDPA / FlashAttention2 implementations return None here (they build
        # their own causal mask internally inside the attention kernel). Our
        # sparse layers manually call eager_attention_forward, which REQUIRES
        # an explicit additive mask. Build the standard (1,1,S,S) -inf-above-
        # diagonal mask used by HF's eager path.
        if causal_mask_full is None:
            mask = torch.zeros((1, 1, total_seq, total_seq), dtype=dtype, device=device)
            triu = torch.triu(
                torch.ones(total_seq, total_seq, dtype=torch.bool, device=device),
                diagonal=1,
            )
            mask.masked_fill_(triu, float("-inf"))
            causal_mask_full = mask

        # ── Layers 0..check_layer-1: full fresh forward (matches LMCache) ───
        # NOTE: HF transformers 4.49 MistralDecoderLayer uses kwarg
        # `past_key_value` (singular). Plural silently goes into **kwargs and
        # is ignored → cache stays empty. Same for MistralAttention.
        for li in range(check_layer):
            out = inner.layers[li](
                hidden_states=hidden_states,
                attention_mask=causal_mask_full,
                position_ids=position_ids_full,
                past_key_value=past_key_values,
                use_cache=True,
                cache_position=cache_position_full,
                position_embeddings=position_embeddings_full,
            )
            hidden_states = out if not isinstance(out, tuple) else out[0]

        # ── Layer check_layer: manual forward with HKVD + sparse slicing ────
        layer_ck = inner.layers[check_layer]
        attn_ck = layer_ck.self_attn

        # input_layernorm + qkv projection (FULL)
        residual_full = hidden_states
        h_normed = layer_ck.input_layernorm(hidden_states)

        q_full = attn_ck.q_proj(h_normed)         # (1, S, num_heads*head_dim)
        k_full_pre = attn_ck.k_proj(h_normed)     # (1, S, hidden_kv) PRE-RoPE
        v_full = attn_ck.v_proj(h_normed)         # (1, S, hidden_kv)

        # HKVD selection (pre-RoPE K is invariant; LMCache compares post-RoPE,
        # but RoPE preserves squared L2 → mathematically identical).
        deviations = kv_deviation(k_full_pre, K_stored_pre[check_layer])
        top_indices = select_top_k(deviations, recompute_ratio)
        # Force-include last position so greedy decode has valid logits there.
        # (Minor deviation from LMCache; in practice last position usually
        # ranks top by deviation, but guard for safety.)
        last_pos = total_seq - 1
        if last_pos not in top_indices.tolist():
            # Drop the lowest-deviation selected to make room.
            sel_devs = deviations[top_indices]
            drop_idx = top_indices[sel_devs.argmin()].item()
            top_indices = torch.tensor(
                sorted([i for i in top_indices.tolist() if i != drop_idx] + [last_pos]),
                dtype=top_indices.dtype, device=top_indices.device,
            )
        topk_num = top_indices.shape[0]

        # Build mixed K_pre and V (cached at non-top, fresh at top).
        k_mixed_pre = k_full_pre.clone()
        v_mixed = v_full.clone()
        non_top_mask = torch.ones(total_seq, dtype=torch.bool, device=device)
        non_top_mask[top_indices] = False
        k_mixed_pre[:, non_top_mask, :] = K_stored_pre[check_layer][:, non_top_mask, :]
        v_mixed[:, non_top_mask, :] = V_stored[check_layer][:, non_top_mask, :]

        # Reshape to heads (matches HF MistralAttention reshape).
        hidden_shape_full = (1, total_seq, -1, head_dim)
        q_full_heads = q_full.view(hidden_shape_full).transpose(1, 2)            # (1, num_heads, S, hd)
        k_mixed_heads_pre = k_mixed_pre.view(hidden_shape_full).transpose(1, 2)  # (1, num_kv, S, hd)
        v_mixed_heads = v_mixed.view(hidden_shape_full).transpose(1, 2)

        # Apply RoPE on full Q and full mixed K (positions = full).
        q_full_heads_post, k_full_post = apply_rotary_pos_emb(
            q_full_heads, k_mixed_heads_pre, cos_full, sin_full
        )

        # Slice Q (and residual) to top_indices — this is LMCache's process_qkv.
        q_sparse_post = q_full_heads_post[:, :, top_indices, :]  # (1, num_heads, topk_num, hd)
        residual_sparse = residual_full[:, top_indices, :]       # (1, topk_num, hidden)

        # Update past_key_values with FULL mixed K + V at check_layer.
        past_key_values.update(k_full_post, v_mixed_heads, check_layer)

        # Causal mask for sparse Q × full K: slice rows of the full mask.
        # NOTE: HF's _update_causal_mask with DynamicCache pads K with a +1
        # future cache slot [L38]. SDPA requires exact (Q, K) match — slice
        # K dim to total_seq.
        sparse_causal_mask = causal_mask_full[:, :, top_indices, :total_seq]

        # SDPA: sparse Q × full mixed K/V. Repeat KV heads for GQA.
        n_rep = num_heads // num_kv_heads
        k_rep = k_full_post.repeat_interleave(n_rep, dim=1)
        v_rep = v_mixed_heads.repeat_interleave(n_rep, dim=1)
        attn_out_ck = _sdpa(
            q_sparse_post, k_rep, v_rep,
            attn_mask=sparse_causal_mask,
            scale=attn_ck.scaling,
        )  # (1, num_heads, topk_num, head_dim)
        attn_out_ck = attn_out_ck.transpose(1, 2).reshape(
            1, topk_num, num_heads * head_dim,
        ).contiguous()
        attn_out_ck = attn_ck.o_proj(attn_out_ck)

        # Residual addition + FFN (all sparse).
        h_sparse = residual_sparse + attn_out_ck
        residual_sparse2 = h_sparse
        h_sparse_normed = layer_ck.post_attention_layernorm(h_sparse)
        h_sparse = layer_ck.mlp(h_sparse_normed)
        h_sparse = residual_sparse2 + h_sparse

        # ── Layers check_layer+1..end: sparse hidden, full mixed K/V cache ─
        cos_sparse = cos_full[:, top_indices, :]
        sin_sparse = sin_full[:, top_indices, :]

        # cache_position for sparse: original positions of top_indices.
        # (Matches what LMCache stores — sparse Q's "true" positions.)

        for li in range(check_layer + 1, n_layers):
            layer_li = inner.layers[li]
            attn_li = layer_li.self_attn

            residual_sparse_in = h_sparse
            h_normed_sparse = layer_li.input_layernorm(h_sparse)

            q_sparse = attn_li.q_proj(h_normed_sparse)       # (1, topk_num, num_heads*hd)
            k_sparse_pre = attn_li.k_proj(h_normed_sparse)   # (1, topk_num, hidden_kv) PRE-RoPE
            v_sparse = attn_li.v_proj(h_normed_sparse)       # (1, topk_num, hidden_kv)

            sparse_shape = (1, topk_num, -1, head_dim)
            q_sparse_heads = q_sparse.view(sparse_shape).transpose(1, 2)
            k_sparse_heads_pre = k_sparse_pre.view(sparse_shape).transpose(1, 2)
            v_sparse_heads = v_sparse.view(sparse_shape).transpose(1, 2)

            # RoPE on sparse Q, K at the actual (top_indices) positions.
            q_sparse_post, k_sparse_post = apply_rotary_pos_emb(
                q_sparse_heads, k_sparse_heads_pre, cos_sparse, sin_sparse
            )

            # Build full K cache at this layer: cached(RoPE-shifted) at non-top,
            # fresh(post-RoPE) at top. cached K_pre is at chunk-local positions;
            # we shift to fused-global positions using HF's apply_rotary_pos_emb.
            k_cached_pre_full = K_stored_pre[li]  # (1, S, hidden_kv)
            k_cached_pre_heads = k_cached_pre_full.view(
                1, total_seq, num_kv_heads, head_dim
            ).transpose(1, 2)  # (1, num_kv, S, hd)
            # apply_rotary_pos_emb wants (q, k, cos, sin); dummy q.
            dummy_q = torch.zeros_like(k_cached_pre_heads)
            _q_dummy, k_cached_post_heads = apply_rotary_pos_emb(
                dummy_q, k_cached_pre_heads, cos_full, sin_full
            )

            # Scatter fresh at top.
            k_full_li = k_cached_post_heads.clone()
            k_full_li[:, :, top_indices, :] = k_sparse_post

            # V: cached at non-top, fresh at top (V has no RoPE).
            v_cached_heads = V_stored[li].view(
                1, total_seq, num_kv_heads, head_dim
            ).transpose(1, 2)
            v_full_li = v_cached_heads.clone()
            v_full_li[:, :, top_indices, :] = v_sparse_heads

            past_key_values.update(k_full_li, v_full_li, li)

            # SDPA: sparse Q × full mixed K/V attention.
            sparse_causal_mask_li = causal_mask_full[:, :, top_indices, :total_seq]
            k_rep_li = k_full_li.repeat_interleave(n_rep, dim=1)
            v_rep_li = v_full_li.repeat_interleave(n_rep, dim=1)
            attn_out_li = _sdpa(
                q_sparse_post, k_rep_li, v_rep_li,
                attn_mask=sparse_causal_mask_li,
                scale=attn_li.scaling,
            )
            attn_out_li = attn_out_li.transpose(1, 2).reshape(
                1, topk_num, num_heads * head_dim,
            ).contiguous()
            attn_out_li = attn_li.o_proj(attn_out_li)

            h_sparse = residual_sparse_in + attn_out_li
            residual_sparse_in2 = h_sparse
            h_sparse_normed = layer_li.post_attention_layernorm(h_sparse)
            h_sparse = layer_li.mlp(h_sparse_normed)
            h_sparse = residual_sparse_in2 + h_sparse

        # ── Final norm + lm_head on SPARSE hidden_state ─────────────────────
        h_sparse_normed = inner.norm(h_sparse)
        logits_sparse = layerwise_model.model.lm_head(h_sparse_normed)  # (1, topk_num, vocab)

        # Scatter sparse logits back into full-length tensor at top_indices.
        # Only the last position's logits is used by greedy decode; other
        # non-top positions get zeros (no fresh computation occurred there).
        vocab_size = logits_sparse.shape[-1]
        logits_full = torch.zeros((1, total_seq, vocab_size), dtype=logits_sparse.dtype, device=device)
        logits_full[:, top_indices, :] = logits_sparse

    out_obj = LayerwiseOutput(logits=logits_full, past_key_values=past_key_values)
    result = out_obj if return_layerwise_output else logits_full
    if return_hkvd_indices:
        return result, top_indices
    return result
