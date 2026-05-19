"""LayerwiseModel — wraps HF causal LM for layer-by-layer forward + pre-RoPE K capture.

Phase 1 implementation. Mistral-7B / Llama-3.1 family.

Design:
- `attn_implementation="eager"` 의무 (FlashAttention/SDPA 회피, bit-exact 검증 가능, v3 핵심).
- HF model 의 internal modules 를 직접 호출:
    embed_tokens, rotary_emb, layers[i], norm, lm_head
- Pre-RoPE K capture: 매 layer 의 `self_attn.k_proj` 에 forward-hook.
  k_proj output 은 RoPE 적용 전 K (shape [batch, seq, num_kv_heads * head_dim]).
- DynamicCache 사용 — past_key_values=None 으로 호출 시 layer 별 KV 누적.

This module replaces the Phase 0 stub.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


_DTYPE_MAP = {"float16": torch.float16, "fp16": torch.float16,
              "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
              "float32": torch.float32, "fp32": torch.float32}


@dataclass
class LayerwiseOutput:
    """Output of LayerwiseModel.forward_layerwise — drop-in for HF CausalLMOutputWithPast."""
    logits: torch.Tensor
    past_key_values: Optional[DynamicCache] = None


class LayerwiseModel:
    """Wraps an HF causal LM to expose its forward as layer-by-layer steps.

    Public API (Phase 1 acceptance, 7 methods):
        embed_tokens, compute_position_embeddings, build_causal_mask,
        prefill_layer, final_norm_and_lm_head, forward_layerwise, get_pre_rope_k
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        device: Optional[str] = None,
        attn_implementation: str = "eager",
    ):
        torch_dtype = _DTYPE_MAP[dtype]
        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Default attn_implementation is "eager" (bit-exact for Phase 0-7 logit checks).
        # For long-context runs (Loong), pass "flash_attention_2" to fit 50k+ tokens in
        # 24-48GB GPUs. fuse_selective manually calls SDPA for sparse layers
        # (check_layer+), so the configured impl only affects the full-forward
        # layers (0..check_layer-1) — flash/eager mix is safe.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        # Direct refs to internal modules to avoid repeated attribute lookup.
        self._inner = self.model.model        # MistralModel / LlamaModel
        self.num_layers = len(self._inner.layers)
        self.dtype = torch_dtype

        # Pre-RoPE K capture state. layer_idx → tensor of shape (batch, seq, num_kv_heads*head_dim).
        # Cleared at the start of every forward_layerwise call.
        self._pre_rope_k: dict[int, torch.Tensor] = {}
        self._hook_handles: list = []
        self._install_k_proj_hooks()

    # ── internal: hook plumbing ─────────────────────────────────────────────

    def _install_k_proj_hooks(self) -> None:
        for layer_idx, layer in enumerate(self._inner.layers):
            k_proj = layer.self_attn.k_proj

            def make_hook(idx: int):
                def hook(_module, _inputs, output):
                    # output is pre-RoPE K (k_proj output, before view/transpose/RoPE).
                    self._pre_rope_k[idx] = output.detach()
                return hook

            handle = k_proj.register_forward_hook(make_hook(layer_idx))
            self._hook_handles.append(handle)

    # ── 7 required methods ──────────────────────────────────────────────────

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token IDs → embeddings. Wraps `model.model.embed_tokens`."""
        return self._inner.embed_tokens(input_ids)

    def compute_position_embeddings(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) tensors for RoPE. Shared across all decoder layers."""
        return self._inner.rotary_emb(hidden_states, position_ids)

    def build_causal_mask(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
    ) -> Optional[torch.Tensor]:
        """Standard causal mask via HF's `_update_causal_mask`."""
        inputs_embeds = self.embed_tokens(input_ids)
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device,
        )
        return self._inner._update_causal_mask(
            attention_mask=None,
            input_tensor=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            output_attentions=False,
        )

    def prefill_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[DynamicCache] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One decoder-layer forward. position_embeddings (cos, sin) shared across layers.

        Returns updated hidden_states. past_key_values is mutated in-place by
        DynamicCache.update inside the layer.
        """
        layer = self._inner.layers[layer_idx]
        if position_embeddings is None:
            position_embeddings = self.compute_position_embeddings(hidden_states, position_ids)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + hidden_states.shape[1], device=hidden_states.device,
            )
        out = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=past_key_values is not None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        return out[0]

    def final_norm_and_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """RMSNorm + LM head → logits."""
        hidden_states = self._inner.norm(hidden_states)
        return self.model.lm_head(hidden_states)

    def forward_layerwise(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> LayerwiseOutput:
        """Replicate the HF MistralModel.forward path step-by-step."""
        # Reset pre-RoPE K capture for this call.
        self._pre_rope_k = {}

        past_key_values = DynamicCache() if use_cache else None

        hidden_states = self.embed_tokens(input_ids)

        # Build position_ids if absent.
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + hidden_states.shape[1], device=hidden_states.device,
            )
            position_ids = cache_position.unsqueeze(0)
        else:
            cache_position = torch.arange(
                position_ids.min().item(),
                position_ids.min().item() + hidden_states.shape[1],
                device=hidden_states.device,
            )

        # Build causal mask the same way HF does.
        causal_mask = self._inner._update_causal_mask(
            attention_mask=attention_mask,
            input_tensor=hidden_states,
            cache_position=cache_position,
            past_key_values=past_key_values,
            output_attentions=False,
        )

        # cos/sin shared across all decoder layers (HF does it once).
        position_embeddings = self.compute_position_embeddings(hidden_states, position_ids)

        for layer_idx in range(self.num_layers):
            hidden_states = self.prefill_layer(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
            )

        logits = self.final_norm_and_lm_head(hidden_states)
        return LayerwiseOutput(logits=logits, past_key_values=past_key_values)

    def get_pre_rope_k(self, layer_idx: int) -> torch.Tensor:
        """Pre-RoPE K (k_proj output) captured during the most recent forward.

        Shape: (batch, seq_len, num_kv_heads * head_dim). Raises KeyError if no
        forward has run yet, or if layer_idx is out of range.
        """
        if layer_idx not in self._pre_rope_k:
            raise KeyError(
                f"No pre-RoPE K for layer {layer_idx}. "
                f"Run forward_layerwise() first. Available: {sorted(self._pre_rope_k.keys())}"
            )
        return self._pre_rope_k[layer_idx]

    # ── cleanup ─────────────────────────────────────────────────────────────

    def __del__(self):
        for h in getattr(self, "_hook_handles", []):
            try:
                h.remove()
            except Exception:
                pass
