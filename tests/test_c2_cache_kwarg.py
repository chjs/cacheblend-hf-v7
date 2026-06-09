"""C2 regression test — decoder-layer cache kwarg must never be silently dropped.

Background (docs/CODE-REVIEW-2026-06.md §C2): transformers renamed the decoder
layer cache kwarg `past_key_value` (singular, <=4.51) -> `past_key_values`
(plural, >=4.53/5.x). The old code passed the singular name unconditionally; on
newer transformers it falls into the layer's **kwargs and is SILENTLY IGNORED,
leaving the DynamicCache empty (no error) -> decode reads no KV.

These tests run on CPU with a tiny random Mistral model (no GPU, no HF download).
They verify that cacheblend.model.call_decoder_layer populates the cache on the
*installed* transformers version, and document the raw-singular failure mode.
"""
import sys
from pathlib import Path

import torch
from transformers import MistralConfig, MistralForCausalLM
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cacheblend.model import call_decoder_layer, _layer_spec  # noqa: E402


def _tiny_inner():
    torch.manual_seed(0)
    cfg = MistralConfig(
        vocab_size=320, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512,
    )
    return MistralForCausalLM(cfg).eval().model


def _layer_inputs(inner, S=10):
    input_ids = torch.arange(S).unsqueeze(0) % inner.config.vocab_size
    hs = inner.embed_tokens(input_ids)
    pos = torch.arange(S).unsqueeze(0)
    cos, sin = inner.rotary_emb(hs, pos)
    cache_pos = torch.arange(S)
    mask = torch.zeros((1, 1, S, S))
    mask.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float("-inf"))
    return hs, pos, cache_pos, (cos, sin)


def test_call_decoder_layer_populates_cache():
    """THE fix: call_decoder_layer must fill the cache on this transformers version."""
    inner = _tiny_inner()
    S = 10
    hs, pos, cache_pos, pe = _layer_inputs(inner, S)
    mask = torch.zeros((1, 1, S, S))
    mask.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float("-inf"))

    pkv = DynamicCache()
    call_decoder_layer(
        inner.layers[0], hs,
        attention_mask=mask, position_ids=pos, past_key_values=pkv,
        use_cache=True, cache_position=cache_pos, position_embeddings=pe,
    )
    assert pkv.get_seq_length() == S, (
        f"cache not populated (seq_len={pkv.get_seq_length()}, expected {S}) — C2 regression"
    )


def test_layer_spec_resolves_a_cache_kwarg():
    """The signature introspection must find a usable cache kwarg name."""
    inner = _tiny_inner()
    cache_name, accepted, has_var_kw = _layer_spec(inner.layers[0])
    assert cache_name in ("past_key_values", "past_key_value"), cache_name


def test_raw_singular_kwarg_failure_mode_is_documented():
    """Document the bug: raw singular kwarg leaves cache empty on >=4.53/5.x.

    On <=4.51 singular is valid (cache fills); on newer it is ignored. We only
    assert that the helper's choice differs from / fixes whatever the raw path
    does — i.e. helper always fills, regardless of version.
    """
    inner = _tiny_inner()
    S = 8
    hs, pos, cache_pos, pe = _layer_inputs(inner, S)
    mask = torch.zeros((1, 1, S, S))
    mask.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float("-inf"))

    pkv_raw = DynamicCache()
    inner.layers[0](
        hidden_states=hs, attention_mask=mask, position_ids=pos,
        past_key_value=pkv_raw, use_cache=True, cache_position=cache_pos,
        position_embeddings=pe,
    )
    raw_len = pkv_raw.get_seq_length()

    pkv_fixed = DynamicCache()
    call_decoder_layer(
        inner.layers[0], hs,
        attention_mask=mask, position_ids=pos, past_key_values=pkv_fixed,
        use_cache=True, cache_position=cache_pos, position_embeddings=pe,
    )
    fixed_len = pkv_fixed.get_seq_length()

    # The helper must ALWAYS populate; the raw singular path may or may not,
    # depending on transformers version.
    assert fixed_len == S
    print(f"[C2] raw-singular cache_len={raw_len}  helper cache_len={fixed_len} "
          f"(transformers handles singular={'yes' if raw_len == S else 'NO -> would silently break'})")


if __name__ == "__main__":
    import transformers
    print(f"transformers {transformers.__version__}, torch {torch.__version__}")
    test_call_decoder_layer_populates_cache()
    print("PASS test_call_decoder_layer_populates_cache")
    test_layer_spec_resolves_a_cache_kwarg()
    print("PASS test_layer_spec_resolves_a_cache_kwarg")
    test_raw_singular_kwarg_failure_mode_is_documented()
    print("PASS test_raw_singular_kwarg_failure_mode_is_documented")
    print("\nALL C2 TESTS PASSED")
