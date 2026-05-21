"""Generic musique workload — runs the YaoJiayi/CacheBlend musique experiment
on ANY HuggingFace instruction model.

`blend_musique.py` is the YaoJiayi original (Mistral-7B-Instruct-v0.2,
verbatim, frozen). This file generalizes it.

The musique experiment has exactly ONE model-dependent part: the instruction
wrapper around the prompt —

    Mistral   [INST] ... [/INST]
    Llama-3   <|start_header_id|>user<|end_header_id|> ... <|eot_id|>...assistant...
    Qwen      <|im_start|>user ... <|im_end|>...

Everything else — dataset (musique_s.json), prefix/query prompts,
`build_qa_prompt`, chunking ([prefix] [doc1]..[docN] [query]), `compute_f1`,
and the CacheBlend-vs-full-prefill comparison — is model-independent.

So there is NO need for a new script per model. This single file picks the
wrapper from a small per-family table; for a model family not in the table it
derives the wrapper from the tokenizer's chat template. A genuinely new model
is just `CACHEBLEND_MODEL=...` — no new file (and, at most, a one-line table
entry if the chat-template fallback misbehaves).

NOT a verbatim copy of any YaoJiayi file — this is our own generalization and
is freely editable. The experiment definition it reproduces is identical to
blend_musique.py.

Env:
    CACHEBLEND_MODEL   HF model id (default: mistralai/Mistral-7B-Instruct-v0.2)

Run via the shared runner:
    CACHEBLEND_WORKLOAD=blend_musique_generic.py \
    CACHEBLEND_MODEL=meta-llama/Llama-3.1-8B-Instruct \
        python benchmarks/musique/run_blend_musique.py
"""
import os

from vllm import LLM, SamplingParams
import numpy as np
from transformers import AutoTokenizer
from utils import load_dataset, build_qa_prompt, compute_f1


MODEL = os.environ.get("CACHEBLEND_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# Per-family instruction wrapper: (user_turn_open, assistant_turn_open).
# Single user turn, no system message — matches blend_musique.py, which places
# prefix_prompt directly after [INST]. The BOS token is NOT included here; the
# tokenizer/shim prepends it on encode.
_WRAPPERS = {
    "mistral": ("[INST]", "[/INST]"),
    "llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama3":  ("<|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "qwen":    ("<|im_start|>user\n",
                "<|im_end|>\n<|im_start|>assistant\n"),
}


def _resolve_wrapper(model_id, tokenizer):
    """Return (user_open, assistant_open) for the model.

    Known families come from the table; otherwise the wrapper is derived from
    the tokenizer's chat template by templating a sentinel and splitting.
    """
    mid = model_id.lower()
    for key, wrap in _WRAPPERS.items():
        if key in mid:
            return wrap

    # Fallback — derive from the chat template.
    sentinel = "\x00CONTENT\x00"
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False, add_generation_prompt=True,
    )
    if sentinel not in templated:
        raise RuntimeError(
            f"could not derive instruction wrapper for {model_id!r}; "
            f"add an entry to _WRAPPERS in blend_musique_generic.py"
        )
    pre, post = templated.split(sentinel, 1)
    # Strip a leading BOS — the tokenizer/shim adds BOS on encode; a literal
    # BOS string in the chunk-0 text would double it.
    bos = tokenizer.bos_token or ""
    if bos and pre.startswith(bos):
        pre = pre[len(bos):]
    return pre, post


eval_dataset = load_dataset("inputs/musique_s.json")

llm = LLM(model=MODEL, gpu_memory_utilization=0.5)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
llm.set_tokenizer(tokenizer)

user_open, assistant_open = _resolve_wrapper(MODEL, tokenizer)
print(f"[blend_musique_generic] model: {MODEL}")
print(f"[blend_musique_generic] user_open:      {user_open!r}")
print(f"[blend_musique_generic] assistant_open: {assistant_open!r}")

# Instruction prompts — VERBATIM from blend_musique.py (experiment definition,
# model-independent).
prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

ttft_blend = []
ttft_full = []
f1_blend = []
f1_full = []

for ex in eval_dataset:
    answers = ex["answers"]
    doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)

    # Chunk layout (parallel to blend_musique.py's doc_chunk_ids):
    #   chunk 0    = instruction-wrapper user-open + prefix_prompt
    #   chunk 1..N = one document each
    #   chunk N+1  = query + instruction-wrapper assistant-open
    chunk_texts = [user_open + prefix_prompt]
    chunk_texts += list(doc_prompts)
    chunk_texts += [q_prompt + assistant_open]

    input_prompt = "".join(chunk_texts)

    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata

    # ── Collect: per-chunk KV precompute ───────────────────────────────────
    # blend_musique.py reads layers[j].self_attn.hack_kv here to build
    # chunk_past_key_values; our shim stores per-chunk K/V in its own KVStore
    # during this generate() call, so no hack_kv read is needed.
    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata['check'] = False
    collect_params = SamplingParams(temperature=0, max_tokens=1)
    for ct in chunk_texts:
        llm.generate([ct], collect_params)

    gen_params = SamplingParams(temperature=0, max_tokens=32)

    # ── Check: CacheBlend selective recompute ──────────────────────────────
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['check'] = True
    output = llm.generate([input_prompt], gen_params)
    res = output[0].outputs[0].text
    print(f"Cached generation: {res}")
    ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
    print(f"TTFT with cache: {ttft}")
    ttft_blend.append(ttft)
    f1 = max([compute_f1(res, answer, tokenizer) for answer in answers])
    f1_blend.append(f1)

    # ── Normal: full prefill ───────────────────────────────────────────────
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['check'] = False
    output = llm.generate([input_prompt], gen_params)
    res = output[0].outputs[0].text
    print(f"Normal generation: {res}")
    ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
    print(f"TTFT with full prefill: {ttft}")
    ttft_full.append(ttft)
    f1 = max([compute_f1(res, answer, tokenizer) for answer in answers])
    f1_full.append(f1)
    print("------------")

print("---------------Result Summary---------------------")
print(f"Model: {MODEL}")
print(f"TTFT with cache: {np.mean(ttft_blend)}")
print(f"TTFT with full prefill: {np.mean(ttft_full)}")
print(f"F1 with cache: {np.mean(f1_blend)}")
print(f"F1 with full prefill: {np.mean(f1_full)}")
