"""
Real small LLMs from the Hugging Face Hub, one per family: same logits and same greedy text as the
reference implementation, with our special tokens added. ~1.7 GB of downloads, so opt-in:

    TINY_LLM_TEST_HUB=1 pytest tests/test_hub_models.py
"""
import os

import pytest
import torch

from tiny_llm import TinyLLM, create_tokenizer, generate_text
from tiny_llm.generation import has_chat_template, stream_chat

HUB_MODELS = ["distilgpt2", "HuggingFaceTB/SmolLM2-135M-Instruct", "EleutherAI/pythia-70m",
              "Qwen/Qwen2.5-0.5B-Instruct"]

pytestmark = [pytest.mark.network, pytest.mark.slow,
              pytest.mark.skipif(os.environ.get("TINY_LLM_TEST_HUB") != "1", reason="set TINY_LLM_TEST_HUB=1")]


LOGITS_PROMPT = "The quick brown fox jumps over the lazy dog because"
GEN_PROMPT = "The capital of France is"


@pytest.fixture(scope="module", params=HUB_MODELS)
def pair(request):
    """Reference outputs from Hugging Face first, then free it before building ours (keeps RAM low)."""
    import gc

    from transformers import AutoModelForCausalLM
    name = request.param
    tokenizer = create_tokenizer(name)
    logits_ids = torch.tensor([tokenizer.encode(LOGITS_PROMPT)])
    gen_ids = torch.tensor([tokenizer.encode(GEN_PROMPT)])

    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    with torch.no_grad():
        ref_logits = hf(logits_ids).logits
        ref_gen = hf.generate(gen_ids, attention_mask=torch.ones_like(gen_ids), max_new_tokens=12,
                              do_sample=False, pad_token_id=tokenizer.pad_token_id)
    del hf
    gc.collect()

    ours = TinyLLM.from_pretrained(name, tokenizer).eval()
    ref = dict(logits_ids=logits_ids, logits=ref_logits, gen_ids=gen_ids, gen=ref_gen)
    return name, tokenizer, ours, ref


def test_logits_match(pair):
    name, tokenizer, ours, ref = pair
    with torch.no_grad():
        a, _ = ours(ref["logits_ids"], ref["logits_ids"])
    b = ref["logits"]
    n = min(a.size(-1), b.size(-1), len(tokenizer) - 5)  # compare the original vocabulary
    a, b = a[..., :n], b[..., :n]
    # float32 noise is relative to the logit scale (Pythia's logits reach ~1000)
    assert (a - b).abs().max() <= 1e-4 * b.abs().max() + 1e-3, (a - b).abs().max()
    assert torch.equal(a.argmax(-1), b.argmax(-1))


def test_greedy_text_matches(pair):
    name, tokenizer, ours, ref = pair
    T = ref["gen_ids"].size(1)
    expected = GEN_PROMPT + tokenizer.decode(ref["gen"][0, T:], skip_special_tokens=True)
    got = generate_text(ours, tokenizer, GEN_PROMPT, max_new_tokens=12, temperature=0)
    # identical unless one of them stopped earlier at an end-of-text token
    assert got == expected or expected.startswith(got) or got.startswith(expected)


def test_chat_models_answer(pair):
    name, tokenizer, ours, _ = pair
    if not has_chat_template(tokenizer):
        pytest.skip("base model")
    reply = "".join(stream_chat(ours, tokenizer, [{"role": "user", "content": "What is the capital of France?"}],
                                max_new_tokens=30, temperature=0))
    print(f"{name}: {reply!r}")
    assert "Paris" in reply
