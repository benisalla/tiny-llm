"""Our from-scratch implementation must reproduce OpenAI GPT-2 exactly when loaded with its weights."""
import pytest
import torch

from tiny_llm import TinyLLM, answer_me, generate_text

pytestmark = [pytest.mark.network, pytest.mark.slow]


@pytest.fixture(scope="module")
def models(tokenizer):
    try:
        from transformers import GPT2LMHeadModel
        hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        ours = TinyLLM.from_pretrained("gpt2", tokenizer).eval()
    except OSError as e:
        pytest.skip(f"GPT-2 weights unavailable: {e}")
    return ours, hf


def test_logits_match_huggingface(models, tokenizer):
    ours, hf = models
    ids = torch.tensor([tokenizer.encode("Hello, my name is Ismail and I love neural networks because")])
    with torch.no_grad():
        ours_logits, _ = ours(ids, ids)
        hf_logits = hf(ids).logits
    assert torch.allclose(ours_logits[..., :50257], hf_logits, atol=1e-3)


def test_vocab_extended_with_special_tokens(models, tokenizer):
    ours, _ = models
    assert ours.config.vocab_size % 8 == 0 and ours.config.vocab_size >= len(tokenizer)
    new = ours.transformer.wte.weight[50257:]
    assert torch.isfinite(new).all() and new.std() > 0


def test_greedy_generation_matches_huggingface(models, tokenizer):
    ours, hf = models
    prompt = "The capital of France is"
    ids = torch.tensor([tokenizer.encode(prompt)])
    hf_out = hf.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=50256)
    ours_text = generate_text(ours, tokenizer, prompt, max_new_tokens=10, temperature=0)
    assert ours_text == tokenizer.decode(hf_out[0])


def test_answer_me_runs(models, tokenizer):
    ours, _ = models
    answer = answer_me(ours, tokenizer, "What is the capital of France?", "Paris is the capital of France.",
                       max_new_tokens=10, temperature=0)
    assert isinstance(answer, str)


def test_lora_from_pretrained_matches_base(tokenizer, models):
    """With LoRA enabled (B = 0) the model is still exactly GPT-2."""
    ours, _ = models
    lora = TinyLLM.from_pretrained("gpt2", tokenizer, {"lora_rank": 8}).eval()
    ids = torch.tensor([tokenizer.encode("LoRA should not change anything at step zero")])
    with torch.no_grad():
        a, _ = ours(ids, ids)
        b, _ = lora(ids, ids)
    assert torch.allclose(a[..., :50257], b[..., :50257], atol=1e-4)
