"""
Every supported family must give exactly the same outputs as the Hugging Face implementation.
Tiny random models are built locally, so these tests need no download.
"""
import pytest
import torch
import transformers as tf

from tiny_llm import FAMILIES, TinyLLM

V, T = 120, 64  # vocab, context


def tiny_hf_models():
    torch.manual_seed(0)
    return {
        "gpt2": tf.GPT2LMHeadModel(tf.GPT2Config(n_layer=2, n_head=4, n_embd=32, vocab_size=V, n_positions=T)),
        "llama (GQA, untied)": tf.LlamaForCausalLM(tf.LlamaConfig(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=2, intermediate_size=96,
            num_hidden_layers=2, vocab_size=V, max_position_embeddings=T, tie_word_embeddings=False,
            head_dim=16)),
        "llama (tied, MHA)": tf.LlamaForCausalLM(tf.LlamaConfig(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=4, intermediate_size=96,
            num_hidden_layers=2, vocab_size=V, max_position_embeddings=T, tie_word_embeddings=True,
            head_dim=16)),
        "qwen2 (qkv bias)": tf.Qwen2ForCausalLM(tf.Qwen2Config(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=2, intermediate_size=96,
            num_hidden_layers=2, vocab_size=V, max_position_embeddings=T, tie_word_embeddings=True)),
        "gpt_neox (parallel residual)": tf.GPTNeoXForCausalLM(tf.GPTNeoXConfig(
            hidden_size=64, num_attention_heads=4, intermediate_size=128, num_hidden_layers=2,
            vocab_size=V, max_position_embeddings=T, use_parallel_residual=True)),
        "gpt_neox (sequential)": tf.GPTNeoXForCausalLM(tf.GPTNeoXConfig(
            hidden_size=64, num_attention_heads=4, intermediate_size=128, num_hidden_layers=2,
            vocab_size=V, max_position_embeddings=T, use_parallel_residual=False)),
    }


@pytest.fixture(scope="module", params=list(tiny_hf_models()))
def pair(request, tmp_path_factory):
    hf = tiny_hf_models()[request.param].eval()
    # random init leaves some biases at 0, perturb every weight so a wrong mapping can't hide
    with torch.no_grad():
        for p in hf.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    path = tmp_path_factory.mktemp(request.param.split()[0])
    hf.save_pretrained(path)
    ours = TinyLLM.from_pretrained(str(path)).eval()
    return ours, hf


def test_every_family_is_registered():
    assert {"gpt2", "llama", "qwen2", "gpt_neox"} <= set(FAMILIES)


def test_logits_match_huggingface(pair):
    ours, hf = pair
    torch.manual_seed(1)
    x = torch.randint(0, V, (2, 20))
    with torch.no_grad():
        ours_logits, _ = ours(x, x)
        hf_logits = hf(x).logits
    assert torch.allclose(ours_logits, hf_logits, atol=1e-4), (ours_logits - hf_logits).abs().max()


def test_greedy_generation_matches_huggingface(pair):
    ours, hf = pair
    x = torch.randint(0, V, (1, 7))
    hf_out = hf.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=15, do_sample=False,
                         pad_token_id=0, eos_token_id=None)
    for use_cache in (True, False):
        assert torch.equal(ours.generate(x, 15, temperature=0, use_cache=use_cache), hf_out)


def test_cache_beyond_the_context_window(pair):
    """Past block_size the cached version must behave exactly like the sliding-window one."""
    ours, _ = pair
    x = torch.randint(0, V, (1, T - 5))
    a = ours.generate(x, 12, temperature=0, use_cache=True)
    b = ours.generate(x, 12, temperature=0, use_cache=False)
    assert torch.equal(a, b)


def test_unsupported_family_has_a_clear_error(tmp_path):
    cfg = tf.OPTConfig(hidden_size=32, num_attention_heads=2, ffn_dim=64, num_hidden_layers=1, vocab_size=V,
                       word_embed_proj_dim=32)
    tf.OPTForCausalLM(cfg).save_pretrained(tmp_path)
    with pytest.raises(NotImplementedError, match="not supported yet"):
        TinyLLM.from_pretrained(str(tmp_path))
