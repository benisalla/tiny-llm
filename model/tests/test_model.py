import math

import pytest
import torch

from tiny_llm import TinyLLM

from conftest import tiny_config


def test_forward_shapes(tiny_model):
    x = torch.randint(0, 97, (2, 16))
    logits, loss = tiny_model(x, x)
    assert logits.shape == (2, 16, 97) and loss.ndim == 0
    logits, loss = tiny_model(x)
    assert logits.shape == (2, 1, 97) and loss is None


def test_initial_loss_is_close_to_uniform(tiny_model):
    """A freshly initialised model should be ~uniform over the vocabulary: loss ~ ln(V)."""
    x = torch.randint(0, 97, (4, 32))
    y = torch.randint(0, 97, (4, 32))
    _, loss = tiny_model(x, y)
    assert abs(loss.item() - math.log(97)) < 0.3


def test_ignore_index_is_ignored(tiny_model):
    x = torch.randint(0, 97, (1, 8))
    y = torch.randint(0, 97, (1, 8))
    y_masked = y.clone()
    y_masked[:, 4:] = -1
    _, loss_masked = tiny_model(x, y_masked)
    logits, _ = tiny_model(x, y)
    expected = torch.nn.functional.cross_entropy(logits[0, :4], y[0, :4])
    assert torch.allclose(loss_masked, expected, atol=1e-6)


def test_weights_are_tied(tiny_model):
    assert tiny_model.lm_head.weight is tiny_model.transformer.wte.weight


def test_too_long_sequence_raises(tiny_model):
    with pytest.raises(AssertionError):
        tiny_model(torch.zeros(1, 65, dtype=torch.long))


def test_generate_length_and_greedy_determinism(tiny_model):
    x = torch.randint(0, 97, (1, 5))
    out1 = tiny_model.generate(x, 10, temperature=0)
    out2 = tiny_model.generate(x, 10, temperature=0)
    assert out1.shape == (1, 15)
    assert torch.equal(out1, out2)
    assert torch.equal(out1[:, :5], x)


def test_generate_beyond_block_size(tiny_model):
    out = tiny_model.generate(torch.zeros(1, 60, dtype=torch.long), 20)
    assert out.shape == (1, 80)


def test_generate_stops_at_eos(tiny_model):
    x = torch.randint(0, 97, (1, 5))
    first = tiny_model.generate(x, 1, temperature=0)[0, -1].item()
    out = tiny_model.generate(x, 10, temperature=0, eos_token_id=first)
    assert out.shape == (1, 6)


def test_top_k_one_equals_greedy(tiny_model):
    x = torch.randint(0, 97, (1, 5))
    assert torch.equal(tiny_model.generate(x, 8, temperature=1.0, top_k=1), tiny_model.generate(x, 8, temperature=0))


def test_tiny_top_p_equals_greedy(tiny_model):
    x = torch.randint(0, 97, (1, 5))
    assert torch.equal(tiny_model.generate(x, 8, temperature=1.0, top_p=1e-6),
                       tiny_model.generate(x, 8, temperature=0))


def test_banned_tokens_are_never_sampled(tiny_model):
    torch.manual_seed(0)
    x = torch.randint(0, 97, (1, 5))
    banned = list(range(0, 90))
    out = tiny_model.generate(x, 30, temperature=2.0, banned_token_ids=banned)
    assert (out[0, 5:] >= 90).all()


def test_crop_block_size(tiny_model):
    tiny_model.crop_block_size(16)
    assert tiny_model.config.block_size == 16
    assert tiny_model.transformer.wpe.weight.shape[0] == 16
    tiny_model(torch.zeros(1, 16, dtype=torch.long))


def test_can_overfit_a_single_batch():
    """Sanity check that gradients flow everywhere: the model must memorise one batch."""
    torch.manual_seed(0)
    model = TinyLLM(tiny_config(vocab_size=97))
    opt = model.configure_optimizers(0.0, 3e-3, (0.9, 0.95), "cpu")
    x = torch.randint(0, 97, (4, 16))
    y = torch.roll(x, -1, dims=1)
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_all_params_trainable_without_lora():
    model = TinyLLM(tiny_config(vocab_size=97))
    assert all(p.requires_grad for p in model.parameters())
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == sum(p.numel() for p in model.parameters())
