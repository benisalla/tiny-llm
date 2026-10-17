import torch
from torch import nn

from tiny_llm import TinyLLM, LoraEmbedding, LoraLinear, mark_only_lora_as_trainable, merge_lora

from conftest import tiny_config


def test_rank_zero_is_a_plain_linear():
    torch.manual_seed(0)
    lora, ref = LoraLinear(8, 4, rank=0), nn.Linear(8, 4)
    ref.load_state_dict(lora.state_dict())
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), ref(x))
    assert not hasattr(lora, "lora_A")
    assert lora.weight.requires_grad  # layers don't freeze themselves (pre-training must train W)


def test_lora_starts_as_identity_delta():
    """B = 0 at init, so the LoRA model is exactly the base model before any training."""
    torch.manual_seed(0)
    lora = LoraLinear(8, 4, rank=2, lora_alpha=4)
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), nn.functional.linear(x, lora.weight, lora.bias))


def test_linear_merge_is_equivalent():
    torch.manual_seed(0)
    lora = LoraLinear(8, 4, rank=2, lora_alpha=4).eval()
    nn.init.normal_(lora.lora_B)  # pretend we trained
    x = torch.randn(3, 8)
    before = lora(x)
    lora.merge()
    assert lora.lora_merged
    assert torch.allclose(before, lora(x), atol=1e-5)
    lora.merge()  # merging twice is a no-op
    assert torch.allclose(before, lora(x), atol=1e-5)


def test_embedding_merge_is_equivalent():
    torch.manual_seed(0)
    emb = LoraEmbedding(10, 6, rank=2, lora_alpha=2)
    nn.init.normal_(emb.lora_A)
    idx = torch.tensor([[1, 5, 9]])
    before = emb(idx)
    assert not torch.allclose(before, nn.functional.embedding(idx, emb.weight))
    emb.merge()
    assert torch.allclose(before, emb(idx), atol=1e-5)


def test_only_lora_params_get_gradients():
    torch.manual_seed(0)
    model = TinyLLM(tiny_config(vocab_size=97, lora_rank=4))
    mark_only_lora_as_trainable(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)

    x = torch.randint(0, 97, (2, 10))
    _, loss = model(x, x)
    loss.backward()
    for n, p in model.named_parameters():
        assert (p.grad is not None) == ("lora_" in n), n


def test_extra_trainable_keeps_tied_embeddings():
    model = TinyLLM(tiny_config(vocab_size=97, lora_rank=4))
    mark_only_lora_as_trainable(model, extra_trainable=("wte",))
    assert model.transformer.wte.weight.requires_grad
    assert model.lm_head.weight.requires_grad  # same tensor
    assert not model.transformer.h[0].attn.c_attn.weight.requires_grad


def test_merge_lora_on_whole_model():
    torch.manual_seed(0)
    model = TinyLLM(tiny_config(vocab_size=97, lora_rank=4)).eval()
    for m in model.modules():
        if isinstance(m, LoraLinear) and m.rank > 0:
            nn.init.normal_(m.lora_B, std=0.02)
    x = torch.randint(0, 97, (1, 12))
    before, _ = model(x, x)
    merge_lora(model)
    after, _ = model(x, x)
    assert torch.allclose(before, after, atol=1e-4)
