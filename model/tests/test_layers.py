import math

import torch
from torch import nn

from tiny_llm.attention import CausalSelfAttention
from tiny_llm.block import DBlock
from tiny_llm.norm import LayerNorm, RMSNorm
from tiny_llm.mlp import MLP

from conftest import tiny_config


def test_layer_norm_matches_pytorch():
    x = torch.randn(2, 5, 16)
    ours, ref = LayerNorm(16, bias=True), nn.LayerNorm(16)
    assert torch.allclose(ours(x), ref(x), atol=1e-6)
    assert LayerNorm(16, bias=False).bias is None


def test_shapes_are_preserved():
    cfg = tiny_config()
    x = torch.randn(3, 10, cfg.n_embd)
    for layer in (CausalSelfAttention(cfg), MLP(cfg), DBlock(cfg)):
        assert layer(x).shape == x.shape


def test_attention_is_causal():
    """Changing a future token must not change the output at earlier positions."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(tiny_config()).eval()
    x = torch.randn(1, 8, 32)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(1, 3, 32)
    y, y2 = attn(x), attn(x2)
    assert torch.allclose(y[:, :5], y2[:, :5], atol=1e-6)
    assert not torch.allclose(y[:, 5:], y2[:, 5:])


def naive_attention(attn, x):
    """Textbook attention, written independently of the module, as a reference."""
    B, T, C = x.shape
    q, k, v = attn.c_attn(x).split(attn.split_sizes, dim=2)
    q, k, v = (t.view(B, T, -1, attn.head_dim).transpose(1, 2) for t in (q, k, v))
    k = k.repeat_interleave(attn.n_head // attn.n_kv_head, dim=1)
    v = v.repeat_interleave(attn.n_head // attn.n_kv_head, dim=1)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
    att = att.masked_fill(torch.ones(T, T).tril() == 0, float("-inf")).softmax(-1)
    return attn.c_proj((att @ v).transpose(1, 2).reshape(B, T, -1))


def test_attention_matches_textbook_formula():
    torch.manual_seed(0)
    for n_kv_head in (4, 2, 1):  # multi-head, grouped-query, multi-query
        attn = CausalSelfAttention(tiny_config(n_kv_head=n_kv_head)).eval()
        x = torch.randn(2, 12, 32)
        assert torch.allclose(attn(x), naive_attention(attn, x), atol=1e-5)


def test_rms_norm():
    x = torch.randn(2, 5, 16) * 3 + 1
    y = RMSNorm(16)(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(2, 5), atol=1e-3)


def test_kv_cache_matches_full_attention():
    """Processing tokens one by one with a cache == processing the whole sequence at once."""
    from tiny_llm.attention import KVCache
    torch.manual_seed(0)
    for cfg in (tiny_config(), tiny_config(pos_encoding="rope", n_kv_head=2)):
        attn = CausalSelfAttention(cfg).eval()
        x = torch.randn(1, 10, 32)
        full = attn(x)
        cache = KVCache()
        first = attn(x[:, :6], cache)  # prefill
        steps = [attn(x[:, t:t + 1], cache) for t in range(6, 8)]  # one token at a time
        chunk = attn(x[:, 8:], cache)  # several new tokens on top of a cache
        assert torch.allclose(torch.cat([first, *steps, chunk], dim=1), full, atol=1e-5)
