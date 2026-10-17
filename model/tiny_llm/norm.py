import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """
        Layer Normalization with an optional bias (PyTorch's nn.LayerNorm doesn't support bias=False).
        (different than batch normalization and better in seq problems)   Used by: GPT-2, Pythia.

        y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

        Tensor          Type            Shape
        ===========================================================================
        input           float           (batch_size, seq_len, n_embd)
        ---------------------------------------------------------------------------
        output          float           (batch_size, seq_len, n_embd)
        ===========================================================================
    """

    def __init__(self, ndim, bias, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    """
        Root Mean Square normalization: no mean subtraction and no bias, a bit cheaper than LayerNorm
        and works as well.   Used by: Llama, Qwen2, SmolLM2, TinyLlama.

        y = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()  # normalise in float32 for stability under mixed precision
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def build_norm(config):
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.n_embd, eps=config.norm_eps)
    return LayerNorm(config.n_embd, bias=config.bias, eps=config.norm_eps)
