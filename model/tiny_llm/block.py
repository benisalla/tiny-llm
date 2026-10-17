from torch import nn

from tiny_llm.attention import CausalSelfAttention
from tiny_llm.mlp import MLP
from tiny_llm.norm import build_norm


class DBlock(nn.Module):
    """
        Transformer (Decoder) Block: communication layer (att mechanism) + computation layer (feedforward).
        Pre-normalisation, as in GPT-2 and every model after it.

            sequential (default):        x = x + attn(ln_1(x));  x = x + mlp(ln_2(x))
            parallel   (Pythia/NeoX):    x = x + attn(ln_1(x)) + mlp(ln_2(x))

        Tensor          Type            Shape
        ===========================================================================
        input           float           (batch_size, seq_len, n_embd)
        ---------------------------------------------------------------------------
        output          float           (batch_size, seq_len, n_embd)
        ===========================================================================
    """

    def __init__(self, config, rotary=None):
        super().__init__()
        # names match Hugging Face GPT-2 (other families are renamed in tiny_llm/families.py)
        self.ln_1 = build_norm(config)
        self.attn = CausalSelfAttention(config, rotary)
        self.ln_2 = build_norm(config)
        self.mlp = MLP(config)
        self.parallel_residual = config.parallel_residual

    def forward(self, x, kv_cache=None):
        attn_out = self.attn(self.ln_1(x), kv_cache)
        if self.parallel_residual:
            return x + attn_out + self.mlp(self.ln_2(x))
        x = x + attn_out
        return x + self.mlp(self.ln_2(x))
