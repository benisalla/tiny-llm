import torch
from torch import nn
from torch.nn import functional as F

from tiny_llm.lora import LoraLinear
from tiny_llm.rope import RotaryEmbedding


class KVCache:
    """
        Keys and values of the tokens already processed by one attention layer.
        During generation each new token only computes its own q, k, v and attends to the cached ones,
        instead of recomputing the whole sequence at every step.
    """

    def __init__(self):
        self.k = self.v = None

    @property
    def length(self):
        return 0 if self.k is None else self.k.size(2)

    def update(self, k, v):
        self.k = k if self.k is None else torch.cat((self.k, k), dim=2)
        self.v = v if self.v is None else torch.cat((self.v, v), dim=2)
        return self.k, self.v


class CausalSelfAttention(nn.Module):
    """
        Causal Self-Attention Module
        ( why causal? because each token can only attend to itself and the tokens before it )

        Supports multi-head attention (GPT-2, Pythia) and grouped-query attention (Llama, Qwen2), where
        several query heads share the same key/value head (n_kv_head < n_head: smaller KV cache).

        Tensor          Type            Shape
        ===========================================================================
        input           float           (batch_size, seq_len, n_embd)
        ---------------------------------------------------------------------------
        output          float           (batch_size, seq_len, n_embd)
        ===========================================================================
    """

    def __init__(self, config, rotary=None):
        super().__init__()
        """
            dropout ~= lora_dropout since Pre-Trained W is frozen
            we can either introduce new dropout (lora_dropout) or use dropout
        """
        self.n_head = config.n_head
        self.n_kv_head = config.kv_heads
        self.head_dim = config.head_size
        self.drop_rate = config.drop_rate
        q_dim, kv_dim = self.n_head * self.head_dim, self.n_kv_head * self.head_dim
        self.split_sizes = [q_dim, kv_dim, kv_dim]

        lora_args = dict(rank=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout)
        # query, key, value projections for all heads, in one batch
        self.c_attn = LoraLinear(config.n_embd, q_dim + 2 * kv_dim, bias=config.use_qkv_bias, **lora_args)
        # output projection
        self.c_proj = LoraLinear(q_dim, config.n_embd, bias=config.bias, **lora_args)
        self.res_drop = nn.Dropout(config.drop_rate)

        # the model shares one rotary table between all layers; standalone layers build their own
        self.rotary = None
        if config.pos_encoding == "rope":
            self.rotary = rotary or RotaryEmbedding(self.head_dim, config.block_size, config.rope_theta,
                                                    config.rotary_pct)

    def forward(self, x, kv_cache=None):
        B, T, _ = x.size()
        start_pos = kv_cache.length if kv_cache is not None else 0

        q, k, v = self.c_attn(x).split(self.split_sizes, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv, T, hs)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv, T, hs)

        if self.rotary is not None:
            q, k = self.rotary(q, k, start_pos)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)  # (B, n_kv, start_pos + T, hs)

        if self.n_kv_head != self.n_head:  # grouped-query attention: share each kv head between query heads
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        S = k.size(2)
        if T == 1:  # one new token: it may look at everything before it
            mask, is_causal = None, False
        elif S == T:  # no cache: plain causal mask
            mask, is_causal = None, True
        else:  # new tokens on top of a cache: token i sees cached tokens + new tokens up to i
            mask, is_causal = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T), False

        # flash attention when available: softmax(q k^T / sqrt(hs)) v
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal,
                                           dropout_p=self.drop_rate if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)  # re-assemble all heads

        return self.res_drop(self.c_proj(y))
