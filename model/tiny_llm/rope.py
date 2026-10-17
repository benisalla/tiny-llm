import torch
from torch import nn


def rotate_half(x):
    """[x1, x2] -> [-x2, x1] on the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
        Rotary Position Embedding (RoPE, https://arxiv.org/abs/2104.09864).
        Instead of adding a position vector to the input, q and k are rotated by an angle that grows
        with the position, so q·k only depends on the *distance* between two tokens.
        Used by: Llama, Qwen2, SmolLM2, Pythia (partial: only the first `rotary_pct` of each head).

        Tensor          Type            Shape
        ===========================================================================
        q, k            float           (batch_size, n_heads, seq_len, head_dim)
        ===========================================================================
    """

    def __init__(self, head_dim, max_positions, theta=10000.0, rotary_pct=1.0):
        super().__init__()
        self.rot_dim = int(head_dim * rotary_pct)
        assert self.rot_dim % 2 == 0, "the rotated part of a head must have an even size"
        self.theta, self.max_positions = theta, max_positions
        self.reset_buffers()

    def reset_buffers(self, device=None):
        """(Re)compute the cos/sin tables, they are not saved in checkpoints."""
        arange = torch.arange(0, self.rot_dim, 2, dtype=torch.int64, device=device).float()
        inv_freq = 1.0 / (self.theta ** (arange / self.rot_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(self.max_positions, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # (T, rot_dim / 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, rot_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def crop(self, max_positions):
        self.max_positions = max_positions
        self.cos, self.sin = self.cos[:max_positions], self.sin[:max_positions]

    def _rotate(self, x, cos, sin):
        x_rot, x_pass = x[..., :self.rot_dim], x[..., self.rot_dim:]
        x_rot = x_rot * cos + rotate_half(x_rot) * sin
        return torch.cat((x_rot, x_pass), dim=-1)

    def forward(self, q, k, start_pos=0):
        T = q.size(-2)
        cos = self.cos[start_pos:start_pos + T].to(q.dtype)
        sin = self.sin[start_pos:start_pos + T].to(q.dtype)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)
