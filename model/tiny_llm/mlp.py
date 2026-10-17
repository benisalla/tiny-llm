from torch import nn

from tiny_llm.lora import LoraLinear

ACTIVATIONS = {
    "gelu_tanh": lambda: nn.GELU(approximate="tanh"),  # GPT-2 ("gelu_new")
    "gelu": nn.GELU,  # Pythia
    "silu": nn.SiLU,  # Llama, Qwen2 (inside SwiGLU)
    "relu": nn.ReLU,
}


class MLP(nn.Module):
    """
        Multi-Layer Perceptron (MLP): computation layer.
        Non-linearity = complex mappings + feature extraction  + ... + processing data (reasoning)

            classic (GPT-2, Pythia):   c_proj( act(c_fc(x)) )
            gated   (Llama, Qwen2):    c_proj( act(c_gate(x)) * c_fc(x) )      a.k.a. SwiGLU with act = SiLU

        Tensor          Type            Shape
        ===========================================================================
        input           float           (batch_size, seq_len, n_embd)
        ---------------------------------------------------------------------------
        output          float           (batch_size, seq_len, n_embd)
        ===========================================================================

    """

    def __init__(self, config):
        super().__init__()
        lora_args = dict(rank=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout)
        hidden = config.hidden_size
        self.c_fc = LoraLinear(config.n_embd, hidden, bias=config.bias, **lora_args)  # "up" projection
        self.c_gate = LoraLinear(config.n_embd, hidden, bias=config.bias, **lora_args) if config.gated_mlp else None
        self.act = ACTIVATIONS[config.activation]()
        self.c_proj = LoraLinear(hidden, config.n_embd, bias=config.bias, **lora_args)  # "down" projection
        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, x):
        if self.c_gate is not None:
            h = self.act(self.c_gate(x)) * self.c_fc(x)
        else:
            h = self.act(self.c_fc(x))
        return self.dropout(self.c_proj(h))
