"""
LoRA: Low-Rank Adaptation of Large Language Models (https://arxiv.org/abs/2106.09685)

    Y = W*X + (B*A)*X * scale        W is frozen, only A and B (rank r <<< min(in_dim, out_dim)) are trained

With rank = 0 these layers behave exactly like nn.Linear / nn.Embedding, so the same model
is used for pre-training (no LoRA) and for fine-tuning HUGE LLMs (LoRA).
Freezing is NOT done inside the layers: call `mark_only_lora_as_trainable(model)`.
"""
import math

import torch.nn as nn
from torch.nn import functional as F


class LoraLinear(nn.Linear):
    """ Warning: 0 < rank <<< min(in_dim, out_dim), rank = 0 disables LoRA. """

    def __init__(self, in_dim, out_dim, rank=0, lora_alpha=1.0, lora_dropout=0.0, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_merged = False

        # dropout OR identity f(x) = x
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        if rank > 0:  # Y = Llm_w*X + lora_w*X = W*X + B*A*X
            self.lora_A = nn.Parameter(self.weight.new_zeros((rank, in_dim)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_dim, rank)))
            self.scale = self.lora_alpha / self.rank  # kind of a hyperparameter
            self.reset_lora_parameters()

    def reset_lora_parameters(self):
        # A random, B = 0  =>  B*A = 0 : at step 0 the model is exactly the pre-trained one
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def merge(self):
        """Fold B*A into W (no extra cost at inference time)."""
        if self.rank > 0 and not self.lora_merged:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scale
            self.lora_merged = True

    def forward(self, inputs):
        llm_w = super().forward(inputs)  # frozen pre-trained weights

        if self.rank <= 0 or self.lora_merged:
            return llm_w  # merged means B*A is already inside W

        lora_w = (self.lora_dropout(inputs) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scale
        return llm_w + lora_w


class LoraEmbedding(nn.Embedding):
    """ Warning: 0 < rank <<< min(num_embeddings, embedding_dim), rank = 0 disables LoRA. """

    def __init__(self, num_embeddings, embedding_dim, rank=0, lora_alpha=1.0, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_merged = False

        if rank > 0:
            # Y = Llm_w[x] + (B*A)^T[x]
            self.lora_A = nn.Parameter(self.weight.new_zeros((rank, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, rank)))
            self.scale = self.lora_alpha / self.rank
            self.reset_lora_parameters()

    def reset_lora_parameters(self):
        # for embeddings A = 0 and B random (as in microsoft/LoRA), B*A = 0 either way
        nn.init.zeros_(self.lora_A)
        nn.init.normal_(self.lora_B)

    def merge(self):
        if self.rank > 0 and not self.lora_merged:
            self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scale
            self.lora_merged = True

    def forward(self, inputs):
        llm_w = super().forward(inputs)  # old embeds (frozen)

        if self.rank <= 0 or self.lora_merged:
            return llm_w

        lora_emb_A = F.embedding(inputs,
                                 self.lora_A.transpose(0, 1),
                                 self.padding_idx,
                                 self.max_norm,
                                 self.norm_type,
                                 self.scale_grad_by_freq,
                                 self.sparse)  # lora_A act as a lookup table

        lora_w = (lora_emb_A @ self.lora_B.transpose(0, 1)) * self.scale
        return llm_w + lora_w


def mark_only_lora_as_trainable(model, extra_trainable=()):
    """Freeze everything except LoRA params (and params whose name contains one of `extra_trainable`)."""
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name or any(key in name for key in extra_trainable)


def merge_lora(model):
    """Merge every LoRA layer into its frozen weights (call before inference)."""
    for module in model.modules():
        if isinstance(module, (LoraLinear, LoraEmbedding)):
            module.merge()
    return model
