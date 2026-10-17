import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from tiny_llm.attention import KVCache
from tiny_llm.block import DBlock
from tiny_llm.lora import LoraEmbedding
from tiny_llm.norm import build_norm
from tiny_llm.rope import RotaryEmbedding


class TinyLLM(nn.Module):
    """
    Tiny LLM: a decoder-only transformer. With the default config it is GPT-2; the config
    switches (see ModelConfig / tiny_llm/families.py) turn it into Llama, Qwen2, Pythia-style models.

    Args:
        config: ModelConfig (vocab_size, block_size, n_layer, n_head, n_embd, ...)

    Attributes:
        transformer: ModuleDict containing the components of the GPT model (Embeddings, Dropout, Blocks, Norm).
        lm_head: Linear layer for language modeling prediction (tied with the token embeddings by default).
        config: Model configuration.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        rotary = None
        if config.pos_encoding == "rope":  # one cos/sin table shared by every layer
            rotary = RotaryEmbedding(config.head_size, config.block_size, config.rope_theta, config.rotary_pct)
        layers = dict(
            # Note: no LoRA on embeddings. wte is usually tied with lm_head, so a LoRA delta on wte
            # would not be seen by lm_head and merging it would change the model.
            wte=LoraEmbedding(config.vocab_size, config.n_embd),  # token embeddings
            drop=nn.Dropout(config.drop_rate),
            h=nn.ModuleList([DBlock(config, rotary) for _ in range(config.n_layer)]),  # Block X N
            ln_f=build_norm(config),
        )
        if config.pos_encoding == "learned":
            layers["wpe"] = LoraEmbedding(config.block_size, config.n_embd)  # positional embeddings
        self.transformer = nn.ModuleDict(layers)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)
        # scaled init of residual projections (GPT-2 paper)
        for param_name, param in self.named_parameters():
            if param_name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        # re-initialise LoRA params (the generic init above only touched `weight`, but be explicit)
        for module in self.modules():
            if getattr(module, "rank", 0) > 0:
                module.reset_lora_parameters()

    def get_num_params(self, non_embedding=True):
        """Number of params. Position embeddings are excluded by default; token embeddings are
        counted because they are (usually) shared with lm_head."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and "wpe" in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def materialize_(self, device="cpu"):
        """
        After building the model on the meta device and assigning checkpoint weights, create what the
        checkpoint doesn't contain: LoRA matrices and rotary tables. Also restores weight tying.
        """
        for module in self.modules():
            if getattr(module, "rank", 0) > 0 and module.lora_A.is_meta:
                module.lora_A = nn.Parameter(torch.empty_like(module.lora_A, device=device))
                module.lora_B = nn.Parameter(torch.empty_like(module.lora_B, device=device))
                module.reset_lora_parameters()
            if isinstance(module, RotaryEmbedding) and module.inv_freq.is_meta:
                module.reset_buffers(device)
        if self.config.tie_embeddings:
            self.transformer.wte.weight = self.lm_head.weight
        meta = [n for n, t in list(self.named_parameters()) + list(self.named_buffers()) if t.is_meta]
        assert not meta, f"weights left uninitialised: {meta}"
        return self

    def new_kv_cache(self):
        return [KVCache() for _ in self.transformer.h]

    def forward(self, idx, targets=None, ignore_index=-1, kv_cache=None):
        """
        idx:      (B, T) token ids
        targets:  (B, T) next-token ids, positions equal to `ignore_index` are not part of the loss
        kv_cache: list of KVCache (one per layer) from `new_kv_cache()`; idx are the tokens after the cached ones
        returns:  logits (B, T, V) when targets are given, else (B, 1, V) for the last position only
        """
        device = idx.device
        B, T = idx.size()
        start_pos = kv_cache[0].length if kv_cache is not None else 0
        assert start_pos + T <= self.config.block_size, \
            f"Max sequence length is {self.config.block_size}, got {start_pos + T}"

        x = self.transformer.wte(idx)  # (B, T, n_embd)
        if "wpe" in self.transformer:
            pos = torch.arange(start_pos, start_pos + T, dtype=torch.long, device=device)
            x = x + self.transformer.wpe(pos)  # (T, n_embd) broadcast over the batch

        x = self.transformer.drop(x)
        for i, block in enumerate(self.transformer.h):
            x = block(x, kv_cache[i] if kv_cache is not None else None)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index)
        else:
            # inference: only the last position is needed
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """Model surgery: reduce the context length (e.g. load GPT-2 (1024) but use 256)."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if "wpe" in self.transformer:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
            self.transformer.wpe.num_embeddings = block_size
        for module in self.modules():  # modules() visits the shared rotary table once
            if isinstance(module, RotaryEmbedding):
                module.crop(block_size)

    @classmethod
    def from_pretrained(cls, name_or_path, tokenizer=None, override_args=None):
        """
        Load a Hugging Face checkpoint of any supported family (GPT-2, Llama, Qwen2, GPT-NeoX, ...)
        into our from-scratch implementation. See tiny_llm/pretrained.py.
        """
        from tiny_llm.pretrained import load_pretrained
        return load_pretrained(cls, name_or_path, tokenizer, override_args)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # 2D params (matmuls, embeddings) are decayed; biases and norms are not
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    @torch.no_grad()
    def stream_generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                        eos_token_id=None, banned_token_ids=None, use_cache=True):
        """
        Yield new token ids one by one, shape (B, 1).
        temperature = 0 -> greedy decoding. Generation stops after `eos_token_id` (int or list of ints).
        `banned_token_ids` can never be sampled (e.g. padding).
        use_cache: keep keys/values of past tokens (much faster). Once the context window is full we fall
        back to re-reading the last `block_size` tokens, exactly like the no-cache version.
        """
        eos_ids = set([eos_token_id] if isinstance(eos_token_id, int) else (eos_token_id or []))
        block_size = self.config.block_size
        cache = None
        for _ in range(max_new_tokens):
            if cache is not None and cache[0].length < block_size:
                logits, _ = self(idx[:, -1:], kv_cache=cache)  # only the newest token
            else:
                cache = self.new_kv_cache() if use_cache else None
                logits, _ = self(idx[:, -block_size:], kv_cache=cache)
            logits = logits[:, -1, :].float()
            if banned_token_ids:
                logits[:, list(banned_token_ids)] = -float('Inf')

            if temperature == 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum_probs > top_p
                    remove[:, 1:] = remove[:, :-1].clone()  # always keep the most likely token
                    remove[:, 0] = False
                    logits[remove.scatter(1, sorted_idx, remove)] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next
            if eos_ids and idx_next.numel() == 1 and idx_next.item() in eos_ids:
                break

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_token_id=None,
                 banned_token_ids=None, use_cache=True):
        """Return idx (B, T) extended with up to `max_new_tokens` generated tokens."""
        new_tokens = list(self.stream_generate(idx, max_new_tokens, temperature, top_k, top_p, eos_token_id,
                                               banned_token_ids, use_cache))
        return torch.cat([idx] + new_tokens, dim=1) if new_tokens else idx
