"""Load Hugging Face checkpoints of any supported family (see tiny_llm/families.py) into TinyLLM."""
import math

import torch

from tiny_llm.config import ModelConfig
from tiny_llm.families import get_family
from tiny_llm.tokenizer import SPECIAL_TOKENS

ALLOWED_OVERRIDES = {'drop_rate', 'lora_rank', 'lora_alpha', 'lora_dropout'}


def load_pretrained(cls, name_or_path, tokenizer=None, override_args=None):
    """
    name_or_path: Hugging Face model id ("gpt2", "HuggingFaceTB/SmolLM2-135M", ...) or a local folder.
    tokenizer:    if it contains our special tokens (tiny_llm.tokenizer.create_tokenizer), their embeddings
                  are created: the matrix is extended if needed (padded to a multiple of 8) and the new
                  rows are sampled around the existing embeddings.
    override_args: may change drop_rate and LoRA settings (lora_rank, lora_alpha, lora_dropout).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    override_args = override_args or {}
    assert set(override_args) <= ALLOWED_OVERRIDES, f"can only override {ALLOWED_OVERRIDES}"

    hf_config = AutoConfig.from_pretrained(name_or_path)
    family = get_family(hf_config.model_type)
    config_args = family.to_config(hf_config)
    config_args.update(override_args)

    model_hf = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=torch.float32)
    base_cfg = ModelConfig(**config_args)
    sd = family.convert(model_hf.state_dict(), base_cfg)
    del model_hf

    # ---- vocabulary: make room for our special tokens
    hf_rows = sd["transformer.wte.weight"].size(0)
    new_ids = _new_token_ids(tokenizer)
    if tokenizer is not None and len(tokenizer) > hf_rows:
        config_args["vocab_size"] = math.ceil(len(tokenizer) / 8) * 8  # multiple of 8: faster matmuls

    tied = config_args.get("tie_embeddings", True)
    with torch.no_grad():
        for name in ("transformer.wte.weight",) if tied else ("transformer.wte.weight", "lm_head.weight"):
            sd[name] = _resize_and_init(sd[name], config_args["vocab_size"], new_ids)
        if tied:
            sd["lm_head.weight"] = sd["transformer.wte.weight"]

    # build on the meta device (no memory, no random init) and plug the converted tensors in directly:
    # peak memory ~ one copy of the weights instead of three
    with torch.device("meta"):
        model = cls(ModelConfig(**config_args))
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, f"unexpected weights: {unexpected}"
    assert all('lora_' in k for k in missing), f"weights missing from the checkpoint: {missing}"
    return model.materialize_()


def _new_token_ids(tokenizer):
    """Ids of our special tokens: they have no trained embedding (new rows or unused spare rows)."""
    if tokenizer is None:
        return []
    added = tokenizer.get_added_vocab()  # unlike convert_tokens_to_ids, never falls back to the unk token
    return sorted(added[t] for t in SPECIAL_TOKENS if t in added)


def _resize_and_init(weight, vocab_size, new_ids):
    """Pad `weight` to `vocab_size` rows; rows of new tokens ~ N(mean, 1e-5 * cov) of the trained rows."""
    if weight.size(0) < vocab_size:
        pad = weight.new_zeros(vocab_size - weight.size(0), weight.size(1))
        weight = torch.cat([weight, pad])
    if new_ids:
        trained = weight[:min(new_ids)].float()  # the original tokenizer's rows
        avg = trained.mean(dim=0)
        centered = trained - avg
        cov = (centered.T @ centered) / (trained.size(0) - 1)
        cov = cov + 1e-6 * torch.eye(cov.size(0))  # keep it positive definite
        dist = torch.distributions.MultivariateNormal(avg, covariance_matrix=1e-5 * cov)
        weight[new_ids] = dist.sample((len(new_ids),)).to(weight.dtype)
    return weight
