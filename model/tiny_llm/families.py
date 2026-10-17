"""
Model families: what changes for each type of small LLM.

Every model below is the same decoder-only transformer as GPT-2 (embeddings -> N x [norm, attention,
norm, MLP] -> norm -> lm_head). They only differ by a handful of design choices, which are the fields
of `ModelConfig`. Each family here says:

    1. `changes`    : which choices differ from GPT-2 (human readable, also printed in the README table)
    2. `to_config`  : how to read them from a Hugging Face config
    3. `convert`    : how its weight names map onto ours (our names follow GPT-2)

To support a new family: write these three things and call `register(...)`. If it needs a design choice
that doesn't exist yet (e.g. Gemma scales the embeddings by sqrt(n_embd)), add a field to ModelConfig and
use it in the layer.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch


@dataclass(frozen=True)
class Family:
    model_type: str  # Hugging Face `config.model_type`
    examples: Tuple[str, ...]  # small checkpoints on the Hugging Face Hub
    changes: Dict[str, str]  # ModelConfig field -> what this family uses (vs GPT-2)
    to_config: Callable  # hf_config -> dict of ModelConfig arguments
    convert: Callable  # (hf_state_dict, ModelConfig) -> our state_dict


FAMILIES: Dict[str, Family] = {}


def register(family):
    FAMILIES[family.model_type] = family
    return family


def get_family(model_type):
    if model_type not in FAMILIES:
        raise NotImplementedError(f"model type {model_type!r} is not supported yet. "
                                  f"Supported: {sorted(FAMILIES)}. See tiny_llm/families.py to add one.")
    return FAMILIES[model_type]


# --------------------------------------------------------------------------------------------- helpers

class _Keys:
    """Pops Hugging Face weights by name so we can check at the end that every weight was used."""

    IGNORED = ('.attn.bias', '.attn.masked_bias', 'rotary_emb.inv_freq', '.attention.bias', '.attention.masked_bias')

    def __init__(self, state_dict):
        self.sd = dict(state_dict)

    def pop(self, key):
        return self.sd.pop(key)

    def pop_optional(self, key):
        return self.sd.pop(key, None)

    def check_all_used(self):
        left = [k for k in self.sd if not k.endswith(self.IGNORED)]
        if left:
            raise ValueError(f"these weights were not converted (unsupported variant?): {left[:10]}")


def _rope(c):
    """RoPE settings, for both transformers v4 (rope_theta, rope_scaling) and v5 (rope_parameters)."""
    params = dict(getattr(c, "rope_parameters", None) or getattr(c, "rope_scaling", None) or {})
    theta = params.get("rope_theta", getattr(c, "rope_theta", getattr(c, "rotary_emb_base", 10000.0)))
    pct = params.get("partial_rotary_factor", getattr(c, "partial_rotary_factor", getattr(c, "rotary_pct", 1.0)))
    rope_type = params.get("rope_type", params.get("type", "default"))
    if rope_type != "default":
        raise NotImplementedError(f"RoPE scaling {rope_type!r} is not implemented (only plain RoPE)")
    return float(theta), float(pct)


def _head_dim(c, n_embd, n_head):
    head_dim = getattr(c, "head_dim", None) or n_embd // n_head
    return 0 if head_dim == n_embd // n_head else head_dim


# --------------------------------------------------------------------------------------------- GPT-2

def _gpt2_config(c):
    act = {"gelu_new": "gelu_tanh", "gelu_pytorch_tanh": "gelu_tanh", "gelu": "gelu", "relu": "relu"}
    return dict(n_layer=c.n_layer, n_head=c.n_head, n_embd=c.n_embd, block_size=c.n_positions,
                vocab_size=c.vocab_size, bias=True, norm_eps=c.layer_norm_epsilon,
                activation=act[c.activation_function], intermediate_size=c.n_inner or 0)


def _gpt2_convert(sd_hf, cfg):
    keys = _Keys(sd_hf)
    sd = {}
    # HF GPT-2 uses Conv1D, i.e. weights are stored transposed compared to nn.Linear
    transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
    for k in list(keys.sd):
        if k.endswith(_Keys.IGNORED):
            continue
        name = k if k.startswith(('transformer.', 'lm_head.')) else f"transformer.{k}"  # some saves drop the prefix
        value = keys.pop(k)
        sd[name] = value.t() if k.endswith(transposed) else value
    sd.setdefault("lm_head.weight", sd["transformer.wte.weight"])  # tied, some saves only keep wte
    keys.check_all_used()
    return sd


register(Family(
    model_type="gpt2",
    examples=("gpt2", "distilgpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"),
    changes={},  # the reference
    to_config=_gpt2_config,
    convert=_gpt2_convert,
))


# --------------------------------------------------------------------------------------------- Llama / Qwen2

def _llama_config(c, qkv_bias=False):
    if getattr(c, "mlp_bias", False) or getattr(c, "attention_bias", False):
        raise NotImplementedError("Llama variants with attention_bias / mlp_bias are not supported")
    theta, pct = _rope(c)
    return dict(n_layer=c.num_hidden_layers, n_head=c.num_attention_heads, n_embd=c.hidden_size,
                n_kv_head=c.num_key_value_heads, head_dim=_head_dim(c, c.hidden_size, c.num_attention_heads),
                block_size=c.max_position_embeddings, vocab_size=c.vocab_size,
                bias=False, qkv_bias=qkv_bias, norm_type="rmsnorm", norm_eps=c.rms_norm_eps,
                pos_encoding="rope", rope_theta=theta, rotary_pct=pct,
                activation=c.hidden_act, gated_mlp=True, intermediate_size=c.intermediate_size,
                tie_embeddings=bool(c.tie_word_embeddings))


def _qwen2_config(c):
    if getattr(c, "use_sliding_window", False):
        raise NotImplementedError("Qwen2 sliding-window attention is not supported")
    return _llama_config(c, qkv_bias=True)


def _llama_convert(sd_hf, cfg):
    keys = _Keys(sd_hf)
    sd = {"transformer.wte.weight": keys.pop("model.embed_tokens.weight"),
          "transformer.ln_f.weight": keys.pop("model.norm.weight")}
    for i in range(cfg.n_layer):
        hf, ours = f"model.layers.{i}.", f"transformer.h.{i}."
        sd[ours + "ln_1.weight"] = keys.pop(hf + "input_layernorm.weight")
        sd[ours + "ln_2.weight"] = keys.pop(hf + "post_attention_layernorm.weight")
        # separate q, k, v projections -> one fused c_attn
        sd[ours + "attn.c_attn.weight"] = torch.cat([keys.pop(hf + f"self_attn.{p}_proj.weight") for p in "qkv"])
        if cfg.use_qkv_bias:
            sd[ours + "attn.c_attn.bias"] = torch.cat([keys.pop(hf + f"self_attn.{p}_proj.bias") for p in "qkv"])
        sd[ours + "attn.c_proj.weight"] = keys.pop(hf + "self_attn.o_proj.weight")
        sd[ours + "mlp.c_gate.weight"] = keys.pop(hf + "mlp.gate_proj.weight")
        sd[ours + "mlp.c_fc.weight"] = keys.pop(hf + "mlp.up_proj.weight")
        sd[ours + "mlp.c_proj.weight"] = keys.pop(hf + "mlp.down_proj.weight")
    lm_head = keys.pop_optional("lm_head.weight")
    sd["lm_head.weight"] = sd["transformer.wte.weight"] if cfg.tie_embeddings or lm_head is None else lm_head
    keys.check_all_used()
    return sd


LLAMA_CHANGES = {
    "norm_type": "rmsnorm (no bias)",
    "pos_encoding": "rope (no position embedding matrix)",
    "activation / gated_mlp": "silu, gated (SwiGLU)",
    "n_kv_head": "grouped-query attention",
    "bias": "False everywhere",
    "tie_embeddings": "depends on the checkpoint",
}

register(Family(
    model_type="llama",
    examples=("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-135M-Instruct",
              "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    changes=LLAMA_CHANGES,
    to_config=_llama_config,
    convert=_llama_convert,
))

register(Family(
    model_type="qwen2",
    examples=("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    changes={**LLAMA_CHANGES, "bias": "False, except q/k/v projections (qkv_bias=True)"},
    to_config=_qwen2_config,
    convert=_llama_convert,
))


# --------------------------------------------------------------------------------------------- GPT-NeoX (Pythia)

def _neox_config(c):
    theta, pct = _rope(c)
    return dict(n_layer=c.num_hidden_layers, n_head=c.num_attention_heads, n_embd=c.hidden_size,
                block_size=c.max_position_embeddings, vocab_size=c.vocab_size, bias=True,
                norm_eps=c.layer_norm_eps, pos_encoding="rope", rope_theta=theta, rotary_pct=pct,
                activation={"gelu": "gelu", "gelu_new": "gelu_tanh", "relu": "relu"}[c.hidden_act],
                intermediate_size=c.intermediate_size, parallel_residual=bool(c.use_parallel_residual),
                tie_embeddings=bool(c.tie_word_embeddings))


def _neox_convert(sd_hf, cfg):
    keys = _Keys(sd_hf)
    nh, hd = cfg.n_head, cfg.head_size

    def unfuse(t):
        # NeoX packs qkv per head: [h0q h0k h0v h1q ...] -> ours: [all q | all k | all v]
        t = t.view(nh, 3, hd, *t.shape[1:])
        return torch.cat([t[:, j].reshape(nh * hd, *t.shape[3:]) for j in range(3)])

    sd = {"transformer.wte.weight": keys.pop("gpt_neox.embed_in.weight")}
    for p in ("weight", "bias"):
        sd[f"transformer.ln_f.{p}"] = keys.pop(f"gpt_neox.final_layer_norm.{p}")
    for i in range(cfg.n_layer):
        hf, ours = f"gpt_neox.layers.{i}.", f"transformer.h.{i}."
        for p in ("weight", "bias"):
            sd[ours + f"ln_1.{p}"] = keys.pop(hf + f"input_layernorm.{p}")
            sd[ours + f"ln_2.{p}"] = keys.pop(hf + f"post_attention_layernorm.{p}")
            sd[ours + f"attn.c_attn.{p}"] = unfuse(keys.pop(hf + f"attention.query_key_value.{p}"))
            sd[ours + f"attn.c_proj.{p}"] = keys.pop(hf + f"attention.dense.{p}")
            sd[ours + f"mlp.c_fc.{p}"] = keys.pop(hf + f"mlp.dense_h_to_4h.{p}")
            sd[ours + f"mlp.c_proj.{p}"] = keys.pop(hf + f"mlp.dense_4h_to_h.{p}")
    lm_head = keys.pop_optional("embed_out.weight")  # transformers v4 name
    lm_head = keys.pop("lm_head.weight") if lm_head is None else lm_head  # transformers v5 name
    sd["lm_head.weight"] = sd["transformer.wte.weight"] if cfg.tie_embeddings else lm_head
    keys.check_all_used()
    return sd


register(Family(
    model_type="gpt_neox",
    examples=("EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"),
    changes={
        "pos_encoding / rotary_pct": "rope on 25% of each head",
        "parallel_residual": "True: attention and MLP read the same input",
        "activation": "gelu (exact)",
        "tie_embeddings": "False (separate output matrix)",
    },
    to_config=_neox_config,
    convert=_neox_convert,
))


# --------------------------------------------------------------------------------------------- summary

def markdown_table():
    """What changes for each family, as a markdown table (scripts/list_families.py, README)."""
    rows = ["| family (`model_type`) | what changes vs GPT-2 | small checkpoints |", "| --- | --- | --- |"]
    for f in FAMILIES.values():
        changes = "<br>".join(f"`{k}`: {v}" for k, v in f.changes.items()) or "reference architecture"
        rows.append(f"| `{f.model_type}` | {changes} | {', '.join(f'`{e}`' for e in f.examples)} |")
    return "\n".join(rows)
