"""Device / precision helpers and model initialisation (scratch, resume, or GPT-2 weights)."""
import argparse
import math
import os
from contextlib import nullcontext
from dataclasses import fields

import torch

from tiny_llm.checkpoint import load_checkpoint
from tiny_llm.config import ModelConfig
from tiny_llm.model import TinyLLM
from tiny_llm.tokenizer import create_tokenizer

DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}


def resolve_device(device="auto"):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def resolve_dtype(dtype, device):
    if dtype != "auto":
        return dtype
    if torch.device(device).type == 'cuda':
        return 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    return 'float32'


def autocast_context(device, dtype):
    device_type = torch.device(device).type
    if device_type != 'cuda' or dtype == 'float32':
        return nullcontext()
    return torch.amp.autocast(device_type=device_type, dtype=DTYPES[dtype])


def setup_torch(seed):
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def init_model(init_from, model_config, tokenizer_name="", ckpt_path=None, device='cpu'):
    """
    Build the model and its tokenizer.
        init_from = "scratch"  -> random weights with `model_config` (tokenizer: `tokenizer_name` or gpt2)
        init_from = "resume"   -> weights + config + tokenizer name from `ckpt_path`
        otherwise              -> a Hugging Face model id / folder of a supported family (tiny_llm/families.py),
                                  its vocabulary extended with our special tokens
    Returns (model, tokenizer, checkpoint or None).
    """
    checkpoint = None
    if init_from == 'scratch':
        tokenizer = create_tokenizer(tokenizer_name or "gpt2")
        if model_config.vocab_size < len(tokenizer):
            model_config.vocab_size = math.ceil(len(tokenizer) / 64) * 64
            print(f"vocab_size raised to {model_config.vocab_size} to fit the tokenizer")
        model = TinyLLM(model_config)
    elif init_from == 'resume':
        assert ckpt_path and os.path.exists(ckpt_path), f"no checkpoint found at {ckpt_path}"
        model, checkpoint = load_checkpoint(ckpt_path, device)
        print(f"checkpoint loaded from {ckpt_path} :)")
        tokenizer = create_tokenizer(tokenizer_name or checkpoint.get('tokenizer', 'gpt2'))
        model = _apply_runtime_overrides(model, model_config)
    else:
        tokenizer = create_tokenizer(tokenizer_name or init_from)
        override = dict(drop_rate=model_config.drop_rate, lora_rank=model_config.lora_rank,
                        lora_alpha=model_config.lora_alpha, lora_dropout=model_config.lora_dropout)
        model = TinyLLM.from_pretrained(init_from, tokenizer, override)
        print(f"pre-trained weights loaded from {init_from} :)")

    assert len(tokenizer) <= model.config.vocab_size, \
        f"the tokenizer ({len(tokenizer)} tokens) doesn't fit in the model vocabulary ({model.config.vocab_size})"
    # crop the context length of a pre-trained model if we asked for a smaller one
    if model_config.block_size < model.config.block_size:
        model.crop_block_size(model_config.block_size)
    return model.to(device), tokenizer, checkpoint


def _apply_runtime_overrides(model, model_config):
    """
    Dropout and (newly added) LoRA layers are not part of the learned weights, so a resumed model takes
    them from the requested config. This is what lets us LoRA fine-tune one of our own checkpoints.
    """
    new_args = model.config.to_dict()
    new_args['drop_rate'] = model_config.drop_rate
    if model.config.lora_rank == 0 and model_config.lora_rank > 0:
        new_args.update(lora_rank=model_config.lora_rank, lora_alpha=model_config.lora_alpha,
                        lora_dropout=model_config.lora_dropout)
    if new_args == model.config.to_dict():
        return model

    device = next(model.parameters()).device
    new_model = TinyLLM(ModelConfig.from_dict(new_args)).to(device)
    missing, unexpected = new_model.load_state_dict(model.state_dict(), strict=False)
    assert not unexpected and all('lora_' in k for k in missing), (missing, unexpected)
    return new_model


def parse_config_args(*dataclasses, description=None, argv=None):
    """
    Expose every field of the given dataclasses as a `--name value` CLI flag.
    Returns one instance per dataclass, e.g. `train_cfg, model_cfg = parse_config_args(TrainConfig, ModelConfig)`.
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def str2bool(v):
        if v.lower() in ('1', 'true', 'yes', 'y'):
            return True
        if v.lower() in ('0', 'false', 'no', 'n'):
            return False
        raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")

    defaults = [dc() for dc in dataclasses]
    seen = set()
    for default in defaults:
        group = parser.add_argument_group(type(default).__name__)
        for f in fields(default):
            if f.name in seen:
                continue
            seen.add(f.name)
            value = getattr(default, f.name)
            arg_type = str2bool if isinstance(value, bool) else type(value)
            group.add_argument(f"--{f.name}", type=arg_type, default=value)
    args = vars(parser.parse_args(argv))
    return [type(d)(**{f.name: args[f.name] for f in fields(d)}) for d in defaults]
