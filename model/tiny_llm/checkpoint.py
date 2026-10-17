import os

import torch

from tiny_llm.config import ModelConfig
from tiny_llm.model import TinyLLM

# old module names -> current names (the first version of this repo used attn_net / ff_mlp)
LEGACY_RENAMES = {'.attn_net.': '.attn.', '.ff_mlp.': '.mlp.'}
UNWANTED_PREFIX = '_orig_mod.'  # added by torch.compile


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith(UNWANTED_PREFIX):
            key = key[len(UNWANTED_PREFIX):]
        for old, new in LEGACY_RENAMES.items():
            key = key.replace(old, new)
        cleaned[key] = value
    return cleaned


def save_checkpoint(path, model, optimizer=None, model_args=None, iter_num=0, best_val_loss=float('inf'),
                    extra=None):
    model = getattr(model, '_orig_mod', model)  # unwrap torch.compile
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'model_args': model_args or model.config.to_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        **(extra or {}),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} :)")


def load_checkpoint(path, device='cpu'):
    """Return (model, checkpoint dict). The model is built from the config stored in the checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = clean_state_dict(checkpoint['model'])
    model_args = dict(checkpoint['model_args'])

    # old checkpoints had LoRA layers but no `lora_rank` in model_args: infer it from the weights
    lora_keys = [k for k in state_dict if k.endswith('lora_A')]
    if lora_keys and not model_args.get('lora_rank'):
        model_args['lora_rank'] = state_dict[lora_keys[0]].size(0)

    model = TinyLLM(ModelConfig.from_dict(model_args))
    model.load_state_dict(state_dict)
    model.to(device)
    checkpoint['model_args'] = model.config.to_dict()
    return model, checkpoint
