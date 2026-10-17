"""
Pre-train TinyLLM on a flat token stream (data_dir/train.bin, data_dir/val.bin).

    python scripts/train.py --data_dir ./data/openwebtext --init_from scratch
    python scripts/train.py --init_from resume                  # continue from out_dir/ckpt_name
    python scripts/train.py --init_from HuggingFaceTB/SmolLM2-135M   # continue pre-training another small LLM
    python scripts/train.py --n_layer 4 --n_head 4 --n_embd 256 --block_size 256 --batch_size 8   # small run
"""
import os

from tiny_llm.config import ModelConfig, TrainConfig
from tiny_llm.data import get_batch, load_bin
from tiny_llm.runtime import autocast_context, init_model, parse_config_args, resolve_device, resolve_dtype, setup_torch
from tiny_llm.training import build_optimizer, train


def main(argv=None):
    cfg, model_cfg = parse_config_args(TrainConfig, ModelConfig, description=__doc__, argv=argv)
    setup_torch(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    ctx = autocast_context(device, dtype)

    # connect to dataset
    train_data = load_bin(cfg.data_dir, "train")
    val_data = load_bin(cfg.data_dir, "val")

    ckpt_path = cfg.resume_from or os.path.join(cfg.out_dir, cfg.ckpt_name)
    model, tokenizer, checkpoint = init_model(cfg.init_from, model_cfg, cfg.tokenizer, ckpt_path, device)
    block_size = model.config.block_size
    print(f"number of parameters: {model.get_num_params() / 1e6:.2f}M | device: {device} | dtype: {dtype}")

    optimizer = build_optimizer(model, cfg, device)
    iter_num, best_val_loss = 0, float('inf')
    if checkpoint is not None:
        if checkpoint.get('optimizer'):
            optimizer.load_state_dict(checkpoint['optimizer'])
        iter_num, best_val_loss = checkpoint['iter_num'], checkpoint['best_val_loss']
        checkpoint = None  # free memory

    get_batch_fns = {
        "train": lambda: get_batch(train_data, cfg.batch_size, block_size, device),
        "val": lambda: get_batch(val_data, cfg.batch_size, block_size, device),
    }
    best = train(model, optimizer, cfg, get_batch_fns, device, ctx, dtype, iter_num, best_val_loss,
                 ckpt_extra={'stage': 'pretrain', 'tokenizer': tokenizer.name_or_path})
    print(f"done, best val loss {best:.4f}")


if __name__ == "__main__":
    main()
