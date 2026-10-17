"""
Fine-tune TinyLLM for (context-)question answering, optionally with LoRA.
Data: data_dir/train.json and data_dir/val.json, lists of {"context", "question", "answer"}.

    # LoRA fine-tuning of GPT-2 small (it is better to fine tune a pre-trained LLM)
    python scripts/finetune.py --init_from gpt2 --data_dir ./data/qa --lora_rank 8 \
        --learning_rate 3e-4 --drop_rate 0.1 --max_iters 2000 --warmup_iters 100 --lr_decay_iters 2000

    # LoRA fine-tuning of other small LLMs (Llama / Qwen2 / Pythia families)
    python scripts/finetune.py --init_from HuggingFaceTB/SmolLM2-135M --data_dir ./data/qa --lora_rank 8 ...

    # full fine-tuning of one of our own pre-trained checkpoints
    python scripts/finetune.py --init_from resume --resume_from ./checkpoints/ckpt.pt --learning_rate 3e-5
"""
import os
import sys

from tiny_llm.config import ModelConfig, TrainConfig
from tiny_llm.data import BatchLoader
from tiny_llm.lora import mark_only_lora_as_trainable
from tiny_llm.runtime import autocast_context, init_model, parse_config_args, resolve_device, resolve_dtype, setup_torch
from tiny_llm.training import build_optimizer, train

FINETUNE_DEFAULTS = dict(init_from="gpt2", learning_rate=3e-5, warmup_iters=100, lr_decay_iters=5000,
                         max_iters=5000, min_lr=3e-6, grad_accum_steps=4, eval_interval=100, eval_iters=50,
                         log_interval=10, ckpt_name="finetuned.pt")


def main(argv=None):
    # fine-tuning has its own defaults, still overridable from the command line
    cfg, model_cfg = parse_config_args(TrainConfig, ModelConfig, description=__doc__,
                                       argv=_with_defaults(argv, FINETUNE_DEFAULTS))
    setup_torch(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    ctx = autocast_context(device, dtype)

    ckpt_in = cfg.resume_from or os.path.join(cfg.out_dir, cfg.ckpt_name)
    model, tokenizer, checkpoint = init_model(cfg.init_from, model_cfg, cfg.tokenizer, ckpt_in, device)

    if model.config.lora_rank > 0:
        # LoRA: freeze the LLM, train only A and B (+ the tied embeddings so new special tokens are learned)
        mark_only_lora_as_trainable(model, extra_trainable=("wte",) if cfg.train_embeddings else ())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train / 1e6:.2f}M / {n_total / 1e6:.2f}M | device: {device} | dtype: {dtype}")

    # create dataloader for both train and val datasets
    loader_args = dict(tokenizer=tokenizer, toks_in_batch=cfg.toks_per_batch, device=device,
                       max_len=min(cfg.max_seq_len, model.config.block_size + 1),
                       answer_only=cfg.loss_on_answer_only, seed=cfg.seed)
    train_loader = BatchLoader(cfg.data_dir, "train", **loader_args)
    val_loader = BatchLoader(cfg.data_dir, "val", **loader_args)
    print(f"{len(train_loader)} train batches, {len(val_loader)} val batches")

    optimizer = build_optimizer(model, cfg, device)
    iter_num, best_val_loss = 0, float('inf')
    if checkpoint is not None and checkpoint.get('stage') == 'finetune':
        # resuming an interrupted fine-tuning run
        optimizer.load_state_dict(checkpoint['optimizer'])
        iter_num, best_val_loss = checkpoint['iter_num'], checkpoint['best_val_loss']
    checkpoint = None

    get_batch_fns = {"train": lambda: next(train_loader), "val": lambda: next(val_loader)}
    best = train(model, optimizer, cfg, get_batch_fns, device, ctx, dtype, iter_num, best_val_loss,
                 ckpt_extra={'stage': 'finetune', 'tokenizer': tokenizer.name_or_path})
    print(f"done, best val loss {best:.4f}")


def _with_defaults(argv, defaults):
    """Prepend `--key value` defaults so that flags given by the user (which come later) win."""
    argv = list(sys.argv[1:] if argv is None else argv)
    prefix = []
    for key, value in defaults.items():
        prefix += [f"--{key}", str(value)]
    return prefix + argv


if __name__ == "__main__":
    main()
