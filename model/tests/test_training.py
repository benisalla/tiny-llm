import os
import sys

import pytest
import torch

from tiny_llm import TinyLLM, TrainConfig, load_checkpoint, save_checkpoint
from tiny_llm.checkpoint import clean_state_dict
from tiny_llm.runtime import parse_config_args
from tiny_llm.config import ModelConfig
from tiny_llm.training import get_lr

from conftest import tiny_config

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))


def test_lr_schedule():
    cfg = TrainConfig(learning_rate=1.0, min_lr=0.1, warmup_iters=10, lr_decay_iters=110)
    assert 0 < get_lr(0, cfg) < get_lr(5, cfg) < get_lr(9, cfg) <= 1.0
    assert get_lr(10, cfg) == pytest.approx(1.0)
    assert get_lr(60, cfg) == pytest.approx(0.55)  # halfway through the cosine
    assert get_lr(110, cfg) == pytest.approx(0.1)
    assert get_lr(10_000, cfg) == 0.1


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = TinyLLM(tiny_config(vocab_size=97, lora_rank=2))
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), model, opt, iter_num=7, best_val_loss=1.5, extra={"stage": "finetune"})

    loaded, ckpt = load_checkpoint(str(path))
    assert ckpt["iter_num"] == 7 and ckpt["best_val_loss"] == 1.5 and ckpt["stage"] == "finetune"
    assert loaded.config == model.config
    x = torch.randint(0, 97, (1, 10))
    assert torch.allclose(model.eval()(x)[0], loaded.eval()(x)[0])


def test_legacy_checkpoint_keys_are_renamed():
    sd = {"_orig_mod.transformer.h.0.attn_net.c_attn.weight": 1, "transformer.h.0.ff_mlp.c_fc.bias": 2}
    assert clean_state_dict(sd) == {"transformer.h.0.attn.c_attn.weight": 1, "transformer.h.0.mlp.c_fc.bias": 2}


def test_lora_rank_is_inferred_for_old_checkpoints(tmp_path):
    model = TinyLLM(tiny_config(vocab_size=97, lora_rank=3))
    args = model.config.to_dict()
    del args["lora_rank"]  # old checkpoints didn't store it
    path = tmp_path / "old.pt"
    torch.save({"model": model.state_dict(), "model_args": args, "iter_num": 0, "best_val_loss": 0.0}, path)
    loaded, _ = load_checkpoint(str(path))
    assert loaded.config.lora_rank == 3


def test_cli_parsing():
    cfg, model_cfg = parse_config_args(TrainConfig, ModelConfig,
                                       argv=["--n_layer", "3", "--decay_lr", "false", "--learning_rate", "1e-4"])
    assert model_cfg.n_layer == 3 and cfg.decay_lr is False and cfg.learning_rate == 1e-4


TINY = ["--n_layer", "2", "--n_head", "2", "--n_embd", "32", "--block_size", "32", "--device", "cpu",
        "--eval_interval", "5", "--eval_iters", "2", "--log_interval", "5", "--warmup_iters", "2",
        "--lr_decay_iters", "10", "--grad_accum_steps", "2"]


@pytest.mark.slow
def test_pretrain_script_end_to_end(bin_dir, tmp_path, tokenizer):
    import train
    out = str(tmp_path / "ckpts")
    common = TINY + ["--data_dir", str(bin_dir), "--out_dir", out, "--batch_size", "4",
                     "--learning_rate", "1e-3", "--always_save_checkpoint", "true"]
    train.main(common + ["--max_iters", "10"])
    _, ckpt = load_checkpoint(os.path.join(out, "ckpt.pt"))
    assert ckpt["iter_num"] == 10 and ckpt["stage"] == "pretrain"

    # resume continues from iteration 10
    train.main(common + ["--max_iters", "15", "--init_from", "resume"])
    _, ckpt = load_checkpoint(os.path.join(out, "ckpt.pt"))
    assert ckpt["iter_num"] == 15


@pytest.mark.slow
def test_finetune_script_from_scratch_checkpoint(bin_dir, qa_dir, tmp_path, tokenizer):
    """pre-train (tiny) -> LoRA fine-tune -> answer a question, all through the real scripts."""
    import ask
    import finetune
    import train
    out = str(tmp_path / "ckpts")
    train.main(TINY + ["--data_dir", str(bin_dir), "--out_dir", out, "--max_iters", "5", "--batch_size", "2",
                       "--always_save_checkpoint", "true"])
    finetune.main(TINY + ["--init_from", "resume", "--resume_from", os.path.join(out, "ckpt.pt"),
                          "--data_dir", str(qa_dir), "--out_dir", out, "--lora_rank", "4", "--max_iters", "10",
                          "--toks_per_batch", "200", "--learning_rate", "1e-3", "--always_save_checkpoint", "true"])
    model, ckpt = load_checkpoint(os.path.join(out, "finetuned.pt"))
    assert ckpt["stage"] == "finetune" and model.config.lora_rank == 4
    # the pre-training checkpoint was not overwritten
    assert load_checkpoint(os.path.join(out, "ckpt.pt"))[1]["stage"] == "pretrain"
    ask.main(["--ckpt", os.path.join(out, "finetuned.pt"), "--question", "What color is the sky?",
              "--max_new_tokens", "5", "--device", "cpu"])
