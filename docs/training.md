# Training

All commands run from `model/`. Every field of `ModelConfig` and `TrainConfig` is a flag
(`python scripts/train.py --help`). On a GPU, mixed precision (bfloat16 / float16) is automatic.
For free GPUs use the self-contained notebook `notebooks/train_kaggle_colab.ipynb`.

## Data

| stage | format | made by |
| --- | --- | --- |
| pre-training | `train.bin`, `val.bin`: flat uint16 GPT-2 token ids | `scripts/prepare_data.py --src <txt folder> --dst <dir>` |
| fine-tuning | `train.json`, `val.json`: `[{"context", "question", "answer"}]` | your data, or `scripts/make_toy_data.py` |

A fine-tuning sample becomes `<sost>{context}<ques>{question}<ans>{answer}<eost>`. By default only the
answer tokens are in the loss (`--loss_on_answer_only`) and padding never is.

## Pre-training

```bash
python scripts/train.py --data_dir ./data/openwebtext --init_from scratch          # GPT-2 small size
python scripts/train.py --data_dir ./data/openwebtext --init_from resume           # continue
python scripts/train.py --data_dir ./data/openwebtext --init_from HuggingFaceTB/SmolLM2-135M   # continue another LLM
```

The loop (`tiny_llm/training.py`): linear warmup + cosine decay, gradient accumulation
(`--grad_accum_steps`), gradient clipping, evaluation every `--eval_interval` steps, checkpoint
`out_dir/ckpt_name` written when the validation loss improves (or always with `--always_save_checkpoint true`).

## Fine-tuning (LoRA)

```bash
python scripts/finetune.py --init_from gpt2 --data_dir ./data/qa --lora_rank 8 --learning_rate 3e-4
python scripts/finetune.py --init_from resume --resume_from ./checkpoints/ckpt.pt --data_dir ./data/qa --lora_rank 8
```

With `--lora_rank > 0` the LLM is frozen and only the LoRA matrices (+ the token embeddings, so the new
special tokens are learned, `--train_embeddings`) are trained. LoRA is merged into the weights before
inference, so it costs nothing at generation time. `--lora_rank 0` is full fine-tuning.

Checkpoints store the model config, the optimizer state, the stage (`pretrain` / `finetune`) and the
tokenizer name, so `ask.py`, `generate.py` and the chatbot rebuild everything from the file alone.

## Sanity check on simple data

```bash
python scripts/make_toy_data.py --out ./data/toy
```

- `arithmetic/`: `a+b=c` lines, 1,000 (a, b) pairs held out → exact-match accuracy of a model trained from scratch.
- `qa/`: reading comprehension about made-up people, val uses unseen names, cities and jobs → accuracy before / after LoRA.

`scripts/eval_toy.py` computes the accuracies. Commands and results are in the README.
