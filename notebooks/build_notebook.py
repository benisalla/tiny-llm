"""Build notebooks/train_kaggle_colab.ipynb: a self-contained notebook.

Every file of the `tiny_llm` package and of model/scripts is copied into the notebook at build time
(so it is identical to the repo); the notebook needs no GitHub access.
Regenerate after changing the code (a test fails if the notebook is out of date):
    python notebooks/build_notebook.py . notebooks/train_kaggle_colab.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PACKAGE = ROOT / "model" / "tiny_llm"
SCRIPTS = ROOT / "model" / "scripts"

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


def write_file_cell(path, dest):
    code(f"%%writefile {dest}\n" + path.read_text().rstrip())


# ---------------------------------------------------------------------------
md("""
# Tiny LLM: training notebook (Kaggle & Colab)

Train and fine-tune **Tiny LLM** on a **free Kaggle or Colab GPU**.
This notebook is **self-contained**: the whole `tiny_llm` package and its scripts are written to disk
by the cells below, identical to the [tiny-llm](https://github.com/benisalla/tiny-llm) repo.
Just upload it and press **Run all**.

1. Setup
2. The code (`tiny_llm` package + scripts)
3. Sanity check on simple data: addition from scratch, LoRA reading comprehension
4. Pre-training on OpenWebText (the Kaggle dataset `benallaismail/gpt-data`)
5. Chat with another small LLM loaded into our architecture
6. Download the checkpoints

**Kaggle:** Settings → Accelerator → GPU, and Internet → on.
**Colab:** Runtime → Change runtime type → GPU.
""")

md("## 1. Setup")
code("""
import os, sys, subprocess

ON_KAGGLE = os.path.exists("/kaggle")
ON_COLAB = "google.colab" in sys.modules
WORK = "/kaggle/working" if ON_KAGGLE else ("/content" if ON_COLAB else os.getcwd())
os.chdir(WORK)
print("working directory:", WORK)

# torch is preinstalled on Kaggle and Colab
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.36", "tiktoken"], check=True)
os.makedirs("tiny_llm", exist_ok=True)
os.makedirs("scripts", exist_ok=True)
os.environ["PYTHONPATH"] = WORK  # so `python scripts/...` finds the package

import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
""")

md("""
## 2. The code

The cells below write the package (`tiny_llm/`) and the command-line scripts (`scripts/`).
You don't need to read them to use the notebook; they are here so it runs without GitHub.
""")
for path in sorted(PACKAGE.glob("*.py")):
    write_file_cell(path, f"tiny_llm/{path.name}")
for path in sorted(SCRIPTS.glob("*.py")):
    write_file_cell(path, f"scripts/{path.name}")

code("""
sys.path.insert(0, WORK)
import tiny_llm
from tiny_llm.families import markdown_table
from IPython.display import Markdown, display

print("tiny_llm", tiny_llm.__version__)
display(Markdown(markdown_table()))  # what changes for each supported model family
""")

md("""
## 3. Sanity check on simple data

Two small synthetic tasks that prove the model learns:
**addition** learned from scratch (scored on problems never seen in training) and
**reading comprehension** with LoRA fine-tuning (scored on people, cities and jobs never seen in training).
""")
code("!python scripts/make_toy_data.py --out data/toy")
code("""
!python scripts/train.py --data_dir data/toy/arithmetic --ckpt_name arithmetic.pt \\
    --n_layer 4 --n_head 4 --n_embd 128 --block_size 64 --batch_size 64 --grad_accum_steps 1 \\
    --learning_rate 2e-3 --min_lr 2e-4 --warmup_iters 100 --lr_decay_iters 3000 --max_iters 3000 \\
    --eval_interval 500 --eval_iters 20 --log_interval 100
!python scripts/eval_toy.py arithmetic --ckpt checkpoints/arithmetic.pt --data data/toy/arithmetic/test.json
""")
code("""
!python scripts/eval_toy.py qa --pretrained distilgpt2 --data data/toy/qa/val.json   # before fine-tuning
!python scripts/finetune.py --init_from distilgpt2 --data_dir data/toy/qa --lora_rank 8 --learning_rate 1e-3 \\
    --max_iters 300 --warmup_iters 20 --lr_decay_iters 300 --grad_accum_steps 1 --toks_per_batch 2000 \\
    --eval_interval 100 --eval_iters 10 --ckpt_name qa_finetuned.pt
!python scripts/eval_toy.py qa --ckpt checkpoints/qa_finetuned.pt --data data/toy/qa/val.json
""")

md("""
## 4. Pre-training on OpenWebText

Add the Kaggle dataset **benallaismail/gpt-data** (a quarter of OpenWebText, already tokenized into
`train.bin` / `val.bin`) to the notebook with *Add Input*. Without it this cell is skipped.
Increase `--max_iters` for a real run, `--init_from resume` continues from the last checkpoint.
""")
code("""
import glob
bins = glob.glob("/kaggle/input/**/train.bin", recursive=True)
if bins:
    DATA_DIR = os.path.dirname(bins[0])
    print("data:", DATA_DIR)
    !python scripts/train.py --data_dir {DATA_DIR} --ckpt_name owt.pt \\
        --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 --batch_size 32 --grad_accum_steps 4 \\
        --learning_rate 1e-3 --min_lr 1e-4 --warmup_iters 200 --lr_decay_iters 5000 --max_iters 5000 \\
        --eval_interval 500 --eval_iters 50 --log_interval 50
    !python scripts/generate.py --ckpt checkpoints/owt.pt --prompt "ismail ben alla is the next elon mask because"
else:
    print("dataset not attached, skipping (Add Input -> benallaismail/gpt-data)")
""")

md("""
## 5. Chat with another small LLM

Tiny LLM is not limited to GPT-2: any supported family (Llama, Qwen2, GPT-NeoX) loads into the same
from-scratch code. Here SmolLM2-135M-Instruct (Llama family) answers through its own chat template.
""")
code("""
from tiny_llm import TinyLLM, create_tokenizer, stream_chat

name = "HuggingFaceTB/SmolLM2-135M-Instruct"   # or "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = create_tokenizer(name)
model = TinyLLM.from_pretrained(name, tokenizer).to(device).eval()

messages = [{"role": "user", "content": "Explain in two sentences what a transformer is."}]
for piece in stream_chat(model, tokenizer, messages, max_new_tokens=80, temperature=0.3):
    print(piece, end="", flush=True)
""")

md("## 6. Download the checkpoints")
code("""
import shutil
if os.path.isdir("checkpoints"):
    archive = shutil.make_archive("tiny-llm-checkpoints", "zip", "checkpoints")
    print("created", archive)
    if ON_COLAB:
        from google.colab import files
        files.download(archive)
    elif ON_KAGGLE:
        print("Kaggle: click 'Save Version', then download it from the notebook's Output tab.")
""")

for i, cell in enumerate(cells):
    cell["id"] = f"cell-{i:03d}"  # deterministic ids: rebuilding gives the same file
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.validate(nb)
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
