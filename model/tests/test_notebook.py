"""The Kaggle/Colab notebook must stay identical to the code and only use options the scripts accept."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "train_kaggle_colab.ipynb"


def cells():
    return nbformat.read(NOTEBOOK, as_version=4).cells


def test_notebook_is_up_to_date(tmp_path):
    fresh = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, str(ROOT / "notebooks" / "build_notebook.py"), str(ROOT), str(fresh)],
                   check=True, capture_output=True)
    assert fresh.read_text() == NOTEBOOK.read_text(), \
        "notebook is out of date: python notebooks/build_notebook.py . notebooks/train_kaggle_colab.ipynb"


def test_written_package_imports_and_runs(tmp_path):
    """Write the files exactly like the notebook does, then import and run the package from there."""
    for cell in cells():
        if cell.source.startswith("%%writefile"):
            header, body = cell.source.split("\n", 1)
            dest = tmp_path / header.split()[1]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body + "\n")
    script = ("import torch, tiny_llm; "
              "m = tiny_llm.TinyLLM(tiny_llm.ModelConfig(vocab_size=50, block_size=16, n_layer=1, n_head=2, "
              "n_embd=16, pos_encoding='rope', gated_mlp=True, norm_type='rmsnorm')); "
              "print(m.generate(torch.zeros(1, 3, dtype=torch.long), 4).shape, tiny_llm.__file__)")
    out = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "torch.Size([1, 7])" in out.stdout and str(tmp_path) in out.stdout


def test_notebook_commands_use_existing_options():
    commands = []
    for cell in cells():
        source = cell.source.replace("\\\n", " ")
        commands += re.findall(r"!python (scripts/\w+\.py)([^\n]*)", source)
    assert len(commands) >= 6
    for script, args in commands:
        help_text = subprocess.run([sys.executable, str(ROOT / "model" / script), "--help"],
                                   capture_output=True, text=True, cwd=ROOT / "model").stdout
        for flag in re.findall(r"(--\w+)", args):
            assert flag in help_text, f"{script} has no option {flag}"
