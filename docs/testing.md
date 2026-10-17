# Testing

```bash
pip install -e "model/[dev]"
cd model && pytest            # 69 tests
cd ../chatbot && pytest       # 9 tests
```

| file | what it proves |
| --- | --- |
| `test_layers.py` | norms, attention vs the textbook formula (MHA / GQA / MQA), causality, KV cache == full attention |
| `test_lora.py` | LoRA starts as a no-op, merging is exact, only LoRA params get gradients |
| `test_model.py` | shapes, initial loss ≈ ln(V), ignore_index, sampling (greedy, top-k, top-p, banned ids), can overfit a batch |
| `test_families.py` | for GPT-2, Llama (tied / untied, GQA), Qwen2, GPT-NeoX (parallel / sequential): same logits and same greedy generation as Hugging Face, with and without cache, also past the context window (tiny random models, no download) |
| `test_pretrained.py` | real GPT-2 weights: logits and generation identical to Hugging Face |
| `test_data.py` | batches, answer-only masking, padding never in the loss, special token ids unchanged |
| `test_training.py` | lr schedule, checkpoints (incl. old checkpoints), CLI, pre-train → resume → LoRA fine-tune → ask end to end |
| `test_notebook.py` | the Kaggle/Colab notebook is up to date, its code imports, its commands use existing options |
| `test_hub_models.py` | opt-in (`TINY_LLM_TEST_HUB=1`, ~1.7 GB): real DistilGPT2, SmolLM2-Instruct, Pythia, Qwen2.5-Instruct |
| `chatbot/tests` | the engine (modes, streaming, truncation, errors) and the Streamlit UI driven headlessly |

`pytest -m "not network"` skips the tests that download from the Hugging Face Hub.
CI (`.github/workflows/tests.yml`) runs both suites on every push.
