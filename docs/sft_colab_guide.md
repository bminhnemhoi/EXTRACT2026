# SFT on free Colab/Kaggle — operator guide

The local RTX 4050 (6 GB) cannot QLoRA a useful (7B) model, and the
installed torch is CPU-only with no native-Windows Unsloth path
(ADR 0008 / Day-12 hardware check). So the SFT runs on a **free** cloud
T4; everything else stays local.

## 0. One-time: produce the training data (local, already done)

```powershell
cd d:\Exact2026\exact-veriscope-agent
uv run python -m exact_agent.train.prepare_sft
# writes data/processed/sft_chat_train.jsonl (1765) + sft_chat_val.jsonl (196)
```

These are Qwen chat-format rows (`messages = [system, user, assistant]`)
where the assistant turn is the exact JSON envelope the API emits. The
SFT objective is format/explanation stability, **not** computation
(the solver still does the math — ADR 0001/0006).

## 1. Train (free GPU, ~30-60 min)

**Colab** (recommended): open `notebooks/sft_qwen_colab.ipynb` in
[colab.research.google.com](https://colab.research.google.com) →
Runtime → Change runtime type → **T4 GPU** → Run all. Upload the two
JSONL files when cell 2 asks.

**Kaggle**: New Notebook → Add Data → upload the two JSONL as a
dataset → set `TRAIN_PATH`/`VAL_PATH` in cell 2 → Settings →
Accelerator **GPU T4 x2** → Run all.

Output: `exact_qwen25_7b_lora.zip` (~80-200 MB — adapter only).
Download it.

Model = **Qwen2.5-7B-Instruct** (open-source, ≤8B, non-thinking →
matches our extractor architecture; ADR 0008 ruled out Qwen3).
Hyperparameters mirror `configs/training/sft_qwen3_8b.yaml`
(r=16, α=32, lr=2e-4, 2 epochs, cosine).

## 2. Consume the adapter

Unzip to `d:\Exact2026\exact-veriscope-agent\models\exact_qwen25_7b_lora\`.

### Path A — local spot-check via Ollama (6 GB, slow but free)

Merge + convert to GGUF, then register with Ollama. On Colab, append a
cell:

```python
model.save_pretrained_merged("merged", tokenizer, save_method="merged_16bit")
# then llama.cpp convert+quantize to q4_K_M (≈4.7 GB) and download the .gguf
```

Local `Modelfile`:

```
FROM ./exact-qwen25-7b-sft.q4_K_M.gguf
PARAMETER temperature 0.2
```

```powershell
ollama create exact-qwen25-sft -f Modelfile
```

Then point `configs/model.yaml`:

```yaml
llm:
  backbone: "exact-qwen25-sft"
  vllm_base_url: "http://localhost:11434/v1"
  mode: "chat"
  disable_thinking: false
```

Re-measure: `uv run python scripts\run_eval.py --task physics --with-llm`
and `--task logic --with-llm`. Compare against
`outputs/eval/day12_llm/` (current scorer reference).
qwen2.5-7b q4 on 6 GB works but is slow (~1-3 tok/s with CPU offload) —
fine for a holdout pass, not for the live submission endpoint.

### Path B — production submission endpoint (rented GPU)

For the actual competition endpoint, serve base + adapter on a rented
GPU with vLLM `--enable-lora`:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-lora --lora-modules exact=/path/exact_qwen25_7b_lora \
  --max-model-len 4096 --port 8001
```

`configs/model.yaml` → `vllm_base_url: http://<host>:8001/v1`,
`backbone: exact`. The FastAPI container (`docker/Dockerfile.api`)
stays slim and just points at this URL.

## 3. What SFT is expected to move

- **Logic** (stuck 37.5%): better NL→FOL translation → more Z3
  fallbacks decide → the biggest lever.
- **Physics extraction consistency**: fewer `missing_input` /
  mis-extractions (e.g. the NL013 14.83→0.47 V flip — qwen2.5:3b
  sampling variance; a 7B SFT'd model is steadier).
- **P2 explanation quality**: phrasing aligned with gold explanations.

It is NOT expected to move solver-modelling gaps (LD/DT vector
composition) — those are deferred CoT work, separate from SFT.

## 4. Reproducibility / rules compliance

- Only the cleaned competition dataset is used (no external data) —
  state this in the 1-page solution PDF.
- Backbone Qwen2.5-7B-Instruct: open-source, 7B ≤ 8B cap. ✓
- Seed 42 throughout; LoRA config is in version control.

## 5. Troubleshooting

**`TypeError: ... unexpected keyword argument 'tokenizer'`** then, if you
only delete it, **`AttributeError: 'NoneType' object has no attribute
'convert_ids_to_tokens'`** (cell 5, both fixed in the committed
notebook). transformers 5.x **renamed** `Trainer(tokenizer=...)` to
`processing_class=...` — it was *not* removed. Dropping it entirely
makes Unsloth's `fix_untrained_tokens` receive `tokenizer=None` →
the AttributeError. The fix is `SFTTrainer(model=model,
processing_class=tokenizer, ...)`, with `dataset_text_field` +
`max_seq_length` in `SFTConfig`.

**Version soup in general**: don't `pip install` a pinned `trl` /
`transformers` next to Unsloth — let `pip install unsloth` resolve one
consistent stack (cell 1). Mixing an old `trl` with Colab's
`transformers` 5.x is the root cause of the `tokenizer` error.

**`warmup_ratio is deprecated`** and `You passed a max_seq_length /
dataset_text_field argument…` — warnings only, safe to ignore; the run
proceeds.

**OOM on T4**: drop `per_device_train_batch_size` to 1 and raise
`gradient_accumulation_steps` to 16 (same effective batch), or set
`max_seq_length=1536`.
