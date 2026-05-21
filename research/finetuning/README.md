# SFT Finetuning — Qwen2.5-1.5B on Disturbed WikiTQ

Finetunes `Qwen/Qwen2.5-1.5B` (base) with QLoRA on the disturbed dataset so the
model learns to **detect → describe repair → reason → answer** when given a
*distorted* table (rendered as Markdown).

Hardware target: **single NVIDIA T4 16 GB** (Turing, no bf16, no FlashAttention-2).

---

## 1. Required training data format

Provide a **JSONL** file at `research/finetuning/data/sft_dataset.jsonl`
(the `build_sft_dataset.py` helper produces it for you). One example per line:

```jsonc
{
  "index": "Clinic_Visits_original_0__original__disturbed",
  "query": "What is the median systolic blood pressure ...?",
  "table_markdown": "| Patient | Condition | ... |\n|---|---|---|\n| ... |",
  "is_distorted": true,                          // optional, derived if missing
  "distortion_type": "semantic",                 // optional metadata
  "diversification_type": "label_mismatch",      // optional metadata

  // ===== Supervised target: ONE flowing reasoning paragraph (2-3 sentences) =====
  // For distorted tables, the reasoning should notice the issue mid-thought,
  // mentally correct it, and continue to the answer. For clean tables, it
  // just reasons straight through. Cite concrete column names / values.
  "reasoning": "Looking at the whole table, the `Population` column appears vertically shifted - two empty cells at the top and two orphan values past the last township; after sliding those values up, four townships have populations greater than 2,000.",

  // ===== Final answer (matches research/agents/output_format.py) =====
  "answer": 4,
  "dtype": "int"
}
```

Rules:
- `table_markdown` is **mandatory** at training time — that is how the table
  will be presented to the model at inference time too. If you only have a
  `data_file` path, `prepare_dataset.py` will auto-render it.
- `answer` must be JSON-serializable; `dtype` ∈ `{int, float, str, list, dict, set, pd.Series}`.
- `reasoning` is a **single string**, 2-3 sentences max.
  - For **clean** rows: just the derivation.
  - For **distorted** rows: weave the detection + correction into the
    reasoning naturally (no headers, no bullet points).
- The training target the model sees is:

  ```
  <reasoning>

  Final Answer:
  ```json
  {"answer": ..., "dtype": "..."}
  ```
  ```

- Legacy schema (`distortion_detection` + `repair_description` +
  `corrected_reasoning`) is still accepted and concatenated into one paragraph
  automatically.

---

## 2. Files

This folder is **fully self-contained** — you can copy `research/finetuning/`
to any machine and run it without the rest of the repository. The only project
dependency (`get_data_preview_markdown`) has been vendored as `data_preview.py`.

| File | Purpose |
|------|---------|
| `prompt_template.py`     | Single source of truth for prompt + target formatting |
| `data_preview.py`        | Vendored xlsx/csv → markdown helper (no repo deps) |
| `build_sft_dataset.py`   | **(repo-only)** Calls `prose.llm` to auto-generate the three one-sentence supervision labels for every record in `disturbed.json` → emits `data/sft_dataset.jsonl` |
| `prepare_dataset.py`     | Renders xlsx → markdown if needed, then splits into either train/val/test (`--mode split`) or K-fold CV folds + held-out test (`--mode kfold`) |
| `train_sft.py`           | QLoRA SFT training loop (T4-tuned). Accepts `--train-file`/`--val-file`/`--output-dir` CLI overrides so the same config drives any CV fold. |
| `evaluate_sft.py`        | Runs an adapter (or the bare base model if `--adapter` is omitted) on a test split and computes accuracy |
| `run_cv.py`              | K-fold CV orchestrator: trains each fold and evaluates it against the shared held-out test set, then writes `cv_summary.json` |
| `plot_training.py`       | Renders loss / lr / grad-norm curves from `training_log.jsonl` (single run or all CV folds, with mean ± stdev band) for the paper |
| `configs/qlora_t4.yaml`  | All hyperparameters (batch size, LR, LoRA rank, ...). Paths are resolved relative to the config file. |
| `requirements.txt`       | All Python deps needed (torch, transformers, peft, trl, bitsandbytes, openpyxl, ...) |

> Note: `build_sft_dataset.py` depends on `prose.llm` and the surrounding repo
> (it is meant to be run **once** on the dev machine to mint the training
> labels). Everything else in this folder is fully self-contained and can be
> copied to the T4 box on its own.

---

## 3. Typical workflow

(Run from inside the `finetuning/` folder, or from anywhere — paths in the
config are resolved relative to the config file.)

```pwsh
# 0a) (one-time) Install the CUDA build of PyTorch for T4 FIRST.
#     T4 = Turing / sm_75 — fully supported by stock PyTorch CUDA wheels.
pip install --index-url https://download.pytorch.org/whl/cu121 `
    torch==2.4.1 torchvision==0.19.1
# (Use cu118 instead of cu121 if your NVIDIA driver is < 525.)

# 0b) Install everything else
pip install -r requirements.txt

# 0c) Verify the GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True Tesla T4

# 1a) (repo-only, one-time) Mint the SFT reasoning labels with prose.llm.
#     Pass BOTH the distorted and the original (clean) splits so the model
#     learns to reason normally on clean tables and to spot/repair issues on
#     distorted ones. Writes research/finetuning/data/sft_dataset.jsonl with
#     one record per line containing a single `reasoning` paragraph.
python research\finetuning\build_sft_dataset.py `
    --input research\dataset\wikitq_dataset_filtered\disturbed.json `
    --input research\dataset\wikitq_dataset_filtered\original.json `
    --nproc 4 --resume

# 1b) Build splits from the annotated JSONL.
#     -- Default 70/10/20 train/val/test:
python prepare_dataset.py `
    --input  data/sft_dataset.jsonl `
    --outdir data/splits `
    --mode split --train 0.7 --val 0.1 --test 0.2 --seed 42

#     -- Or 80/20 train/test (no val) for CV-style runs:
python prepare_dataset.py `
    --input  data/sft_dataset.jsonl `
    --outdir data/splits `
    --mode split --train 0.8 --val 0.0 --test 0.2 --seed 42

#     -- Or 5-fold cross-validation with a 20% held-out test set:
python prepare_dataset.py `
    --input  data/sft_dataset.jsonl `
    --outdir data/splits `
    --mode kfold --k 5 --test 0.2 --seed 42

# 2) Train (QLoRA, ~6–8h on a T4 for ~350 train samples / 3 epochs)
python train_sft.py --config configs/qlora_t4.yaml

# 2b) (CV only) Train + evaluate all folds in one go.
#     Each fold trains into its own outputs/cv-.../fold_k/ directory and is
#     evaluated against the SHARED held-out test set.
python run_cv.py `
    --config  configs/qlora_t4.yaml `
    --splits  data/splits `
    --outroot outputs/cv-qwen25-1p5b

# 3) Evaluate (finetuned adapter)
python evaluate_sft.py `
    --adapter outputs/qwen25-1p5b-disturbed/final `
    --test    data/splits/test.jsonl `
    --out     outputs/qwen25-1p5b-disturbed/test_predictions.jsonl

# 3b) Evaluate the BASE model on the same test split (no --adapter).
#     Useful for an apples-to-apples "before finetuning" baseline.
python evaluate_sft.py `
    --base-model Qwen/Qwen2.5-1.5B-Instruct `
    --test       data/splits/test.jsonl `
    --out        outputs/baseline/test_predictions.jsonl
```

Note: in `prepare_dataset.py`, if your input JSONL uses **relative**
`data_file` paths, pass `--repo-root <dir>` to tell the script where to resolve
them from (defaults to current working directory).

### Cross-validation workflow

For small training sets it's often better to use **K-fold cross-validation**
than a single 70/10/20 split. The pipeline supports it with three pieces:

1. `prepare_dataset.py --mode kfold --k K --test T` carves off a fixed `T`
   fraction as the held-out test set (shared across folds), then partitions
   the remainder into `K` equal-sized folds. It writes:

   ```
   data/splits/
       test.jsonl
       fold_0/train.jsonl    fold_0/val.jsonl
       fold_1/train.jsonl    fold_1/val.jsonl
       ...
       fold_{K-1}/train.jsonl  fold_{K-1}/val.jsonl
   ```

2. `train_sft.py` accepts `--train-file`, `--val-file`, and `--output-dir`
   CLI overrides, so the same YAML config can be reused for every fold
   without editing it.

3. `run_cv.py` loops over all folds, trains each into its own
   `outputs/cv-.../fold_k/` directory, evaluates the resulting adapter
   against the **shared** `test.jsonl`, and writes `cv_summary.json` with
   the per-fold accuracies plus mean ± stdev. Use `--folds 0 1 2` to run
   a subset (e.g. resume on the remaining folds), or `--skip-train` /
   `--skip-eval` if you only want one half of the loop.

   **Crash-safe resume.** `run_cv.py` is idempotent: just rerun the same
   command after a crash and it will pick up exactly where it left off.
   Per fold, the resume logic is:

   * If `fold_k/final/adapter_config.json` exists → training is **skipped**
     (the fold finished). Override with `--force-train`.
   * Otherwise `train_sft.py` runs and auto-resumes from the latest
     `checkpoint-*` inside the fold's `output_dir` (weights + optimizer +
     LR scheduler + RNG + step/epoch). Override with `--no-resume` to
     force a fresh training run per fold.
   * If `fold_k/test_predictions.jsonl` exists → evaluation is **skipped**
     and the cached per-row accuracy is reused for the summary. Override
     with `--force-eval`.

   So a typical recovery after a mid-training crash is simply:

   ```pwsh
   python run_cv.py `
       --config  configs/qlora_t4.yaml `
       --splits  data/splits `
       --outroot outputs/cv-qwen25-1p5b
   ```

To switch back to the classic single 70/10/20 split, just rerun
`prepare_dataset.py --mode split --train 0.7 --val 0.1 --test 0.2` and call
`train_sft.py` / `evaluate_sft.py` directly — no other code change required.

---

## 4. T4-specific notes

- **QLoRA (4-bit NF4)** via `bitsandbytes`; base weights stay on GPU at ~1.5 GB.
- **fp16** compute (T4 has no bf16). `tf32` is also unavailable.
- **No FlashAttention-2** (Turing). We use PyTorch SDPA.
- **`per_device_train_batch_size=1` + `gradient_accumulation_steps=16`** keeps
  peak memory < 14 GB even with `max_seq_len=2048`.
- `gradient_checkpointing=True`.
- LoRA rank 16 on `q,k,v,o,gate,up,down_proj`.

## 5. Crash recovery / checkpointing

Training is **fully crash-safe**:

- A checkpoint is written every **`save_steps` (default = 50)** optimizer steps
  into `outputs/qwen25-1p5b-disturbed/checkpoint-XXX/`. Each checkpoint contains
  the LoRA adapter weights, optimizer state, LR scheduler state, RNG state, and
  the current global step/epoch — i.e. everything needed to resume *exactly*.
- The last `save_total_limit` (default = 3) checkpoints are kept; older ones
  are pruned automatically to save disk.
- On the next launch, `train_sft.py` **auto-detects** the most recent
  `checkpoint-*` folder in `output_dir` and resumes from it. You don't have to
  pass any flag.
- To force a **fresh** run instead, either:
  - delete (or rename) `outputs/qwen25-1p5b-disturbed/`, or
  - pass `--no-resume` on the CLI, or
  - set `resume_from_checkpoint: false` in the YAML.

Tuning the trade-off:
- Lower `save_steps` ⇒ less work lost on crash, more disk I/O.
- A checkpoint for a 1.5B-param model with LoRA r=16 is small (~30 MB adapter
  + optimizer state ≈ a few hundred MB), so `save_steps: 50` is cheap.

Example: if your machine reboots mid-training, just rerun the same command:

```pwsh
python train_sft.py --config configs/qlora_t4.yaml
```

and you'll see `[resume] Found existing checkpoint(s) ... — resuming.`


## 6. Training metrics & paper-ready charts

Every run is tracked in two redundant ways so you can plot it later (and
recover even if one of the formats is missing):

1. **`<output_dir>/training_log.jsonl`** -- one JSON line per Trainer log dict,
   containing at least `step`, `epoch`, `loss`, `learning_rate`, `grad_norm`,
   and (on eval steps) `eval_loss`. Written incrementally and flushed after
   every line, so a crashed run leaves a partial-but-valid file. On resume
   the file is appended to; `plot_training.py` de-duplicates by `step` so the
   curve stays clean.
2. **TensorBoard event files** under `<output_dir>/runs/<timestamp>/` (enabled
   by the YAML default `report_to: ["tensorboard"]`). Live-view with:

   ```pwsh
   tensorboard --logdir outputs/qwen25-1p5b-disturbed/runs
   # or, for CV:
   tensorboard --logdir outputs/cv-qwen25-1p5b
   ```

To render the static figures that go into the paper, use `plot_training.py`:

```pwsh
# Single run
python plot_training.py `
    --log    outputs/qwen25-1p5b-disturbed/training_log.jsonl `
    --outdir outputs/qwen25-1p5b-disturbed/plots `
    --pdf

# All CV folds at once -- auto-discovers fold_*/training_log.jsonl and
# overlays one curve per fold plus a mean +/- stdev band.
python plot_training.py `
    --log    outputs/cv-qwen25-1p5b `
    --outdir outputs/cv-qwen25-1p5b/plots `
    --pdf
```

This produces `loss.png`, `eval_loss.png`, `learning_rate.png`,
`grad_norm.png` (plus matching PDFs when `--pdf` is set) at 200 dpi, ready
to drop into a paper. Pass `--log-y` to render losses on a log scale,
`--no-mean` to suppress the mean/stdev overlay.

You can also report final per-fold accuracies straight from `cv_summary.json`
produced by `run_cv.py` (mean accuracy, stdev, min/max, and per-fold rows).