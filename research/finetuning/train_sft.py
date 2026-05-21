"""QLoRA SFT training for Qwen2.5-1.5B on the disturbed-table dataset.

Designed for a single NVIDIA T4 (16 GB, Turing, fp16).

Run:
    python research/finetuning/train_sft.py --config research/finetuning/configs/qlora_t4.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# Local import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_template import build_messages, build_target  # noqa: E402


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def make_dataset(rows: List[Dict[str, Any]], tokenizer, max_seq_len: int) -> Dataset:
    """Tokenize prompt + completion. Mask the prompt tokens in `labels`.

    We encode in plain Python first and then build the `datasets.Dataset` from
    the encoded records. Building it directly from `rows` would fail because
    `answer` is heterogeneous across samples (int / list / dict / ...), and
    PyArrow cannot infer a single schema for that column.
    """

    def _encode(ex: Dict[str, Any]) -> Dict[str, Any]:
        messages = build_messages(ex)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        completion_text = build_target(ex) + tokenizer.eos_token

        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False, truncation=False
        )["input_ids"]
        completion_ids = tokenizer(
            completion_text, add_special_tokens=False, truncation=False
        )["input_ids"]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids[:]

        # Truncate from the LEFT of the prompt to preserve the completion / answer
        if len(input_ids) > max_seq_len:
            overflow = len(input_ids) - max_seq_len
            if overflow >= len(prompt_ids):
                # Pathological case: completion alone exceeds budget; truncate completion tail.
                input_ids = input_ids[-max_seq_len:]
                labels = labels[-max_seq_len:]
            else:
                input_ids = input_ids[overflow:]
                labels = labels[overflow:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "length": len(input_ids),
        }

    encoded = [_encode(r) for r in rows]
    return Dataset.from_list(encoded)


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force a fresh training run, ignoring any existing checkpoints in output_dir.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Override data.train_file from the YAML (useful for K-fold CV loops).",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=None,
        help="Override data.val_file from the YAML (useful for K-fold CV loops).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override training.output_dir from the YAML (useful for K-fold CV loops).",
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides on top of the YAML.
    if args.train_file is not None:
        cfg["data"]["train_file"] = args.train_file
    if args.val_file is not None:
        cfg["data"]["val_file"] = args.val_file
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir

    set_seed(cfg["training"]["seed"])

    # --- Sanity check: T4-class GPU is required for this config -----------
    # QLoRA + bitsandbytes 4-bit quantization needs a CUDA GPU. If torch was
    # installed as a CPU-only wheel, fail loudly NOW rather than after the
    # first forward pass (which would crash deep inside bnb).
    if not torch.cuda.is_available():
        raise SystemExit(
            "\n[FATAL] torch.cuda.is_available() is False.\n"
            "  - Did you install the CUDA build of PyTorch?\n"
            "  - For an NVIDIA T4, run BEFORE `pip install -r requirements.txt`:\n"
            "      pip install --index-url https://download.pytorch.org/whl/cu121 "
            "torch==2.4.1 torchvision==0.19.1\n"
            "  - Then verify with: "
            "python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"\n"
        )
    dev_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[gpu] {dev_name}  sm_{cap[0]}{cap[1]}  torch={torch.__version__}  cuda={torch.version.cuda}")
    if cap < (7, 5):
        print(
            f"[gpu] WARNING: device capability sm_{cap[0]}{cap[1]} is older than "
            "Turing (sm_75); bitsandbytes 4-bit may not work."
        )

    # --- tokenizer ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name_or_path"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- model (4-bit QLoRA) ----------------------------------------------
    qcfg = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name_or_path"],
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs=cfg["training"].get("gradient_checkpointing_kwargs"),
    )

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- data --------------------------------------------------------------
    # Resolve data paths relative to the config file's directory if not absolute,
    # so the folder is portable.
    cfg_dir = args.config.resolve().parent
    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (cfg_dir / pp).resolve()

    train_path = _resolve(cfg["data"]["train_file"])
    val_path = _resolve(cfg["data"]["val_file"])
    if not train_path.exists():
        # Fallback: try CWD-relative (useful when paths are given relative to repo root).
        alt = (Path.cwd() / cfg["data"]["train_file"]).resolve()
        if alt.exists():
            train_path = alt
            val_path = (Path.cwd() / cfg["data"]["val_file"]).resolve()
    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)
    print(f"[data] train={len(train_rows)}  val={len(val_rows)}")

    train_ds = make_dataset(train_rows, tokenizer, cfg["data"]["max_seq_len"])
    val_ds = make_dataset(val_rows, tokenizer, cfg["data"]["max_seq_len"])

    # --- training args -----------------------------------------------------
    tcfg = cfg["training"]
    output_dir = str(_resolve(tcfg["output_dir"]))
    resume_from_checkpoint = tcfg.get("resume_from_checkpoint", True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        optim=tcfg["optim"],
        fp16=tcfg["fp16"],
        bf16=tcfg["bf16"],
        tf32=tcfg["tf32"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs=tcfg.get("gradient_checkpointing_kwargs"),
        logging_steps=tcfg["logging_steps"],
        logging_first_step=True,
        logging_nan_inf_filter=False,
        disable_tqdm=False,
        save_strategy=tcfg["save_strategy"],
        save_steps=tcfg.get("save_steps", 500),
        eval_strategy=tcfg["eval_strategy"],
        eval_steps=tcfg.get("eval_steps", tcfg.get("save_steps", 500)),
        save_total_limit=tcfg["save_total_limit"],
        save_safetensors=tcfg.get("save_safetensors", True),
        load_best_model_at_end=tcfg["load_best_model_at_end"],
        metric_for_best_model=tcfg["metric_for_best_model"],
        greater_is_better=tcfg["greater_is_better"],
        report_to=tcfg["report_to"],
        seed=tcfg["seed"],
        dataloader_num_workers=tcfg["dataloader_num_workers"],
        group_by_length=tcfg["group_by_length"],
        length_column_name="length",
        remove_unused_columns=False,
    )

    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)

    # --- Live progress callback -------------------------------------------
    # HF's tqdm bar may not refresh when stdout is not a TTY (ssh + tee, nohup,
    # nohup.out, jupyter terminals, etc.). This callback prints a one-line
    # heartbeat every optimizer step so you can confirm training is actually
    # progressing inside a single epoch.
    import time as _time

    class StepHeartbeat(TrainerCallback):
        def __init__(self, total_steps_hint: int | None = None) -> None:
            self.t0 = None
            self.last_t = None
            self.total = total_steps_hint

        def on_train_begin(self, args, state, control, **kwargs):
            self.t0 = _time.time()
            self.last_t = self.t0
            total = state.max_steps if state.max_steps and state.max_steps > 0 else self.total
            print(f"[hb] training started; total optimizer steps = {total}", flush=True)

        def on_step_end(self, args, state, control, **kwargs):
            now = _time.time()
            dt = now - (self.last_t or now)
            self.last_t = now
            elapsed = now - (self.t0 or now)
            total = state.max_steps if state.max_steps and state.max_steps > 0 else self.total
            eta_s = (total - state.global_step) * dt if total and dt > 0 else 0
            print(
                f"[hb] step {state.global_step}/{total or '?'}  "
                f"epoch {state.epoch:.2f}  "
                f"step_time={dt:.1f}s  elapsed={elapsed/60:.1f}m  "
                f"eta~{eta_s/60:.1f}m",
                flush=True,
            )

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            # Print loss/lr lines immediately (HF normally batches these).
            keys = ("loss", "learning_rate", "grad_norm", "eval_loss")
            shown = {k: logs[k] for k in keys if k in logs}
            if shown:
                print(f"[log] step={state.global_step} {shown}", flush=True)

    class JsonlLogWriter(TrainerCallback):
        """Append every Trainer log dict (loss, lr, grad_norm, eval_loss, ...)
        as one JSON line to ``<output_dir>/training_log.jsonl`` so the run can
        be replotted offline with no extra deps.

        Crash-safe: opened in append mode and flushed after every line; on
        resume it simply continues appending. ``plot_training.py`` keeps only
        the LAST occurrence of each ``global_step`` so duplicates from resumes
        don't double-count.
        """

        def __init__(self, log_path: Path) -> None:
            self.log_path = Path(log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = None

        def _open(self) -> None:
            if self._fh is None:
                self._fh = self.log_path.open("a", encoding="utf-8")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            self._open()
            rec = {
                "step":  state.global_step,
                "epoch": state.epoch,
                **{k: v for k, v in logs.items() if isinstance(v, (int, float, str, bool))},
            }
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()

        def on_train_end(self, args, state, control, **kwargs):
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[
            StepHeartbeat(),
            JsonlLogWriter(Path(output_dir) / "training_log.jsonl"),
        ],
    )

    # --- Resume logic ------------------------------------------------------
    # `resume_from_checkpoint=True` (default) makes HF Trainer look inside
    # `output_dir` for the most recent `checkpoint-XXX` folder and pick up
    # exactly where it left off (model weights, optimizer state, LR schedule,
    # RNG state, current step/epoch). If no checkpoint exists yet it starts
    # from scratch. Set `resume_from_checkpoint: false` in the YAML or pass
    # `--no-resume` on the CLI to force a fresh run.
    if args.no_resume:
        resume_from_checkpoint = False

    has_ckpt = any(Path(output_dir).glob("checkpoint-*")) if Path(output_dir).exists() else False
    if resume_from_checkpoint and has_ckpt:
        print(f"[resume] Found existing checkpoint(s) in {output_dir} — resuming.")
        trainer.train(resume_from_checkpoint=True)
    else:
        if resume_from_checkpoint and not has_ckpt:
            print(f"[resume] No prior checkpoint in {output_dir} — starting fresh.")
        trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[done] adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
