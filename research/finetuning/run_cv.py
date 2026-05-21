"""K-fold cross-validation orchestrator for the QLoRA SFT pipeline.

Assumes you already ran `prepare_dataset.py --mode kfold` so that the splits
directory looks like:

    <splits>/
        test.jsonl
        fold_0/train.jsonl
        fold_0/val.jsonl
        fold_1/train.jsonl
        fold_1/val.jsonl
        ...

For each fold k this script:
  1. Calls `train_sft.py` with the fold's train/val files and a per-fold
     `output_dir` so checkpoints don't clobber each other.
  2. Calls `evaluate_sft.py` to score the fold's final adapter on the
     SHARED held-out test set.
  3. Records the per-fold accuracy.

At the end it prints mean/std accuracy across folds, and writes a small
JSON summary alongside the splits directory.

Resume semantics (safe to re-run after a crash):
  * If `fold_k/final/adapter_config.json` already exists, training is
    SKIPPED for that fold. Use `--force-train` to retrain anyway.
  * If `fold_k/test_predictions.jsonl` already exists, evaluation is
    SKIPPED and the cached accuracy is reused. Use `--force-eval` to redo.
  * Otherwise `train_sft.py` runs and, since `resume_from_checkpoint=true`
    is the YAML default, it auto-continues from the latest
    `checkpoint-*` inside the per-fold output dir (weights, optimizer
    state, LR scheduler, RNG, step/epoch). Pass `--no-resume` to force
    a fresh training run per fold.

Example:
    python research/finetuning/run_cv.py `
        --config  configs/qlora_t4.yaml `
        --splits  data/splits `
        --outroot outputs/cv-qwen25-1p5b
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import List


HERE = Path(__file__).resolve().parent


def discover_folds(splits_dir: Path) -> List[Path]:
    folds = sorted(p for p in splits_dir.glob("fold_*") if p.is_dir())
    if not folds:
        raise SystemExit(f"[cv] no fold_* subdirectories found under {splits_dir}")
    return folds


def parse_test_accuracy(predictions_path: Path) -> float:
    """Replay the metric used by evaluate_sft.py to get accuracy from the
    per-example JSONL it wrote."""
    n = 0
    n_correct = 0
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1
            n_correct += int(bool(row.get("correct")))
    return n_correct / n if n else 0.0


def run(cmd: List[str]) -> None:
    print(f"[cv] $ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=HERE)
    if res.returncode != 0:
        raise SystemExit(f"[cv] subprocess failed (exit={res.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="Path to the QLoRA YAML config (relative to finetuning/).")
    ap.add_argument("--splits", required=True, type=Path, help="Splits directory produced by prepare_dataset.py --mode kfold.")
    ap.add_argument("--outroot", required=True, type=Path, help="Root directory for per-fold outputs.")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--skip-train", action="store_true", help="Skip training for ALL folds (use existing per-fold adapters).")
    ap.add_argument("--skip-eval",  action="store_true", help="Skip evaluation for ALL folds (just train).")
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="Only run the listed fold indices.")
    ap.add_argument(
        "--force-train",
        action="store_true",
        help="Re-run training for folds whose `final/` adapter already exists "
             "(by default such folds are skipped on rerun).",
    )
    ap.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-run evaluation for folds whose `test_predictions.jsonl` "
             "already exists (by default such folds are skipped and the "
             "cached accuracy is reused for the summary).",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Pass `--no-resume` through to train_sft.py so each fold trains "
             "from scratch instead of continuing from its last checkpoint.",
    )
    args = ap.parse_args()

    splits_dir = args.splits.resolve()
    test_file  = splits_dir / "test.jsonl"
    if not test_file.exists():
        raise SystemExit(f"[cv] missing held-out test file: {test_file}")

    fold_dirs = discover_folds(splits_dir)
    if args.folds is not None:
        wanted = set(args.folds)
        fold_dirs = [d for d in fold_dirs if int(d.name.split("_")[1]) in wanted]
        if not fold_dirs:
            raise SystemExit(f"[cv] no matching folds for --folds={args.folds}")

    args.outroot.mkdir(parents=True, exist_ok=True)

    per_fold = []
    for fold_dir in fold_dirs:
        k = int(fold_dir.name.split("_")[1])
        fold_out = (args.outroot / f"fold_{k}").resolve()
        fold_out.mkdir(parents=True, exist_ok=True)
        adapter_dir = fold_out / "final"
        preds_path  = fold_out / "test_predictions.jsonl"

        # ---- train (skip if `final/` already exists, unless --force-train) ----
        adapter_done = (adapter_dir / "adapter_config.json").exists()
        if args.skip_train:
            print(f"[cv] fold_{k}: --skip-train set, not training")
        elif adapter_done and not args.force_train:
            print(
                f"[cv] fold_{k}: adapter already at {adapter_dir} "
                "-> skipping training (pass --force-train to retrain)"
            )
        else:
            train_cmd = [
                sys.executable, "train_sft.py",
                "--config",     str(args.config),
                "--train-file", str(fold_dir / "train.jsonl"),
                "--val-file",   str(fold_dir / "val.jsonl"),
                "--output-dir", str(fold_out),
            ]
            if args.no_resume:
                train_cmd.append("--no-resume")
            # NOTE: when --no-resume is NOT set, train_sft.py auto-detects the
            # latest checkpoint-* inside `fold_out` and continues from exactly
            # that step (weights, optimizer, LR schedule, RNG, step/epoch).
            run(train_cmd)

        # ---- evaluate (skip if predictions already exist, unless --force-eval) ----
        if args.skip_eval:
            print(f"[cv] fold_{k}: --skip-eval set, not evaluating")
            continue

        if preds_path.exists() and not args.force_eval:
            acc = parse_test_accuracy(preds_path)
            print(
                f"[cv] fold_{k}: reusing cached predictions {preds_path} "
                f"-> test_accuracy = {acc:.4f}  (pass --force-eval to redo)"
            )
        else:
            if not (adapter_dir / "adapter_config.json").exists():
                print(
                    f"[cv] fold_{k}: WARNING adapter not found at {adapter_dir}, "
                    "skipping eval for this fold"
                )
                continue
            run([
                sys.executable, "evaluate_sft.py",
                "--adapter",    str(adapter_dir),
                "--base-model", args.base_model,
                "--test",       str(test_file),
                "--out",        str(preds_path),
                "--max-new-tokens", str(args.max_new_tokens),
            ])
            acc = parse_test_accuracy(preds_path)
            print(f"[cv] fold_{k}: test_accuracy = {acc:.4f}")

        per_fold.append({"fold": k, "accuracy": acc, "predictions": str(preds_path)})

    summary_path = args.outroot / "cv_summary.json"
    if per_fold:
        accs = [r["accuracy"] for r in per_fold]
        summary = {
            "folds":      per_fold,
            "mean":       statistics.mean(accs),
            "stdev":      statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "min":        min(accs),
            "max":        max(accs),
            "n_folds":    len(accs),
            "test_file":  str(test_file),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"\n[cv] mean accuracy = {summary['mean']:.4f} "
            f"+/- {summary['stdev']:.4f}  (n={summary['n_folds']})"
        )
        print(f"[cv] summary written to {summary_path}")


if __name__ == "__main__":
    main()
