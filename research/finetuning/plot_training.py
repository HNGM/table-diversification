"""Plot training curves from `training_log.jsonl` files.

Each `train_sft.py` run streams its log dicts to
    <output_dir>/training_log.jsonl
This script reads one or many of those files and produces:
  * Train loss vs step
  * Eval loss vs step (if present)
  * Learning-rate schedule vs step
  * Grad norm vs step (if present)

For K-fold CV (where each fold writes its own training_log.jsonl), point
`--log` at the parent directory (e.g. `outputs/cv-qwen25-1p5b`) and the script
will overlay one curve per fold AND compute mean+/-stdev shaded bands.

Outputs PNG (and PDF if `--pdf` is set) into `--outdir` so they are ready to
drop straight into a paper.

Examples:
    # Single run
    python research/finetuning/plot_training.py `
        --log outputs/qwen25-1p5b-disturbed/training_log.jsonl `
        --outdir outputs/qwen25-1p5b-disturbed/plots

    # All CV folds (auto-discovers fold_*/training_log.jsonl)
    python research/finetuning/plot_training.py `
        --log outputs/cv-qwen25-1p5b `
        --outdir outputs/cv-qwen25-1p5b/plots --pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Lazy / soft import so missing matplotlib gives a clean error.
try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(
        "matplotlib is required for plotting. Install with:\n"
        "    pip install matplotlib"
    )

try:
    import numpy as np
except ImportError:
    np = None  # only needed for the CV mean+/-stdev bands


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_log(path: Path) -> List[Dict]:
    """Load training_log.jsonl, keeping only the LAST record per global_step
    (handles duplicates introduced by checkpoint resumes)."""
    by_step: Dict[int, Dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = rec.get("step")
            if step is None:
                continue
            # Merge so an eval_loss record and a train-loss record at the same
            # step are unioned rather than overwriting each other.
            existing = by_step.get(step, {})
            existing.update(rec)
            by_step[step] = existing
    return [by_step[s] for s in sorted(by_step.keys())]


def discover_runs(log_arg: Path) -> List[tuple[str, Path]]:
    """Return [(label, path_to_training_log.jsonl)]."""
    if log_arg.is_file():
        return [(log_arg.parent.name or "run", log_arg)]
    # Directory: look for direct file then fold_*/training_log.jsonl.
    direct = log_arg / "training_log.jsonl"
    if direct.exists():
        return [(log_arg.name, direct)]
    found = []
    for fold_dir in sorted(log_arg.glob("fold_*")):
        cand = fold_dir / "training_log.jsonl"
        if cand.exists():
            found.append((fold_dir.name, cand))
    if not found:
        raise SystemExit(f"[plot] no training_log.jsonl found under {log_arg}")
    return found


def series(rows: List[Dict], key: str) -> tuple[List[float], List[float]]:
    xs, ys = [], []
    for r in rows:
        if key in r and isinstance(r[key], (int, float)):
            xs.append(r.get("step", 0))
            ys.append(float(r[key]))
    return xs, ys


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _save(fig, outdir: Path, name: str, also_pdf: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.png", dpi=200)
    if also_pdf:
        fig.savefig(outdir / f"{name}.pdf")
    plt.close(fig)
    print(f"[plot] wrote {outdir / (name + '.png')}")


def plot_metric(
    runs: List[tuple[str, List[Dict]]],
    metric: str,
    ylabel: str,
    title: str,
    outdir: Path,
    log_y: bool,
    also_pdf: bool,
    show_mean: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    all_x, all_y = [], []
    plotted_any = False
    for label, rows in runs:
        xs, ys = series(rows, metric)
        if not xs:
            continue
        plotted_any = True
        ax.plot(xs, ys, label=label, alpha=0.7 if show_mean and len(runs) > 1 else 1.0)
        all_x.append(xs)
        all_y.append(ys)
    if not plotted_any:
        plt.close(fig)
        print(f"[plot] no records for metric={metric}, skipping")
        return

    if show_mean and len(runs) > 1 and np is not None:
        # Resample each run onto a shared step grid (union of all steps) by
        # linear interpolation, then compute mean +/- stdev across runs.
        union = sorted({s for xs in all_x for s in xs})
        grid = np.array(union, dtype=float)
        stacked = []
        for xs, ys in zip(all_x, all_y):
            if len(xs) < 2:
                continue
            stacked.append(np.interp(grid, np.array(xs), np.array(ys)))
        if stacked:
            arr = np.vstack(stacked)
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            ax.plot(grid, mean, color="black", linewidth=2.0, label="mean")
            ax.fill_between(grid, mean - std, mean + std, color="black", alpha=0.15, label="+/- stdev")

    ax.set_xlabel("optimizer step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    _save(fig, outdir, metric, also_pdf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, type=Path,
                    help="Path to a training_log.jsonl OR a directory containing one or more (e.g. CV outroot).")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--pdf", action="store_true", help="Also save PDFs (paper-ready).")
    ap.add_argument("--no-mean", action="store_true",
                    help="When plotting multiple runs, don't overlay the mean +/- stdev band.")
    ap.add_argument("--log-y", action="store_true", help="Use log scale on the y axis for loss plots.")
    args = ap.parse_args()

    pairs = discover_runs(args.log)
    runs = [(label, load_log(path)) for label, path in pairs]
    nz = [(l, r) for l, r in runs if r]
    if not nz:
        raise SystemExit("[plot] all logs are empty -- has training produced any logs yet?")
    print(f"[plot] {len(nz)} run(s):")
    for label, rows in nz:
        print(f"  - {label}: {len(rows)} log records")

    show_mean = len(nz) > 1 and not args.no_mean

    plot_metric(nz, "loss",          "training loss",       "Training loss",                args.outdir, args.log_y, args.pdf, show_mean)
    plot_metric(nz, "eval_loss",     "validation loss",     "Validation loss",              args.outdir, args.log_y, args.pdf, show_mean)
    plot_metric(nz, "learning_rate", "learning rate",       "Learning-rate schedule",       args.outdir, False,      args.pdf, show_mean)
    plot_metric(nz, "grad_norm",     "gradient norm",       "Gradient norm",                args.outdir, args.log_y, args.pdf, show_mean)
    print(f"[plot] all charts saved to {args.outdir}")


if __name__ == "__main__":
    main()
