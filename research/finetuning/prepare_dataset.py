"""Build train/val/test splits (or K-fold CV folds) from an SFT JSONL dataset.

Two modes:

  --mode split   (default): write `train.jsonl`, `val.jsonl`, `test.jsonl`
                 using `--train/--val/--test` ratios. Set `--val 0` for a
                 plain 80/20 train/test split (val will not be written).

  --mode kfold:  carve off a fixed held-out `test.jsonl` (`--test`, default
                 0.2 of the data), then partition the REMAINING data into
                 K equal-sized folds. For each fold k in [0, K-1] write
                     <outdir>/fold_k/train.jsonl   (k-1 folds combined)
                     <outdir>/fold_k/val.jsonl     (fold k)
                 The same held-out `test.jsonl` is shared across folds and
                 is used only for the FINAL evaluation.

If your JSONL has only `data_file` (no inline `table_markdown`), this script
also renders the xlsx/csv to Markdown on the fly via `data_preview.py`.

Examples:

  # Classic 70/10/20 split (legacy default)
  python prepare_dataset.py `
      --input  data/sft_dataset.jsonl `
      --outdir data/splits `
      --mode split --train 0.7 --val 0.1 --test 0.2 --seed 42

  # 80/20 (no val) for use with K-fold CV done elsewhere
  python prepare_dataset.py `
      --input  data/sft_dataset.jsonl `
      --outdir data/splits `
      --mode split --train 0.8 --val 0.0 --test 0.2 --seed 42

  # 5-fold CV with a 20% held-out test set
  python prepare_dataset.py `
      --input  data/sft_dataset.jsonl `
      --outdir data/splits `
      --mode kfold --k 5 --test 0.2 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_preview import get_data_preview_markdown  # noqa: E402


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ensure_table_markdown(rows: List[Dict[str, Any]], repo_root: Path) -> None:
    """If a record only has `data_file`, render it to `table_markdown` in-place."""
    for r in rows:
        if r.get("table_markdown"):
            continue
        df_path = r.get("data_file")
        if not df_path:
            raise ValueError(
                f"Record {r.get('index')} has neither table_markdown nor data_file"
            )
        p = Path(df_path)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"data_file does not exist: {p}")
        r["table_markdown"] = get_data_preview_markdown(p)


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------
def simple_split(
    rows: List[Dict[str, Any]],
    train: float,
    val: float,
    test: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError(
            f"train+val+test must sum to 1.0; got {train}+{val}+{test}={train+val+test}"
        )
    rng = random.Random(seed)
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    # Test gets the remainder so we don't drop/duplicate rows due to rounding.
    n_test = n - n_train - n_val
    if n_test < 0:
        # Shouldn't happen because ratios sum to 1, but guard anyway.
        n_train += n_test
        n_test = 0
    out = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train : n_train + n_val],
        "test":  shuffled[n_train + n_val :],
    }
    assert len(out["test"]) == n_test
    return out


def kfold_split(
    rows: List[Dict[str, Any]],
    k: int,
    test_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    """Carve off `test_ratio` for a held-out test set, then split the rest
    into K equal folds. Returns:
        {
          "test":  [...],
          "folds": [ {"train": [...], "val": [...]} for _ in range(K) ],
        }
    """
    if k < 2:
        raise ValueError(f"--k must be >= 2 for K-fold CV; got {k}")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"--test must be in [0, 1); got {test_ratio}")
    rng = random.Random(seed)
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(test_ratio * n))
    test_rows = shuffled[:n_test]
    cv_rows = shuffled[n_test:]
    if len(cv_rows) < k:
        raise ValueError(
            f"After removing test set ({n_test}/{n}), only {len(cv_rows)} rows "
            f"remain for {k}-fold CV - need at least {k}."
        )

    # Round-robin assignment so fold sizes differ by at most 1.
    folds: List[List[Dict[str, Any]]] = [[] for _ in range(k)]
    for i, r in enumerate(cv_rows):
        folds[i % k].append(r)

    cv_pairs = []
    for i in range(k):
        val_i = folds[i]
        train_i = [r for j, fold in enumerate(folds) if j != i for r in fold]
        cv_pairs.append({"train": train_i, "val": val_i})

    return {"test": test_rows, "folds": cv_pairs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, type=Path, help="Input .jsonl produced by build_sft_dataset.py")
    p.add_argument("--outdir", required=True, type=Path, help="Output directory for the splits")
    p.add_argument(
        "--mode",
        choices=["split", "kfold"],
        default="split",
        help="Splitting strategy (default: split).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Used to resolve relative `data_file` paths when rendering tables to markdown. "
             "Defaults to the current working directory.",
    )

    # --- split mode -----------------------------------------------------------
    p.add_argument("--train", type=float, default=0.7, help="(split mode) train fraction.")
    p.add_argument("--val",   type=float, default=0.1, help="(split mode) val fraction. Set 0 for no val file.")
    p.add_argument("--test",  type=float, default=0.2, help="(both modes) test fraction.")

    # --- kfold mode -----------------------------------------------------------
    p.add_argument("--k", type=int, default=5, help="(kfold mode) number of folds.")

    args = p.parse_args()

    rows = load_jsonl(args.input)
    print(f"[prep] loaded {len(rows)} rows from {args.input}")
    ensure_table_markdown(rows, args.repo_root)

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "split":
        # Allow val=0 for a plain train/test split.
        train, val, test = args.train, args.val, args.test
        if abs(train + val + test - 1.0) > 1e-6:
            raise SystemExit(
                f"[prep] train+val+test must sum to 1.0; got {train}+{val}+{test}={train+val+test}"
            )
        splits = simple_split(rows, train, val, test, args.seed)

        write_jsonl(splits["train"], args.outdir / "train.jsonl")
        if val > 0:
            write_jsonl(splits["val"], args.outdir / "val.jsonl")
        else:
            # Remove any stale val.jsonl from a previous run.
            stale = args.outdir / "val.jsonl"
            if stale.exists():
                stale.unlink()
        write_jsonl(splits["test"], args.outdir / "test.jsonl")

        print(
            f"[prep] split: train={len(splits['train'])}  "
            f"val={len(splits['val'])}  test={len(splits['test'])}"
        )
        print(f"[prep] wrote -> {args.outdir}")
        return

    # ---- kfold mode ----
    out = kfold_split(rows, args.k, args.test, args.seed)
    write_jsonl(out["test"], args.outdir / "test.jsonl")
    print(f"[prep] held-out test: {len(out['test'])}")
    for i, fold in enumerate(out["folds"]):
        fold_dir = args.outdir / f"fold_{i}"
        write_jsonl(fold["train"], fold_dir / "train.jsonl")
        write_jsonl(fold["val"],   fold_dir / "val.jsonl")
        print(f"[prep] fold_{i}: train={len(fold['train'])}  val={len(fold['val'])}")
    print(f"[prep] wrote {args.k} folds + test -> {args.outdir}")


if __name__ == "__main__":
    main()
