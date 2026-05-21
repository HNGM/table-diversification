"""Build a summary .xlsx with two sheets (``original`` and ``disturbed``) from
result file paths listed in ``results.txt``.

Result file naming convention (from ``research/evaluation/prose_llm_main.py``):
    {dataset}_{data_mode}_{prompt_mode}_{ingest_mode}_{model}.json

- ``data_mode``   -> ``original`` or ``disturbed`` (determines target sheet).
- ``prompt_mode`` -> e.g. ``default``, ``mistake``, ``default_no_sandbox``,
                     ``mistake_no_sandbox`` (may contain underscores).
- ``ingest_mode`` -> ``markdown``, ``screenshot``, ``none``.
- ``model``       -> remainder of the file stem.

Columns produced:

original sheet:
    file_name | model | prompt_mode | ingest_mode | accuracy (%) | sample_size

disturbed sheet:
    file_name | model | prompt_mode | ingest_mode |
    accuracy (%) | semantic_accuracy (%) | structural_accuracy (%) |
    sample_size | semantic_n | structural_n

Aggregation logic re-uses ``research/report/agg_score.py`` (including the
``wikitq_dataset_`` scaling by disturbed variants).

Usage:
    python research/report/build_summary_xlsx.py \
        [--results-txt results.txt] [--output research/results/summary.xlsx]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Optional

import pandas as pd

# Make repo root importable.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.utils import ROOT_DIR, read_json  # noqa: E402
from research.report.agg_score import (  # noqa: E402
    WIKITQ_ORIGINAL_PREFIX,
    _build_query_to_disturbed,
    _extract_eval,
)


DATA_MODES = ["original", "disturbed"]
INGEST_MODES = ["markdown", "screenshot", "none"]


def parse_filename(file_name: str) -> Optional[dict]:
    """Parse a result file's base name into its components.

    Pattern: ``{dataset}_{data_mode}_{prompt_mode}_{ingest_mode}_{model}.json``
    """
    base = os.path.basename(file_name)
    stem, _ext = os.path.splitext(base)
    tokens = stem.split("_")

    # Locate data_mode (original/disturbed).
    dm_idx = next((i for i, t in enumerate(tokens) if t in DATA_MODES), None)
    if dm_idx is None:
        return None

    # Locate ingest_mode after data_mode (prompt_mode lives between them and
    # can contain underscores, e.g. ``default_no_sandbox``).
    im_idx = next(
        (i for i in range(dm_idx + 2, len(tokens)) if tokens[i] in INGEST_MODES),
        None,
    )
    if im_idx is None:
        return None

    dataset = "_".join(tokens[:dm_idx])
    data_mode = tokens[dm_idx]
    prompt_mode = "_".join(tokens[dm_idx + 1:im_idx])
    ingest_mode = tokens[im_idx]
    model = "_".join(tokens[im_idx + 1:])
    if not prompt_mode or not model:
        return None

    return {
        "dataset": dataset,
        "data_mode": data_mode,
        "prompt_mode": prompt_mode,
        "ingest_mode": ingest_mode,
        "model": model,
        "file": file_name,
        "base": base,
    }


def compute_overall(dataset, scale_map=None):
    """Return (accuracy_pct, total_samples)."""
    total = 0
    success = 0
    for data in dataset:
        is_success, _ = _extract_eval(data)
        weight = 1 if scale_map is None else len(scale_map.get(data.get("query"), []))
        if weight == 0:
            continue
        total += weight
        if is_success:
            success += weight
    acc = (success / total * 100) if total else 0.0
    return acc, total


def compute_by_distortion(dataset, scale_map=None):
    """Return dict[distortion_type] = (accuracy_pct, total_samples)."""
    buckets: dict = defaultdict(lambda: {"total": 0, "success": 0})
    if scale_map is not None:
        for data in dataset:
            is_success, _ = _extract_eval(data)
            for variant in scale_map.get(data.get("query"), []):
                dt = variant.get("distortion_type", "unknown")
                buckets[dt]["total"] += 1
                if is_success:
                    buckets[dt]["success"] += 1
    else:
        for data in dataset:
            dt = data.get("distortion_type", "unknown")
            is_success, _ = _extract_eval(data)
            buckets[dt]["total"] += 1
            if is_success:
                buckets[dt]["success"] += 1
    return {
        dt: ((b["success"] / b["total"] * 100) if b["total"] else 0.0, b["total"])
        for dt, b in buckets.items()
    }


def _load(file_path: str):
    p = file_path
    if not os.path.isabs(p) and not os.path.exists(p):
        p = os.path.join(str(ROOT_DIR), file_path)
    return read_json(p)


def _scale_map_for(meta: dict):
    # Scaling (counting each original query as many times as it has disturbed
    # variants) is only meaningful for ``original`` result files of the
    # ``wikitq_dataset_*`` family. Disturbed result files are already one row
    # per variant, so they must NOT be scaled.
    if meta["data_mode"] != "original":
        return None
    if not meta["dataset"].startswith(WIKITQ_ORIGINAL_PREFIX.rstrip("_")):
        return None
    return _build_query_to_disturbed()


def build_original_row(meta: dict) -> Optional[dict]:
    try:
        data = _load(meta["file"])
    except Exception as e:
        print(f"[warn] failed to read {meta['file']}: {e}")
        return None
    scale_map = _scale_map_for(meta)
    acc, n = compute_overall(data, scale_map=scale_map)
    return {
        "file_name": meta["file"],
        "model": meta["model"],
        "prompt_mode": meta["prompt_mode"],
        "ingest_mode": meta["ingest_mode"],
        "sample_size": n,
        "accuracy (%)": round(acc, 2),
    }


def build_disturbed_row(meta: dict) -> Optional[dict]:
    try:
        data = _load(meta["file"])
    except Exception as e:
        print(f"[warn] failed to read {meta['file']}: {e}")
        return None
    scale_map = _scale_map_for(meta)
    all_acc, all_n = compute_overall(data, scale_map=scale_map)
    per = compute_by_distortion(data, scale_map=scale_map)
    sem_acc, sem_n = per.get("semantic", (0.0, 0))
    str_acc, str_n = per.get("structural", (0.0, 0))
    return {
        "file_name": meta["file"],
        "model": meta["model"],
        "prompt_mode": meta["prompt_mode"],
        "ingest_mode": meta["ingest_mode"],
        "sample_size": all_n,
        "accuracy (%)": round(all_acc, 2),
        "semantic_n": sem_n,
        "semantic_accuracy (%)": round(sem_acc, 2),
        "structural_n": str_n,
        "structural_accuracy (%)": round(str_acc, 2),
    }


def read_results_txt(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    metas = []
    for ln in lines:
        meta = parse_filename(ln)
        if meta is None:
            print(f"[warn] could not parse filename: {ln}")
            continue
        metas.append(meta)
    return metas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-txt",
        type=str,
        default=str(ROOT_DIR / "results.txt"),
        help="Path to text file containing one results JSON path per line.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path of the .xlsx file to write "
            "(default: research/results/summary_<timestamp>.xlsx)."
        ),
    )
    args = parser.parse_args()

    out_path = args.output
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d")
        out_path = str(ROOT_DIR / "research" / "results" / f"summary_{ts}.xlsx")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    metas = read_results_txt(args.results_txt)
    if not metas:
        print("No valid result files parsed from results.txt; nothing to do.")
        return

    original_rows = []
    disturbed_rows = []
    for meta in metas:
        if meta["data_mode"] == "original":
            row = build_original_row(meta)
            if row is not None:
                original_rows.append(row)
        elif meta["data_mode"] == "disturbed":
            row = build_disturbed_row(meta)
            if row is not None:
                disturbed_rows.append(row)

    original_cols = [
        "file_name", "model", "prompt_mode", "ingest_mode",
        "sample_size", "accuracy (%)",
    ]
    disturbed_cols = [
        "file_name", "model", "prompt_mode", "ingest_mode",
        "sample_size", "accuracy (%)",
        "semantic_n", "semantic_accuracy (%)",
        "structural_n", "structural_accuracy (%)",
    ]

    df_original = pd.DataFrame(original_rows, columns=original_cols)
    df_disturbed = pd.DataFrame(disturbed_rows, columns=disturbed_cols)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_original.to_excel(writer, sheet_name="original", index=False)
        df_disturbed.to_excel(writer, sheet_name="disturbed", index=False)

    print(f"Wrote summary to: {out_path}")
    print(f"  original sheet rows : {len(df_original)}")
    print(f"  disturbed sheet rows: {len(df_disturbed)}")


if __name__ == "__main__":
    main()
