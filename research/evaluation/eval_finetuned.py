"""Compute accuracy on prediction jsonl files.

For each input jsonl (e.g. ``test_predictions_baseline.jsonl``), this script:

1. Calls a prose LLM (default ``dev-gpt-54-reasoning``) to re-extract the
   structured ``{"answer", "dtype"}`` envelope from each row's ``pred_raw``
   following the format defined in ``research/agents/output_format.py``.
2. Scores the extracted answer against ``gold_answer`` / ``gold_dtype`` using
   :func:`research.evaluation.evaluate.evaluate` (returns True/False).
3. Buckets rows into Original vs. Disturbed by looking them up in
   ``research/dataset/wikitq_dataset_filtered/{original,disturbed}.json``.
   Disturbed rows are further split into Semantic vs. Structural via the
   ``distortion_type`` field on the disturbed dataset.
4. Reports per-bucket accuracy as fractions and percentages, plus a
   "Scaled Original" accuracy: every disturbed test pulls in its parent
   original's correctness (so a single original counted once per disturbed
   variant). The scaled-original denominator therefore matches the number of
   disturbed rows whose parent original is present in the jsonl.

Inputs:
    --input  Either a single .jsonl file or a directory containing .jsonl
             files. When multiple files are processed, the script also
             reports mean ± stddev across files.

The extraction step is cached to ``<stem>.extracted.jsonl`` next to the
input file; subsequent runs reuse it (use ``--no-resume`` to redo).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tqdm import tqdm  # noqa: E402

from research.report.diversification_taxonomy import (  # noqa: E402
    map_diversification,
    ordered_groups,
)

from prose.llm import (  # noqa: E402
    ChatModel,
    ChatRequest,
    Message,
    Role,
    SubstrateClient,
)
from prose.llm.models import ModelSpecification, ModelSupports  # noqa: E402

from research.agents.output_format import get_response_format  # noqa: E402
from research.agents.utils.model_response import JsonResponseParser  # noqa: E402
from research.evaluation.evaluate import evaluate  # noqa: E402
from research.evaluation.utils import fix_json_serialization  # noqa: E402


DEFAULT_DATASET_DIR = ROOT_DIR / "research" / "dataset" / "wikitq_dataset_filtered"
DEFAULT_MODEL = "dev-gpt-54-reasoning"

EXTRACT_SYSTEM_PROMPT = (
    "You are an answer-extraction assistant. The user will give you the raw, "
    "free-form output of another model that was supposed to answer a table-QA "
    "query. Your job is to read that raw output and re-emit ONLY the final "
    "answer in the strict JSON format described below. Do not solve the "
    "question yourself -- only repackage what the other model already "
    "concluded. If the other model did not produce a clear final answer, "
    "make your best guess from what it wrote. Never include any commentary "
    "outside of the JSON block.\n\n"
    + get_response_format()
)


# --------------------------------------------------------------------------- #
# Dataset lookup tables                                                       #
# --------------------------------------------------------------------------- #

class DatasetIndex:
    """Lookups built from ``original.json`` and ``disturbed.json``."""

    def __init__(self, dataset_dir: Path):
        with (dataset_dir / "original.json").open("r", encoding="utf-8") as f:
            orig = json.load(f)
        with (dataset_dir / "disturbed.json").open("r", encoding="utf-8") as f:
            dist = json.load(f)

        self.orig_indices: set = {x["index"] for x in orig}
        # disturbed index -> distortion_type ("semantic" | "structural")
        self.dist_type: Dict[str, str] = {
            x["index"]: x.get("distortion_type") for x in dist
        }
        # disturbed index -> diversification_type (e.g. "vertical_shift")
        self.dist_div_type: Dict[str, str] = {
            x["index"]: x.get("diversification_type") for x in dist
        }
        self.dist_indices: set = set(self.dist_type)

        # Build dist_index -> original_index map.
        suffix = "__original__disturbed"

        def to_orig(idx: str) -> Optional[str]:
            if idx.endswith(suffix):
                return idx[: -len(suffix)]
            if idx in self.orig_indices:
                return idx
            cur = idx
            while "_" in cur:
                cur = cur.rsplit("_", 1)[0]
                if cur in self.orig_indices:
                    return cur
            return None

        self.dist_to_orig: Dict[str, Optional[str]] = {
            i: to_orig(i) for i in self.dist_indices
        }


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {ln} of {path}: {e}") from e
    return rows


def _extracted_path(pred_file: Path) -> Path:
    return pred_file.with_name(f"{pred_file.stem}.extracted{pred_file.suffix}")


# --------------------------------------------------------------------------- #
# Extraction                                                                  #
# --------------------------------------------------------------------------- #

def _build_user_message(record: Dict[str, Any]) -> str:
    query = record.get("query", "")
    pred_raw = record.get("pred_raw", "") or ""
    return (
        "## Original query\n"
        f"{query}\n\n"
        "## Raw model output to extract from\n"
        f"{pred_raw}\n\n"
        "Now return ONLY the JSON envelope per the system instructions."
    )


class ThrottlingError(Exception):
    """Raised when the LLM call fails due to rate limiting / throttling.

    Such rows must NOT be persisted to the extraction cache so that a later
    ``--resume`` run can retry them.
    """


_THROTTLE_HINTS = (
    "429",
    "rate limit",
    "ratelimit",
    "too many requests",
    "throttl",
    "quota",
    "overloaded",
    "timeout",
    "timed out",
    "service unavailable",
    "503",
)


def _is_throttling_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "throttl" in name or "timeout" in name:
        return True
    msg = str(exc).lower()
    return any(h in msg for h in _THROTTLE_HINTS)


def _extract_one(model: ChatModel, record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"answer": None, "dtype": None, "parse_ok": False}
    try:
        response = model.chat(
            [
                Message(role=Role.System, content=EXTRACT_SYSTEM_PROMPT),
                Message(role=Role.User, content=_build_user_message(record)),
            ],
            ChatRequest(max_completion_tokens=512, n=1),
        )
        out["raw"] = response.text
        parsed = JsonResponseParser._parse_raw_response(response.text)
        parsed = fix_json_serialization(parsed)
        out["answer"] = parsed.get("answer")
        out["dtype"] = parsed.get("dtype")
        out["parse_ok"] = out["dtype"] is not None
    except Exception as e:  # noqa: BLE001
        if _is_throttling_error(e):
            raise ThrottlingError(f"{type(e).__name__}: {e}") from e
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _score(extracted: Dict[str, Any], record: Dict[str, Any]) -> bool:
    if not extracted.get("parse_ok"):
        return False
    try:
        return bool(
            evaluate(
                gt_answer=record["gold_answer"],
                gt_dtype=record["gold_dtype"],
                pred_answer=extracted["answer"],
                pred_dtype=extracted["dtype"],
            )
        )
    except Exception:
        return False


def _row_key(rec: Dict[str, Any], occurrence: int) -> str:
    """Unique key for a jsonl row (index + occurrence number)."""
    return f"{rec.get('index')}#{occurrence}"


def _assign_occurrences(rows: List[Dict[str, Any]]) -> List[int]:
    """Return per-row occurrence counter (0-based) for handling duplicate indices."""
    seen: Dict[str, int] = {}
    occs: List[int] = []
    for r in rows:
        idx = r.get("index")
        n = seen.get(idx, 0)
        occs.append(n)
        seen[idx] = n + 1
    return occs


def _extract_all(
    records: List[Dict[str, Any]],
    model: ChatModel,
    out_path: Path,
    resume: bool,
    nproc: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Run prose extraction over all records, caching results to ``out_path``."""
    occs = _assign_occurrences(records)
    keyed = [(r, _row_key(r, o)) for r, o in zip(records, occs)]

    already_done: Dict[str, Dict[str, Any]] = {}
    if resume and out_path.exists():
        cached = _load_jsonl(out_path)
        for r in cached:
            k = r.get("_row_key")
            if k:
                already_done[k] = r
        print(f"  Resuming with {len(already_done)} cached rows.")
    elif out_path.exists():
        out_path.unlink()

    pending = [(r, k) for r, k in keyed if k not in already_done]
    print(f"  To extract: {len(pending)} / {len(records)}")

    write_lock = threading.Lock()
    throttle_counter = {"n": 0}

    def _work(rec_key: Tuple[Dict[str, Any], str]) -> Optional[Dict[str, Any]]:
        rec, key = rec_key
        try:
            extracted = _extract_one(model, rec)
        except ThrottlingError as e:
            with write_lock:
                throttle_counter["n"] += 1
            tqdm.write(
                f"[throttle] skipping {rec.get('index')} ({key}) -- {e}. "
                "Re-run with --resume to retry."
            )
            return None
        except Exception as e:  # noqa: BLE001
            print(f"[!] hard failure on {rec.get('index')}: {e}")
            print(traceback.format_exc())
            return None
        rescored = _score(extracted, rec)
        enriched = dict(rec)
        enriched["_row_key"] = key
        enriched["extracted"] = extracted
        enriched["rescored_correct"] = rescored
        return enriched

    with out_path.open("a", encoding="utf-8") as f:
        if nproc <= 1:
            for rk in tqdm(pending, desc=f"extract {out_path.name}"):
                enriched = _work(rk)
                if enriched is None:
                    continue
                with write_lock:
                    f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    f.flush()
                    already_done[enriched["_row_key"]] = enriched
        else:
            with ThreadPoolExecutor(max_workers=nproc) as ex:
                futs = {ex.submit(_work, rk): rk for rk in pending}
                for fut in tqdm(
                    as_completed(futs),
                    total=len(futs),
                    desc=f"extract {out_path.name}",
                ):
                    enriched = fut.result()
                    if enriched is None:
                        continue
                    with write_lock:
                        f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                        f.flush()
                        already_done[enriched["_row_key"]] = enriched

    # Preserve original row order.
    ordered: List[Dict[str, Any]] = []
    missing = 0
    for r, k in keyed:
        if k in already_done:
            ordered.append(already_done[k])
        else:
            missing += 1
            # Hard failure / throttled -- treat as parse-failure for reporting,
            # but the row is NOT persisted to disk so --resume will retry it.
            stub = dict(r)
            stub["_row_key"] = k
            stub["extracted"] = {"answer": None, "dtype": None, "parse_ok": False}
            stub["rescored_correct"] = False
            ordered.append(stub)
    if throttle_counter["n"]:
        print(
            f"  [!] {throttle_counter['n']} row(s) skipped due to throttling. "
            "Re-run with --resume (default) to retry them."
        )
    if missing and missing != throttle_counter["n"]:
        print(f"  [!] {missing - throttle_counter['n']} row(s) had non-throttle failures.")
    return ordered, throttle_counter["n"]


# --------------------------------------------------------------------------- #
# Bucketing                                                                   #
# --------------------------------------------------------------------------- #

def _bucket_rows(
    rows: List[Dict[str, Any]],
    ds: DatasetIndex,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split rows into (original_rows, disturbed_rows) lists, preserving order.

    Rules:
      * If index is only in ``disturbed.json`` -> disturbed.
      * If index is only in ``original.json`` -> original.
      * If index is in both: allocate by occurrence -- first row to original,
        subsequent rows to disturbed (matching the canonical dataset order).
      * Otherwise (unknown index) -> disturbed (treated as worst-case for
        original accuracy; matches behavior of legacy extract_and_eval.py).
    """
    orig_bucket: List[Dict[str, Any]] = []
    dist_bucket: List[Dict[str, Any]] = []
    occs = _assign_occurrences(rows)
    for rec, n in zip(rows, occs):
        idx = rec.get("index")
        in_o = idx in ds.orig_indices
        in_d = idx in ds.dist_indices
        if in_d and not in_o:
            dist_bucket.append(rec)
        elif in_o and not in_d:
            orig_bucket.append(rec)
        elif in_o and in_d:
            (orig_bucket if n == 0 else dist_bucket).append(rec)
        else:
            dist_bucket.append(rec)
    return orig_bucket, dist_bucket


def _split_by_distortion(
    dist_rows: List[Dict[str, Any]], ds: DatasetIndex
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sem: List[Dict[str, Any]] = []
    struct: List[Dict[str, Any]] = []
    for r in dist_rows:
        dt = ds.dist_type.get(r.get("index"))
        if dt == "semantic":
            sem.append(r)
        elif dt == "structural":
            struct.append(r)
    return sem, struct


# --------------------------------------------------------------------------- #
# Accuracy computation                                                        #
# --------------------------------------------------------------------------- #

def _frac(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    n = len(rows)
    c = sum(1 for r in rows if bool(r.get("rescored_correct", False)))
    return c, n


def _scaled_original(
    orig_rows: List[Dict[str, Any]],
    dist_rows: List[Dict[str, Any]],
    ds: DatasetIndex,
) -> Tuple[int, int]:
    """For each disturbed row, look up its parent original. If that original is
    present in ``orig_rows``, add its correctness to the numerator and 1 to the
    denominator. Disturbed rows whose parent original is missing from the jsonl
    are skipped (they cannot contribute a weighted original)."""
    by_idx: Dict[str, Dict[str, Any]] = {r.get("index"): r for r in orig_rows}
    num = 0
    den = 0
    for d in dist_rows:
        parent = ds.dist_to_orig.get(d.get("index"))
        if parent is None:
            # Fall back to the disturbed index itself (handles indices shared
            # between original and disturbed datasets).
            parent = d.get("index")
        if parent in by_idx:
            den += 1
            if bool(by_idx[parent].get("rescored_correct", False)):
                num += 1
    return num, den


def _report_file(
    name: str,
    rows: List[Dict[str, Any]],
    ds: DatasetIndex,
    detailed: bool = False,
) -> Dict[str, Tuple[int, int]]:
    orig_rows, dist_rows = _bucket_rows(rows, ds)
    sem_rows, struct_rows = _split_by_distortion(dist_rows, ds)

    buckets: Dict[str, Tuple[int, int]] = {
        "Original": _frac(orig_rows),
        "Disturbed": _frac(dist_rows),
        "Semantic": _frac(sem_rows),
        "Structural": _frac(struct_rows),
        "Scaled Original": _scaled_original(orig_rows, dist_rows, ds),
        "Overall": _frac(rows),
    }

    col_w = 18
    print(f"\n=== {name} ===")
    print(f"{'Bucket':<{col_w}}{'Count':>14}{'Accuracy':>14}")
    print("-" * (col_w + 28))
    for label, (c, n) in buckets.items():
        frac = f"{c}/{n}"
        pct = f"{(c / n * 100) if n else 0.0:.2f}%"
        print(f"{label:<{col_w}}{frac:>14}{pct:>14}")

    if detailed:
        # Per-diversification_type breakdown on the disturbed bucket only.
        from collections import defaultdict
        dv_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in dist_rows:
            raw = ds.dist_div_type.get(r.get("index"))
            dv = map_diversification(raw)
            dv_buckets[dv].append(r)
        if dv_buckets:
            print()
            print("By diversification_type (disturbed only):")
            col_w2 = 28
            print(f"{'diversification_type':<{col_w2}}{'Count':>14}{'Accuracy':>14}")
            print("-" * (col_w2 + 28))
            for dv in ordered_groups(dv_buckets.keys()):
                c, n = _frac(dv_buckets[dv])
                frac = f"{c}/{n}"
                pct = f"{(c / n * 100) if n else 0.0:.2f}%"
                print(f"{dv:<{col_w2}}{frac:>14}{pct:>14}")
            # Stash detailed fractions on the returned dict so aggregation
            # can compute mean ± stddev across files as well.
            for dv, bucket_rows in dv_buckets.items():
                buckets[f"div::{dv}"] = _frac(bucket_rows)
    return buckets


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def _collect_inputs(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.iterdir()
                       if p.is_file()
                       and p.suffix == ".jsonl"
                       and not p.stem.endswith(".extracted"))
        if not files:
            raise SystemExit(f"No .jsonl files in directory {path}")
        return files
    raise SystemExit(f"Input path not found: {path}")


def _aggregate(per_file: Dict[str, Dict[str, Tuple[int, int]]], detailed: bool = False) -> None:
    """Print mean ± stddev across multiple files."""
    bucket_labels = ["Original", "Disturbed", "Semantic", "Structural",
                     "Scaled Original", "Overall"]
    file_names = list(per_file)
    print("\n" + "=" * 70)
    print(f"Aggregate across {len(file_names)} files")
    print("=" * 70)
    col_w = 18
    print(f"{'Bucket':<{col_w}}{'mean%':>12}{'stddev%':>12}{'n_files':>10}")
    print("-" * (col_w + 34))
    for label in bucket_labels:
        pcts: List[float] = []
        for fname in file_names:
            c, n = per_file[fname][label]
            if n:
                pcts.append(c / n * 100)
        if not pcts:
            continue
        m = statistics.fmean(pcts)
        sd = statistics.stdev(pcts) if len(pcts) > 1 else 0.0
        print(f"{label:<{col_w}}{m:>11.2f}%{sd:>11.2f}%{len(pcts):>10d}")

    if not detailed:
        return
    div_labels_raw = {
        k for buckets in per_file.values() for k in buckets if k.startswith("div::")
    }
    if not div_labels_raw:
        return
    div_labels = [
        "div::" + g for g in ordered_groups({k[len("div::"):] for k in div_labels_raw})
    ]
    print()
    print("By diversification_type (disturbed only):")
    col_w2 = 28
    print(f"{'diversification_type':<{col_w2}}{'mean%':>12}{'stddev%':>12}{'n_files':>10}")
    print("-" * (col_w2 + 34))
    for label in div_labels:
        pcts: List[float] = []
        for fname in file_names:
            cn = per_file[fname].get(label)
            if not cn:
                continue
            c, n = cn
            if n:
                pcts.append(c / n * 100)
        if not pcts:
            continue
        m = statistics.fmean(pcts)
        sd = statistics.stdev(pcts) if len(pcts) > 1 else 0.0
        print(f"{label[len('div::'):]:<{col_w2}}{m:>11.2f}%{sd:>11.2f}%{len(pcts):>10d}")


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", "-i", type=Path, required=True,
                   help="A .jsonl file or a directory containing .jsonl files.")
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                   help="Directory with original.json / disturbed.json.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Prose LLM model name (default: {DEFAULT_MODEL}).")
    p.add_argument("--nproc", type=int, default=8,
                   help="Concurrent extraction workers.")
    p.add_argument("--no-resume", action="store_true",
                   help="Recompute extractions even if a cache exists.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N rows per file (debugging).")
    p.add_argument("--detailed", action="store_true",
                   help="Also show accuracy broken down per diversification_type "
                        "(mapped from disturbed.json).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    inputs = _collect_inputs(args.input)
    ds = DatasetIndex(args.dataset_dir)

    model_spec = ModelSpecification(
        args.model, ModelSupports.Chat | ModelSupports.Completion
    )
    model = ChatModel(model_spec, SubstrateClient(), suppress=True)

    per_file: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for pred_file in inputs:
        print(f"\n>>> {pred_file}")
        records = _load_jsonl(pred_file)
        if args.limit:
            records = records[: args.limit]
        out_path = _extracted_path(pred_file)
        rows, throttled = _extract_all(
            records, model, out_path,
            resume=not args.no_resume,
            nproc=args.nproc,
        )
        if throttled:
            print(
                f"  Skipping accuracy report for {pred_file.name}: "
                f"{throttled} row(s) throttled and not persisted. "
                "Re-run with --resume to complete."
            )
            continue
        per_file[pred_file.name] = _report_file(
            pred_file.name, rows, ds, detailed=args.detailed
        )

    if len(per_file) > 1:
        _aggregate(per_file, detailed=args.detailed)


if __name__ == "__main__":
    main()
