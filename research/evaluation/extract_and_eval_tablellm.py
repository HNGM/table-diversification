"""Extract structured answers from TableLLM-style ``code_response_log`` outputs
and score them with :func:`research.evaluation.evaluate.evaluate`.

Input
-----
A JSON file like ``research/results/table_finetuned/tablellm_original.json``:
a list of items each shaped roughly like::

    {
        "index": ...,
        "query": ...,
        "answer": <gold>,
        "dtype":  <gold dtype>,
        ...,
        "eval": [
            {
                "code_response_log": "<stdout of the executed code>",
                "exit_code": 0,
                ...
            },
            ...
        ]
    }

For each ``eval`` attempt we ask a prose LLM to read the query + the
``code_response_log`` and produce the strict JSON envelope defined in
``research/agents/output_format.py`` (``{"answer": ..., "dtype": ...}``),
then score that against the gold answer via ``evaluate()``.

Output
------
A sibling JSON file ``<stem>.extracted_eval.json`` in the same directory,
with the same structure as the input plus, on each attempt:

    "extracted":  {answer, dtype, parse_ok, raw, [error]}
    "rescored":   bool

Per-item, also adds ``"rescored_pass": bool`` (True iff any attempt scored
correct), and prints overall accuracy.

Usage
-----
    python -m research.evaluation.extract_and_eval_tablellm \
        --in-file research/results/table_finetuned/tablellm_original.json \
        [--model dev-gpt-54-reasoning] [--nproc 4] [--resume]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tqdm import tqdm  # noqa: E402

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
from research.report.diversification_taxonomy import (  # noqa: E402
    map_diversification,
    ordered_groups,
)


DEFAULT_IN_FILE = (
    ROOT_DIR / "research" / "results" / "table_finetuned" / "tablellm_original.json"
)
DEFAULT_MODEL = "dev-gpt-54-reasoning"

# Used to scale accuracy on "*original*" inputs so the denominator matches
# the number of disturbed variants (same logic as research/report/agg_score.py).
WIKITQ_DISTURBED_DATASET = (
    ROOT_DIR / "research" / "dataset" / "wikitq_dataset_filtered" / "disturbed.json"
)

EXTRACT_SYSTEM_PROMPT = (
    "You are an answer-extraction assistant. The user gives you a table-QA "
    "query together with the raw stdout printed by a Python program that was "
    "executed to answer the query. Your job is to repackage what that "
    "program printed into the strict JSON format described below. Do not "
    "solve the problem yourself -- only convert what the program output "
    "already shows. If the program output is empty, an error, or clearly "
    "unrelated, make a best-effort guess from whatever signal it provides. "
    "Never include any commentary outside of the JSON block.\n\n"
    + get_response_format()
)


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #

def _default_out_path(in_file: Path) -> Path:
    return in_file.with_name(f"{in_file.stem}.extracted_eval{in_file.suffix}")


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    return data


# --------------------------------------------------------------------------- #
# Extraction                                                                  #
# --------------------------------------------------------------------------- #

def _build_user_message(query: str, log: str, exit_code: Any) -> str:
    log = log if log is not None else ""
    return (
        "## Query\n"
        f"{query}\n\n"
        f"## Program stdout (exit_code={exit_code})\n"
        f"{log}\n\n"
        "Return ONLY the JSON envelope per the system instructions."
    )


# --------------------------------------------------------------------------- #
# Throttling detection                                                        #
# --------------------------------------------------------------------------- #

class ThrottlingError(Exception):
    """Raised when the LLM call fails due to rate limiting / throttling.

    Items hitting this are NOT persisted, so a subsequent ``--resume`` run can
    retry them cleanly.
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


def _extract_one(model: ChatModel, query: str, attempt: Dict[str, Any]) -> Dict[str, Any]:
    """Call the prose LLM on a single ``eval`` attempt."""
    out: Dict[str, Any] = {"answer": None, "dtype": None, "parse_ok": False}
    log = attempt.get("code_response_log") or ""
    exit_code = attempt.get("exit_code")
    try:
        # NB: some reasoning models reject custom temperature -- omit it.
        response = model.chat(
            [
                Message(role=Role.System, content=EXTRACT_SYSTEM_PROMPT),
                Message(role=Role.User, content=_build_user_message(query, log, exit_code)),
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


def _score(extracted: Dict[str, Any], gold_answer: Any, gold_dtype: str) -> bool:
    if not extracted.get("parse_ok"):
        return False
    try:
        return bool(evaluate(
            gt_answer=gold_answer,
            gt_dtype=gold_dtype,
            pred_answer=extracted["answer"],
            pred_dtype=extracted["dtype"],
        ))
    except Exception:
        return False


def _is_failed_execution(attempt: Dict[str, Any]) -> bool:
    """True if the executed program errored out (no usable stdout to parse).

    A non-zero ``exit_code`` is the canonical signal; we also treat an empty
    log or a log that obviously looks like a Python traceback as a failure.
    """
    exit_code = attempt.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return True
    log = attempt.get("code_response_log")
    if log is None:
        return True
    if isinstance(log, str):
        stripped = log.strip()
        if not stripped:
            return True
        if "Traceback (most recent call last)" in stripped:
            return True
    return False


def _process_item(model: ChatModel, item: Dict[str, Any]) -> Dict[str, Any]:
    """Re-score the last attempt (``eval[-1]``) of ``item``.

    If the program execution failed, short-circuit: don't call the prose
    LLM, just mark extraction as ``None`` and ``rescored=False``.
    """
    new_item = dict(item)
    attempts = item.get("eval") or []
    new_attempts: List[Dict[str, Any]] = [dict(a) for a in attempts]

    if not new_attempts:
        new_item["eval"] = new_attempts
        new_item["rescored_pass"] = False
        return new_item

    last = new_attempts[-1]
    if _is_failed_execution(last):
        last["extracted"] = {
            "answer": None,
            "dtype": None,
            "parse_ok": False,
            "skipped_reason": "code execution failed",
        }
        last["rescored"] = False
    else:
        extracted = _extract_one(model, item.get("query", ""), last)
        last["extracted"] = extracted
        last["rescored"] = _score(
            extracted, item.get("answer"), item.get("dtype")
        )

    new_item["eval"] = new_attempts
    new_item["rescored_pass"] = bool(last.get("rescored"))
    return new_item


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def _process_all(
    items: List[Dict[str, Any]],
    model: ChatModel,
    nproc: int,
    already_done: Dict[str, Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], int]:
    """Process ``items`` (skipping any whose index is in ``already_done``).

    Returns ``(ordered_results, throttled_count)``. Throttled items are NOT
    included in ``ordered_results`` so they can be retried on the next run.
    """
    pending = [it for it in items if it.get("index") not in already_done]
    print(f"To process: {len(pending)} / {len(items)} "
          f"({len(already_done)} already in output).")

    results: Dict[str, Dict[str, Any]] = dict(already_done)
    lock = threading.Lock()
    throttle_counter = {"n": 0}

    def _work(it: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            return _process_item(model, it)
        except ThrottlingError as e:
            with lock:
                throttle_counter["n"] += 1
            tqdm.write(
                f"[throttle] skipping {it.get('index')} -- {e}. "
                "Re-run with --resume to retry."
            )
            return None
        except Exception as e:  # noqa: BLE001
            print(f"[!] failure on {it.get('index')}: {e}")
            print(traceback.format_exc())
            return None

    if nproc <= 1:
        for it in tqdm(pending, desc="extract"):
            res = _work(it)
            if res is not None:
                with lock:
                    results[res["index"]] = res
    else:
        with ThreadPoolExecutor(max_workers=nproc) as ex:
            futs = {ex.submit(_work, it): it for it in pending}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="extract"):
                res = fut.result()
                if res is not None:
                    with lock:
                        results[res["index"]] = res

    if throttle_counter["n"]:
        print(
            f"[!] {throttle_counter['n']} item(s) skipped due to throttling. "
            "Re-run with --resume to retry."
        )

    # Preserve input order in the output list.
    ordered: List[Dict[str, Any]] = []
    seen = set()
    for it in items:
        idx = it.get("index")
        if idx in results and idx not in seen:
            ordered.append(results[idx])
            seen.add(idx)
    # Include any leftovers (shouldn't happen, but be defensive).
    for idx, res in results.items():
        if idx not in seen:
            ordered.append(res)
    return ordered, throttle_counter["n"]


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #

def _attempt_orig_eval(att: Dict[str, Any]) -> Optional[bool]:
    """Best-effort extraction of the originally-reported correctness flag."""
    for key in ("eval", "correct", "is_correct"):
        if key in att:
            v = att[key]
            if isinstance(v, bool):
                return v
    return None


def _build_query_to_disturbed_count(path: Path) -> Dict[Any, int]:
    """query -> number of disturbed.json variants sharing that query.

    Mirrors ``research/report/agg_score.py::_build_query_to_disturbed``.
    """
    from collections import defaultdict
    weights: Dict[Any, int] = defaultdict(int)
    try:
        with path.open("r", encoding="utf-8") as f:
            for d in json.load(f):
                weights[d.get("query")] += 1
    except FileNotFoundError:
        pass
    return weights


def _build_index_to_div_type(path: Path) -> Dict[str, str]:
    """index -> diversification_type from disturbed.json."""
    out: Dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for d in json.load(f):
                idx = d.get("index")
                if idx is not None:
                    out[idx] = d.get("diversification_type") or "unknown"
    except FileNotFoundError:
        pass
    return out


def _report(items: List[Dict[str, Any]], in_file: Path, detailed: bool = False) -> None:
    n = len(items)
    if not n:
        print("No items to report on.")
        return

    # One score per item: the last attempt's re-eval result.
    correct = 0
    parse_ok = 0
    skipped = 0
    for it in items:
        attempts = it.get("eval") or []
        if not attempts:
            continue
        att = attempts[-1]
        ex = att.get("extracted") or {}
        if ex.get("parse_ok"):
            parse_ok += 1
        if ex.get("skipped_reason"):
            skipped += 1
        if bool(att.get("rescored")):
            correct += 1

    pct = (correct / n * 100) if n else 0.0
    parse_pct = (parse_ok / n * 100) if n else 0.0
    print(f"Items                : {n}")
    print(f"Skipped (exec failed): {skipped}/{n}")
    print(f"Extraction parse_ok  : {parse_ok}/{n}  ({parse_pct:.2f}%)")
    print(f"Accuracy (re-eval)   : {correct}/{n}  ({pct:.2f}%)")

    # Per-distortion_type breakdown (only when the key is present).
    from collections import defaultdict
    dt_buckets: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    have_dt = False
    for it in items:
        if "distortion_type" not in it:
            continue
        have_dt = True
        dt = it.get("distortion_type") or "unknown"
        attempts = it.get("eval") or []
        if not attempts:
            continue
        dt_buckets[dt]["total"] += 1
        if bool(attempts[-1].get("rescored")):
            dt_buckets[dt]["correct"] += 1

    if have_dt and dt_buckets:
        print()
        print("By distortion_type:")
        # Stable order: structural, semantic first, then any others alphabetically.
        preferred = ["structural", "semantic"]
        ordered = [d for d in preferred if d in dt_buckets] + sorted(
            d for d in dt_buckets if d not in preferred
        )
        for dt in ordered:
            b = dt_buckets[dt]
            t = b["total"]
            c = b["correct"]
            p = (c / t * 100) if t else 0.0
            print(f"  {dt.capitalize():<12}: {c}/{t}  ({p:.2f}%)")

    if detailed:
        # Per-diversification_type breakdown. Prefer the field already present
        # on the item; fall back to mapping its index via disturbed.json.
        div_map = _build_index_to_div_type(WIKITQ_DISTURBED_DATASET)
        dv_buckets: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "correct": 0}
        )
        have_dv = False
        for it in items:
            dv_raw = it.get("diversification_type")
            if dv_raw is None:
                dv_raw = div_map.get(it.get("index"))
            if dv_raw is None:
                continue
            dv = map_diversification(dv_raw)
            have_dv = True
            attempts = it.get("eval") or []
            if not attempts:
                continue
            dv_buckets[dv]["total"] += 1
            if bool(attempts[-1].get("rescored")):
                dv_buckets[dv]["correct"] += 1
        if have_dv and dv_buckets:
            print()
            print("By diversification_type (grouped):")
            for dv in ordered_groups(dv_buckets.keys()):
                b = dv_buckets[dv]
                t = b["total"]
                c = b["correct"]
                p = (c / t * 100) if t else 0.0
                print(f"  {dv:<28}: {c}/{t}  ({p:.2f}%)")
        elif not have_dv:
            print()
            print("(detailed: no diversification_type info available -- "
                  "items have no `diversification_type` and none of their "
                  f"indices were found in {WIKITQ_DISTURBED_DATASET.name})")

    # Scaled accuracy: only applicable when the input is the *original*
    # split. Each item is weighted by the number of disturbed variants
    # sharing its `query`, so the denominator equals the (matched) count
    # of disturbed variants -- same logic as ``agg_score.py``.
    if "original" not in in_file.name.lower():
        return

    weight_map = _build_query_to_disturbed_count(WIKITQ_DISTURBED_DATASET)
    if not weight_map:
        print("\n(scaled accuracy skipped: "
              f"could not load {WIKITQ_DISTURBED_DATASET})")
        return

    total_w = 0
    succ_w = 0
    unmatched_items = 0
    for it in items:
        w = weight_map.get(it.get("query"), 0)
        if w == 0:
            unmatched_items += 1
            continue
        total_w += w
        attempts = it.get("eval") or []
        if attempts and bool(attempts[-1].get("rescored")):
            succ_w += w

    if total_w == 0:
        print("\n(scaled accuracy skipped: no items matched disturbed.json by query)")
        return

    s_pct = succ_w / total_w * 100
    print()
    print(f"Scaled accuracy      : {succ_w}/{total_w}  ({s_pct:.2f}%)")
    print("  (each item weighted by #disturbed variants sharing its query,")
    print("   matching research/report/agg_score.py)")
    if unmatched_items:
        print(f"  Items with no matching disturbed query: {unmatched_items}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in-file", type=Path, default=DEFAULT_IN_FILE,
                   help=f"Input JSON file (default: {DEFAULT_IN_FILE})")
    p.add_argument("--out-file", type=Path, default=None,
                   help="Output JSON path (default: <stem>.extracted_eval.json "
                        "next to the input).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Prose LLM model name (default: {DEFAULT_MODEL}).")
    p.add_argument("--nproc", type=int, default=1,
                   help="Concurrent extraction workers.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse already-processed items found in --out-file.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N input items (debugging).")
    p.add_argument("--detailed", action="store_true",
                   help="Also show accuracy broken down per diversification_type "
                        "(mapped from disturbed.json).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    in_file: Path = args.in_file
    out_file: Path = args.out_file or _default_out_path(in_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    items = _load_json(in_file)
    if args.limit:
        items = items[: args.limit]

    already_done: Dict[str, Dict[str, Any]] = {}
    if args.resume and out_file.exists():
        try:
            prev = _load_json(out_file)
            already_done = {it["index"]: it for it in prev if "index" in it}
        except Exception as e:  # noqa: BLE001
            print(f"Could not load prior output {out_file}: {e}. Starting fresh.")
    elif out_file.exists():
        out_file.unlink()

    print(f"Input  : {in_file}  ({len(items)} items)")
    print(f"Output : {out_file}")
    print(f"Model  : {args.model}    nproc={args.nproc}    resume={args.resume}")

    model_spec = ModelSpecification(
        args.model, ModelSupports.Chat | ModelSupports.Completion
    )
    model = ChatModel(model_spec, SubstrateClient(), suppress=True)

    results, throttled = _process_all(items, model, args.nproc, already_done)
    _atomic_write_json(out_file, results)

    print()
    if throttled:
        print(
            f"Skipping accuracy report: {throttled} item(s) throttled and not "
            "persisted. Re-run with --resume to complete."
        )
        return
    _report(results, in_file, detailed=args.detailed)


if __name__ == "__main__":
    main()
