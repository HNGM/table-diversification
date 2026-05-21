"""Build the SFT training dataset for the finetuning pipeline.

Reads a table benchmark JSON (default:
`research/dataset/wikitq_dataset_filtered/disturbed.json`) and, for every
record, uses `prose.llm` to generate ONE concise natural-language reasoning
paragraph (2-3 sentences) that:

  - For DISTORTED tables: notices the inconsistency mid-reasoning, mentions
    how it is mentally corrected, and continues to the answer.
  - For UNDISTORTED tables: reasons straight through to the answer with no
    fabricated distortion talk.

The output (default: `research/finetuning/data/sft_dataset.jsonl`) is a JSONL
file in EXACTLY the schema the finetuning pipeline expects:

    {
      "index": ..., "query": ..., "table_markdown": ...,
      "is_distorted": true|false,
      "distortion_type": ..., "diversification_type": ...,
      "reasoning": "<2-3 sentence integrated reasoning>",
      "answer": ..., "dtype": ...
    }

You can mix distorted + undistorted records by passing multiple --input files;
records whose source file path / name indicates `original` or that carry no
`distortion_type` are auto-flagged as undistorted (or set `is_distorted` in
the source records explicitly).

Run:
    python research/finetuning/build_sft_dataset.py \
        --input research/dataset/wikitq_dataset_filtered/disturbed.json \
        --input research/dataset/wikitq_dataset_filtered/original.json \
        --nproc 4 --resume
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

# --- repo imports (this is a one-time data-prep script run from the repo) ---
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prose.llm import ChatModel, ChatRequest, Message, Role, SubstrateClient  # noqa: E402
from prose.llm.models import ModelSpecification, ModelSupports  # noqa: E402

from src.utils.data_preview import get_data_preview_markdown  # noqa: E402
from research.agents.utils.model_response import JsonResponseParser  # noqa: E402
from tqdm import tqdm  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT = REPO_ROOT / "research" / "dataset" / "wikitq_dataset_filtered" / "disturbed.json"
DEFAULT_OUTPUT = REPO_ROOT / "research" / "finetuning" / "data" / "sft_dataset.jsonl"
DEFAULT_MODEL = "dev-anthropic-claude-opus-4-6"


# ---------------------------------------------------------------------------
# Distortion definitions (per diversification_type).
# Only the SINGLE definition matching the record's `diversification_type` is
# injected into the prompt - dumping the full taxonomy biases the labeler
# (and confuses small models downstream).
# ---------------------------------------------------------------------------
DISTORTION_DEFINITIONS: Dict[str, str] = {
    # ----- semantic -----
    "ocr_char_misinterpret": """\
## Distortion: ocr_char_misinterpret (semantic)
Character-level substitutions introduced during OCR or noisy parsing
(0<->O, 1<->l<->I, 5<->S, 8<->B). Examples: `2O24` instead of `2024`,
`1O0.5` instead of `100.5`, `S123` instead of `5123`. Look for alphanumeric
tokens that violate the expected datatype of their column and check whether
replacing the ambiguous character restores semantic validity.""",

    "decimal_separator_swap": """\
## Distortion: decimal_separator_swap (semantic)
Decimal and thousands separators are interchanged or use incompatible
regional formats. Examples: population `23.456` instead of `23,456`,
temperature `23,412`, currency `1.000,50`. Validate magnitudes against the
column's real-world range and check whether swapping `.` and `,` restores
plausible values.""",

    "random_noise_chars": """\
## Distortion: random_noise_chars (semantic)
Random symbols or extraneous characters injected into otherwise valid
values. Examples: `~1.23`, `2#45`, `1@000`, `USD$#120`. Identify
unexpected special characters inside structured values and verify whether
removing the non-semantic characters restores valid formatting.""",

    "label_mismatch": """\
## Distortion: label_mismatch (semantic)
Column header does not semantically correspond to the values in the column.
Examples: a `Diastolic_BP` column consistently containing values larger
than `Systolic_BP`; a `Temperature (°C)` column holding values in the
range of human weight; a `Volume` column containing currency-like values.
Compare column statistics / ranges against the header's expected semantics
and look for inversions where values systematically contradict the label.""",

    "date_format_corruption": """\
## Distortion: date_format_corruption (semantic)
Date values become ambiguous, malformed, or incorrectly converted.
Examples: `01/02/2023` ambiguous between Jan 2 and Feb 1, Excel serial
numbers like `45123` leaking into a date column, mixed formats like
`2024-01-05` and `05/01/24` within the same column. Detect inconsistent
formats, impossible dates, serial-number leakage, and chronological
violations.""",

    # ----- structural -----
    "vertical_shift": """\
## Distortion: vertical_shift (structural)
A subset of columns is vertically displaced relative to the rest of the
table, so values appear under the wrong rows. Examples: population values
appearing under the wrong city, dates shifted downward by one row.
Detection: sudden datatype discontinuities inside a column, missing values
at the top of a column with orphan values at the bottom, neighbouring
columns no longer aligning semantically.""",

    "horizontal_shift": """\
## Distortion: horizontal_shift (structural)
A subset of rows is horizontally displaced across columns, so row-wise
relationships break. Examples: product IDs appearing under `ProductName`,
numeric fields landing under categorical columns, an entire row shifted
by one column. Detection: rows whose values do not match the expected
column datatypes, and whose consistency is restored after a left/right
shift.""",

    "merge_cells": """\
## Distortion: merge_cells (structural)
Cells that should remain separate are merged together, or merged cells are
incorrectly split. Common in multi-line headers, hierarchical tables, and
spreadsheet exports. Examples: `"Revenue Profit"` appearing in one header
cell; a single category spanning multiple columns being flattened.
Detection: unusually long concatenated cell values, header rows spanning
multiple semantic fields.""",

    "broken_rows_merge": """\
## Distortion: broken_rows_merge (structural)
Two or more consecutive rows are incorrectly merged into a single row.
Examples: two patient records appearing in one row; multiple dates or IDs
embedded inside a single entry. Detection: rows containing multiple
independent entity patterns, abnormally long row contents, or repeated
schema fragments within one row.""",

    "broken_rows_split": """\
## Distortion: broken_rows_split (structural)
A single logical row is fragmented across multiple physical rows.
Examples: address fields split across rows; multi-column records
fragmented into two partial rows. Detection: incomplete rows followed by
continuation rows, rows with missing identifiers but continuation-like
content.""",

    "multi_column_collapse": """\
## Distortion: multi_column_collapse (structural)
Values from adjacent columns collapse into a single cell or column.
Examples: `"JohnSalesManager"` instead of separate Name and Role;
`"2024-01-01$1200"` instead of separate date and amount columns.
Detection: concatenated values lacking delimiters, missing neighbouring
columns accompanied by unusually dense text.""",

    "footnote_injection": """\
## Distortion: footnote_injection (structural)
Textual notes, metadata, or annotations from outside the original table
are injected as table rows. Examples: `"Note: values updated after audit"`
appearing as a data row; source citations inserted into the middle of the
table; summary statistics embedded inside transactional rows. Detection:
rows that violate the table schema entirely, explanatory text embedded
among structured records, footer-like patterns.""",
}

# Aliases for legacy / alternate spellings that may appear in older datasets.
_DIVERSIFICATION_ALIASES: Dict[str, str] = {
    "ocr_character_misinterpretation": "ocr_char_misinterpret",
    "random_noise_character_injection": "random_noise_chars",
    "merge_cells_distortion": "merge_cells",
    "broken_row_merge": "broken_rows_merge",
    "broken_row_split": "broken_rows_split",
}


def _get_distortion_definition(diversification_type: Optional[str]) -> str:
    """Return the single definition block matching `diversification_type`.

    Falls back to a short generic note if the type is missing or unknown,
    so the labeler still has SOME guidance without being shown the whole
    taxonomy.
    """
    key = (diversification_type or "").strip().lower()
    key = _DIVERSIFICATION_ALIASES.get(key, key)
    if key in DISTORTION_DEFINITIONS:
        return DISTORTION_DEFINITIONS[key]
    return (
        "## Distortion: (unspecified)\n"
        "The table is distorted but the specific subtype was not provided. "
        "Inspect the table carefully for inconsistencies in headers, value "
        "types, alignment, or row/column structure."
    )


SYSTEM_PROMPT = (
    "You generate concise reasoning traces for a table-QA finetuning dataset. "
    "You write the way a careful analyst would think OUT LOUD while solving "
    "the question, following a small state machine:\n"
    "  SCAN  -> briefly orient on the table that matters for the question\n"
    "  LOCATE -> zero in on the relevant rows/columns/cells\n"
    "  COMPUTE -> derive the value(s) needed for the answer\n"
    "  CONCLUDE -> state the answer\n"
    "Two OPTIONAL states may fire along the way IF and ONLY IF the table "
    "actually shows the issue:\n"
    "  FLAG -> 'something here doesn't look right' (concrete cells/values)\n"
    "  REPAIR -> a brief mental fix, then continue\n"
    "FLAG can fire from SCAN (the oddity is obvious at first glance: orphan "
    "rows, mismatched header semantics, pervasive OCR noise, footer rows "
    "embedded as data, etc.) OR from COMPUTE (a value you actually need "
    "doesn't make sense in context and you correct it on the spot).\n"
    "Hard rules:\n"
    "  - Do NOT fabricate anomalies. If the table looks fine for THIS "
    "question, skip FLAG/REPAIR entirely and reason straight through.\n"
    "  - When you DO flag, cite concrete column names / cell values from THIS "
    "table - never just name the distortion category.\n"
    "  - Output is one flowing paragraph (2-3 sentences), no headers, no "
    "bullet points, no state labels in the prose."
)


# Shared trailing instructions describing the planning + writing protocol.
_PLAN_AND_WRITE_INSTRUCTIONS = """\
# Your task
First, INTERNALLY decide which trajectory fits this specific sample:
  - "upfront"         -> something is clearly off the moment you look at the
                         table; flag during SCAN, repair, then compute.
  - "mid_computation" -> the table looks ordinary at first; only while
                         pulling the value(s) you need does a cell stop
                         making sense, and you fix it on the spot.
  - "none"            -> nothing in the table that matters for THIS question
                         looks wrong; reason straight through.

Then write ONE natural-language paragraph (2-3 sentences) following that
trajectory. Cite concrete column names / cell values from THIS table.
Do NOT use section headers, bullet points, or state names like "SCAN" or
"FLAG" in the prose - just think out loud.

Respond with a single JSON object in a ```json``` code block:

```json
{{
  "detection_point": "upfront" | "mid_computation" | "none",
  "anomaly_evidence": "<one short clause naming the concrete cells/values that triggered FLAG, or \\"\\" if detection_point is \\"none\\">",
  "reasoning": "<the 2-3 sentence paragraph>"
}}
```
"""


USER_TEMPLATE_DISTORTED = """\
{distortion_definition}

# This sample (DISTORTED)
- distortion_type (category): {distortion_type}
- diversification_type (subtype): {diversification_type}

## Table (markdown)
{table_markdown}

## Question
{query}

## Ground-truth answer
{gold_answer}  (dtype: {gold_dtype})

# Guidance for THIS sample
This table is known to contain a distortion of the kind described above.
However, do NOT mechanically announce the distortion - decide naturally
whether a careful reader would catch it UPFRONT (because the oddity jumps
out the moment they look at the table) or MID-COMPUTATION (because the
oddity only matters once they try to read the value the question needs).

Examples of UPFRONT-style trajectories:
  - pervasive OCR noise across many cells
  - obvious orphan rows at the bottom from a vertical shift
  - a column whose values systematically contradict its header
  - a footnote-looking row sitting among real data

Examples of MID_COMPUTATION-style trajectories:
  - a single decimal_separator_swap on the one value you need to compare
  - a date that only looks wrong when you try to order it against others
  - a horizontal_shift that only matters for the specific row you're reading

Good example (upfront, vertical shift):
  "Looking at the table, the `Population` column is clearly off - the first
  two cells are blank and two orphan numbers sit below the last named
  township, so I slide those values up by two rows. With that realignment,
  the populations greater than 2,000 belong to four townships."

Good example (mid-computation, decimal swap):
  "Filtering for the Eibsee tramway row, its Height reads `65 m`, which is
  plausible, but the adjacent value `1.13,6 m` for the previous row can't
  be a real height - reading it as `113.6 m` makes it consistent with the
  rest of the column, and the smallest height in the table remains 65 m."

Bad example (avoid - rote, generic, no concrete evidence):
  "Distortion detected: vertical shift. Repair: realign. Answer: 4."

If, after looking carefully, the distortion does NOT actually affect the
cells the question depends on, it is acceptable to set
`detection_point="mid_computation"` and flag only the specific value you
needed to repair (do not invent unrelated repairs).

{plan_and_write}"""


USER_TEMPLATE_CLEAN = """\
# This sample (CLEAN - no distortion)

## Table (markdown)
{table_markdown}

## Question
{query}

## Ground-truth answer
{gold_answer}  (dtype: {gold_dtype})

# Guidance for THIS sample
This table is clean. `detection_point` MUST be `"none"` and
`anomaly_evidence` MUST be `""`. The paragraph must NOT mention
distortions, repairs, OCR, shifts, mismatches, or anything "looking off" -
just reason straight through SCAN -> LOCATE -> COMPUTE -> CONCLUDE and
derive {gold_answer} from concrete columns / values in the table.

{plan_and_write}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_existing_indices(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[resume] WARNING: skipping malformed line {line_no} in {path}: {e}")
                continue
            idx = obj.get("index")
            if idx is not None:
                seen.add(idx)
    return seen


def _gold_to_str(answer: Any) -> str:
    if isinstance(answer, str):
        return answer
    try:
        return json.dumps(answer, ensure_ascii=False)
    except Exception:
        return str(answer)


def _is_distorted_record(info: Dict[str, Any]) -> bool:
    """Decide whether a source record represents a distorted or clean table.

    Priority:
      1. Explicit `is_distorted` field if present.
      2. `distortion_type` field is non-empty and not 'original'/'none'.
      3. `data_file` path contains a hint like 'distorted'.
    """
    if "is_distorted" in info:
        return bool(info["is_distorted"])
    dt = (info.get("distortion_type") or "").strip().lower()
    if dt and dt not in {"original", "none", "clean", ""}:
        return True
    div = (info.get("diversification_type") or "").strip().lower()
    if div and div not in {"original", "none", "clean", ""}:
        return True
    data_file = str(info.get("data_file") or "").lower()
    if "distorted" in data_file or "disturbed" in data_file:
        return True
    return False


def _build_record(
    info: Dict[str, Any],
    model: ChatModel,
    max_attempts: int = 3,
) -> Optional[Dict[str, Any]]:
    """Render markdown, call the model, parse, and return a finalized record."""
    data_file = info["data_file"]
    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path

    try:
        table_md = get_data_preview_markdown(data_path)
    except Exception as e:
        print(f"[skip] {info.get('index')}: failed to render markdown ({e})")
        return None

    is_distorted = _is_distorted_record(info)

    if is_distorted:
        user_prompt = USER_TEMPLATE_DISTORTED.format(
            distortion_definition=_get_distortion_definition(info.get("diversification_type")),
            distortion_type=info.get("distortion_type") or "(unknown)",
            diversification_type=info.get("diversification_type") or "(unknown)",
            table_markdown=table_md,
            query=info["query"],
            gold_answer=_gold_to_str(info["answer"]),
            gold_dtype=info.get("dtype", ""),
            plan_and_write=_PLAN_AND_WRITE_INSTRUCTIONS,
        )
    else:
        user_prompt = USER_TEMPLATE_CLEAN.format(
            table_markdown=table_md,
            query=info["query"],
            gold_answer=_gold_to_str(info["answer"]),
            gold_dtype=info.get("dtype", ""),
            plan_and_write=_PLAN_AND_WRITE_INSTRUCTIONS,
        )

    messages = [
        Message(role=Role.System, content=SYSTEM_PROMPT),
        Message(role=Role.User, content=user_prompt),
    ]

    parsed: Optional[Dict[str, Any]] = None
    last_err = None
    valid_dp = {"upfront", "mid_computation", "none"}
    for attempt in range(1, max_attempts + 1):
        try:
            response = model.chat(
                messages,
                ChatRequest(max_completion_tokens=512, n=1, temperature=0.0),
            )
            parsed = JsonResponseParser._parse_raw_response(response.text)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
            reasoning = parsed.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError("Missing or empty `reasoning` field")
            dp = (parsed.get("detection_point") or "").strip().lower()
            if dp not in valid_dp:
                raise ValueError(
                    f"`detection_point` must be one of {sorted(valid_dp)}, got {dp!r}"
                )
            if not is_distorted and dp != "none":
                raise ValueError(
                    f"Clean sample must have detection_point='none', got {dp!r}"
                )
            parsed["detection_point"] = dp
            break
        except Exception as e:
            last_err = e
            parsed = None
            continue

    if parsed is None:
        print(f"[skip] {info.get('index')}: LLM parse failed after {max_attempts} attempts ({last_err})")
        return None

    # Keep answer JSON-decoded when possible so the finetuning pipeline can
    # re-serialize it cleanly.
    answer = info["answer"]
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except Exception:
            pass

    return {
        "index": info["index"],
        "query": info["query"],
        "table_markdown": table_md,
        "data_file": info.get("data_file"),
        "is_distorted": is_distorted,
        "distortion_type": info.get("distortion_type"),
        "diversification_type": info.get("diversification_type"),
        "reasoning": parsed["reasoning"].strip(),
        "detection_point": parsed["detection_point"],
        "anomaly_evidence": (parsed.get("anomaly_evidence") or "").strip(),
        "answer": answer,
        "dtype": info.get("dtype"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: List[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, action="append", default=None,
                   help=f"Source benchmark JSON. Can be passed multiple times "
                        f"to mix distorted + clean datasets. "
                        f"(default: {DEFAULT_INPUT})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Destination JSONL (default: {DEFAULT_OUTPUT})")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help=f"prose.llm model name (default: {DEFAULT_MODEL})")
    p.add_argument("--nproc", type=int, default=1,
                   help="Number of parallel LLM workers (threads)")
    p.add_argument("--resume", action="store_true",
                   help="Skip indices already present in --output")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="LLM parse retries per record")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N records (debugging)")
    args = p.parse_args(argv)

    if not args.input:
        args.input = [DEFAULT_INPUT]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # --- load source(s) ----------------------------------------------------
    records: List[Dict[str, Any]] = []
    for inp in args.input:
        with inp.open("r", encoding="utf-8") as f:
            chunk = json.load(f)
        # Tag records with their source so we can flag clean vs distorted by
        # filename when the record itself does not say.
        for r in chunk:
            r.setdefault("_source_file", str(inp))
            if "is_distorted" not in r:
                name = inp.name.lower()
                if "original" in name or "clean" in name:
                    r["is_distorted"] = False
                elif "distorted" in name or "disturbed" in name:
                    r["is_distorted"] = True
        records.extend(chunk)
    n_dist = sum(1 for r in records if _is_distorted_record(r))
    print(f"[input] {len(records)} records total ({n_dist} distorted, "
          f"{len(records)-n_dist} clean) from {len(args.input)} file(s)")

    done: set[str] = set()
    if args.resume:
        done = _read_existing_indices(args.output)
        print(f"[resume] {len(done)} records already in {args.output}; skipping them")

    remaining = [r for r in records if r.get("index") not in done]
    if args.limit:
        remaining = remaining[: args.limit]
    print(f"[work] {len(remaining)} records to process")

    if not remaining:
        print("[done] nothing to do")
        return

    # --- model -------------------------------------------------------------
    model_spec = ModelSpecification(args.model, ModelSupports.Chat | ModelSupports.Completion)
    model = ChatModel(model_spec, SubstrateClient(), suppress=True)

    # --- output stream (append-mode, line-buffered) ------------------------
    out_lock = Lock()
    out_f = args.output.open("a", encoding="utf-8")

    def _close_out():
        try:
            out_f.flush()
            out_f.close()
        except Exception:
            pass

    atexit.register(_close_out)

    def _handle_signal(signum, frame):
        print(f"\n[signal {signum}] flushing and exiting")
        _close_out()
        sys.exit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            pass

    def _write(rec: Dict[str, Any]) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        with out_lock:
            out_f.write(line + "\n")
            out_f.flush()

    # --- process -----------------------------------------------------------
    try:
        if args.nproc <= 1:
            for info in tqdm(remaining, desc="building SFT"):
                try:
                    rec = _build_record(info, model, max_attempts=args.max_attempts)
                except Exception as e:
                    print(f"[error] {info.get('index')}: {e}")
                    print(traceback.format_exc())
                    continue
                if rec is not None:
                    _write(rec)
        else:
            with ThreadPoolExecutor(max_workers=args.nproc) as pool:
                fut_to_idx = {
                    pool.submit(_build_record, info, model, args.max_attempts): info.get("index")
                    for info in remaining
                }
                for fut in tqdm(as_completed(fut_to_idx), total=len(fut_to_idx), desc="building SFT"):
                    idx = fut_to_idx[fut]
                    try:
                        rec = fut.result()
                    except Exception as e:
                        print(f"[error] {idx}: {e}")
                        print(traceback.format_exc())
                        continue
                    if rec is not None:
                        _write(rec)
    finally:
        _close_out()

    print(f"[done] wrote -> {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
