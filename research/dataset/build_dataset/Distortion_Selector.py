"""
Distortion Selector
====================
An LLM-powered agent that analyses a table together with a question/answer
pair and decides which distortions (from Distortion_Generator.py) would
prevent an automated table-QA solver from directly deriving the answer
from the distorted table, while still preserving the underlying
information so that a solver that first *reverses* ("de-distorts") the
corruption can still arrive at the correct answer.

The selector:
  1. Reads the table and serialises it into a compact Markdown string.
  2. Analyses the question to understand which columns, rows, data-types
     and structural features are involved in answering it.
  3. For every candidate distortion, reasons about whether it would
     (a) still preserve the underlying information needed for the answer,
     (b) make it impossible to directly derive the answer without first
         reversing the distortion (i.e. require "de-distortion"), and
     (c) be reversible — a smart solver that detects and undoes the
         distortion can still recover the correct answer.
  4. Returns a ranked list of suitable distortions with explanations.

Usage (CLI):
    python Distortion_Selector.py --input csv/200-csv/3.csv \
        --question "who won in 1973?" --answer "Secretariat"

    python Distortion_Selector.py --input data.xlsx \
        --question "how many rows have value > 50?" --answer "3" \
        --model gpt-4o

Usage (Library):
    from Distortion_Selector import review_distortions
    result = review_distortions(df, question, answer)
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any
import pandas as pd
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Distortion catalogue – must stay in sync with Distortion_Generator.py
# (Distortion_Selector selects which distortions are suitable)
# ---------------------------------------------------------------------------

DISTORTION_CATALOGUE: dict[str, str] = {
    # 1. Structural
    "merge_cells": (
        "Merge random adjacent header cells so that column names become "
        "ambiguous or disappear.  Hinders column look-up."
    ),
    "horizontal_shift": (
        "Shift data in random rows to the right by one position so values "
        "land under wrong column headers.  Corrupts row-level reads."
    ),
    "vertical_shift": (
        "Shift one or more entire columns (header + values) downward by "
        "1-4 rows so the top cells become empty.  Breaks row-level "
        "alignment between the shifted columns and the rest of the table."
    ),
    "broken_rows_split": (
        "Split random rows in half – left-side values on one row, "
        "right-side values on the next.  Breaks row integrity."
    ),
    "broken_rows_merge": (
        "Concatenate two adjacent rows into a single row so their cell "
        "values are joined.  Corrupts individual cell look-ups."
    ),
    # 2. OCR
    "ocr_char_misinterpret": (
        "Replace characters with common OCR look-alikes (0↔O, 1↔l, 5↔S, "
        "8↔B, 2↔Z).  Corrupts text matching and numeric parsing."
    ),
    "ocr_lost_formatting": (
        "Strip formatting symbols (%, $, €, £, etc.) from every cell.  "
        "Removes visual cues about data meaning."
    ),
    # 3. Data Type
    "numbers_as_text": (
        "Prefix and Suffix all numeric cells with an apostrophe so they are stored as "
        "text.  Breaks numeric comparisons and aggregations."
    ),
    "date_format_corruption": (
        "Rewrite date-like values in a random different format (DD/MM ↔ "
        "MM/DD, YYYY-MM-DD, or Excel serial number).  Creates ambiguity."
    ),
    "decimal_separator_swap": (
        "Swap '.' and ',' in numeric cells (e.g. 1,200.50 → 1.200,50).  "
        "Breaks numeric parsing due to locale mismatch."
    ),
    # 4. Header & Schema
    "header_as_data_row": (
        "Push the real header into the first data row and replace column "
        "names with generic labels (Col_0, Col_1 …).  Makes column "
        "identification much harder."
    ),
    # 5. Complex Layout
    "multi_column_collapse": (
        "Merge two adjacent columns into one space-separated column "
        "(e.g. 'Sales' + 'Growth' → 'Sales Growth').  Simulates column-"
        "boundary detection failure."
    ),
    "footnote_injection": (
        "Insert fake footnote/side-note rows (e.g. '* See appendix') "
        "into the table body.  Adds noise rows that confuse row counting "
        "and filtering."
    ),
    # 6. Image Quality
    "random_noise_chars": (
        "Insert random junk characters (#, @, ?, etc.) into numerical "
        "cells.  Corrupts numeric parsing and exact-match searches."
    ),
    # 7. Semantic
    "context_loss": (
        "Strip unit substrings ($, kg, %, etc.) from numeric cells.  "
        "Values lose their semantic meaning."
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_input(filepath: str) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    p = Path(filepath)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(filepath, dtype=str, keep_default_na=False)
    elif p.suffix.lower() in (".xls", ".xlsx"):
        return pd.read_excel(filepath, dtype=str, keep_default_na=False)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 60) -> str:
    """Serialise a DataFrame to a compact Markdown table string."""
    if len(df) > max_rows:
        top = df.head(max_rows // 2)
        bot = df.tail(max_rows // 2)
        ellipsis_row = pd.DataFrame(
            [["…"] * len(df.columns)], columns=df.columns
        )
        df = pd.concat([top, ellipsis_row, bot], ignore_index=True)
    return df.to_markdown(index=False)


def _build_catalogue_block() -> str:
    """Format the distortion catalogue into a numbered reference for the
    system prompt."""
    lines = []
    for i, (key, desc) in enumerate(DISTORTION_CATALOGUE.items(), 1):
        lines.append(f"{i:>2}. **{key}** – {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""\
You are an expert Table-QA Distortion Selector.

You will receive:
  • A table (Markdown format).
  • A question that can be answered from the table.
  • The correct answer.

Your job is to select distortions from the catalogue below that satisfy
ALL THREE of the following criteria simultaneously:

  A. **Blocks direct answering** – After the distortion is applied, an
     automated QA solver reading the distorted table CANNOT directly
     derive the correct answer.  The distortion must actively interfere
     with the lookup, comparison, aggregation, or matching step that
     the solver would need to perform, so that a naïve read of the
     distorted table yields a wrong or unparseable result.

  B. **Preserves underlying information** – The raw information needed
     to answer the question is still present somewhere in the distorted
     table (possibly garbled, shifted, merged, reformatted, etc.).
     Nothing critical is deleted or made permanently unrecoverable.

  C. **Answerable after de-distortion** – A smart solver that first
     detects the type of distortion and reverses ("de-distorts") it can
     reconstruct the original data and arrive at the correct answer.
     The distortion must therefore be *reversible* in principle.

In short: the distortion must FORCE the solver to clean / de-distort
the table before it can answer, but cleaning must be feasible.

### Distortion Catalogue
{_build_catalogue_block()}

### Instructions
1. First, briefly analyse the question and identify:
   - Which columns and rows are relevant to finding the answer.
   - What data types matter (numbers, dates, text, positions).
   - Whether the question relies on ordering, aggregation, filtering,
     or exact text matching.

2. Then, for EACH of the 15 distortions, decide:
   - "suitable"   → the distortion blocks the solver from directly
                     answering, yet the information is preserved and
                     the answer can be recovered after de-distortion.
   - "unsuitable" → the distortion fails at least one criterion:
                     (a) it has NO meaningful effect on answering
                         (solver can still answer directly), OR
                     (b) it irreversibly destroys / removes information
                         needed for the answer, OR
                     (c) it cannot be feasibly reversed to recover the
                         correct answer.
   Give a one-sentence reason for each decision.

3. Finally, return your response as a JSON object (and nothing else)
   with this exact schema:
   ```json
   {{
     "analysis": "<brief analysis of the question>",
     "distortions": {{
       "<distortion_name>": {{
         "verdict": "suitable" | "unsuitable",
         "reason": "<one-sentence explanation>"
       }},
       ...all 15 distortions...
     }},
     "recommended": ["<list of suitable distortion names ranked from most to least impactful>"]
   }}
   ```

Return ONLY the JSON object, no extra commentary.
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _call_llm(
    table_md: str,
    question: str,
    answer: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Send the table + question + answer to the Azure OpenAI LLM and
    parse the structured JSON response."""

    endpoint = os.getenv(
        "ENDPOINT_URL",
        "https://oaiscience-oai-swc.openai.azure.com/",
    )
    deployment = os.getenv("DEPLOYMENT_NAME", model)

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2025-01-01-preview",
    )

    user_message = (
        f"### Table\n{table_md}\n\n"
        f"### Question\n{question}\n\n"
        f"### Answer\n{answer}"
    )

    response = client.chat.completions.create(
        model=deployment,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    # Robust parse: strip possible markdown fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def review_distortions(
    df: pd.DataFrame,
    question: str,
    answer: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Analyse a table + question + answer and return a distortion selection.

    Parameters
    ----------
    df : pd.DataFrame
        The clean / original table.
    question : str
        The natural-language question.
    answer : str
        The correct answer (string).
    model : str
        Azure OpenAI deployment name (default ``gpt-4o-mini``).
        Can be overridden via the ``DEPLOYMENT_NAME`` env var.
    temperature : float
        Sampling temperature (lower = more deterministic).

    Returns
    -------
    dict with keys ``analysis``, ``distortions``, ``recommended``.
    """
    table_md = _df_to_markdown(df)
    return _call_llm(
        table_md=table_md,
        question=question,
        answer=answer,
        model=model,
        temperature=temperature,
    )


def review_distortions_from_file(
    filepath: str,
    question: str,
    answer: str,
    **kwargs,
) -> dict[str, Any]:
    """Convenience wrapper: load a CSV / Excel file then select distortions."""
    df = _load_input(filepath)
    return review_distortions(df, question, answer, **kwargs)


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------


def print_review(review: dict[str, Any]) -> None:
    """Print a human-readable summary of the selection to stdout."""

    print("=" * 70)
    print("DISTORTION SELECTION")
    print("=" * 70)

    print(f"\n📝 Analysis:\n   {review['analysis']}\n")

    suitable = []
    unsuitable = []
    for name, info in review["distortions"].items():
        entry = f"  • {name:30s} → {info['reason']}"
        if info["verdict"] == "suitable":
            suitable.append(entry)
        else:
            unsuitable.append(entry)

    print(f"✅ Suitable distortions ({len(suitable)}):")
    for s in suitable:
        print(s)

    print(f"\n❌ Unsuitable distortions ({len(unsuitable)}):")
    for u in unsuitable:
        print(u)

    print(f"\n🏆 Recommended (most → least impactful):")
    for i, name in enumerate(review.get("recommended", []), 1):
        print(f"   {i}. {name}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LLM-powered selector that chooses suitable table distortions "
            "for a given question/answer pair."
        )
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input CSV or Excel file.",
    )
    parser.add_argument(
        "--question", "-q", required=True,
        help="The natural-language question on the table.",
    )
    parser.add_argument(
        "--answer", "-a", required=True,
        help="The correct answer to the question.",
    )
    parser.add_argument(
        "--model", "-m", default="gpt-4o-mini",
        help="Azure OpenAI deployment name (default: gpt-4o-mini). "
             "Can also be set via DEPLOYMENT_NAME env var.",
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.2,
        help="Sampling temperature (default: 0.2).",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Print raw JSON output instead of formatted summary.",
    )
    args = parser.parse_args()

    review = review_distortions_from_file(
        filepath=args.input,
        question=args.question,
        answer=args.answer,
        model=args.model,
        temperature=args.temperature,
    )

    if args.json_output:
        print(json.dumps(review, indent=2, ensure_ascii=False))
    else:
        print_review(review)


def run_selector(filepath, question, answer, model, show_review: bool = False):
    review = review_distortions_from_file(
        filepath=filepath,
        question=question,
        answer=answer, model=model
    )

    if show_review:
        print(review)
        print_review(review)

    return review


if __name__ == "__main__":
    # main()
    run_selector(filepath="csv/202-csv/184.csv",
                 question="which album released by the band schnell fenster produced the most singles appearing on the australian peak chart?",
                 answer="The Sound Of Trees",
                 show_review=True)
