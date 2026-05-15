"""Evaluator for table-QA predictions.

Public API (stable):
    evaluate(gt_answer, gt_dtype, pred_answer, pred_dtype) -> bool

This module borrows ideas from the official WikiTableQuestions evaluator
(robust string normalization, numeric/date-aware matching, set-style
collection comparison) while keeping the dtype-driven dispatch the rest of
the codebase already relies on.
"""

import ast
import math
import re
import unicodedata
from typing import Any, Optional, Tuple

import pandas as pd

ALLOWED_DTYPES = [
    "int", "float", "str", "list", "pd.Series",
    "set", "dict", "bool", "tuple",
]

# Float tolerance used when both sides are clearly numeric.
FLOAT_ABS_TOL = 1e-6
FLOAT_REL_TOL = 1e-6


# --------------------------------------------------------------------------- #
# String normalization (adapted from the official WTQ evaluator)              #
# --------------------------------------------------------------------------- #

_QUOTE_SINGLE_RE = re.compile(r"[\u2018\u2019\u00B4`]")
_QUOTE_DOUBLE_RE = re.compile(r"[\u201C\u201D]")
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")
_CITATION_RE = re.compile(r"((?<!^)\[[^\]]*\]|\[\d+\]|[\u2022\u2666\u2020\u2021*#+])*$")
_PAREN_TAIL_RE = re.compile(r"(?<!^)( \([^)]*\))*$")
_OUTER_QUOTES_RE = re.compile(r'^"([^"]*)"$')
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize(x: Any) -> str:
    """WTQ-style string normalization.

    - Strip diacritics
    - Convert smart quotes/dashes to ASCII
    - Strip trailing citations ([1], dagger, asterisk, ...)
    - Strip trailing parenthetical details
    - Strip outermost double quotes and a trailing period
    - Lowercase and collapse whitespace
    """
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    # Remove diacritics
    x = "".join(c for c in unicodedata.normalize("NFKD", x)
                if unicodedata.category(c) != "Mn")
    # Normalize quotes and dashes
    x = _QUOTE_SINGLE_RE.sub("'", x)
    x = _QUOTE_DOUBLE_RE.sub('"', x)
    x = _DASH_RE.sub("-", x)
    while True:
        old_x = x
        x = _CITATION_RE.sub("", x.strip())
        x = _PAREN_TAIL_RE.sub("", x.strip())
        x = _OUTER_QUOTES_RE.sub(r"\1", x.strip())
        if x == old_x:
            break
    if x and x[-1] == ".":
        x = x[:-1]
    x = _WS_RE.sub(" ", x).lower().strip()
    return x


# --------------------------------------------------------------------------- #
# Lightweight value parsing                                                   #
# --------------------------------------------------------------------------- #

def try_number(text: Any) -> Optional[float]:
    """Try to interpret a value as a number. Returns None on failure."""
    if isinstance(text, bool):
        # Avoid silently treating True/False as 1/0.
        return None
    if isinstance(text, (int, float)):
        if isinstance(text, float) and (math.isnan(text) or math.isinf(text)):
            return None
        return float(text)
    if not isinstance(text, str):
        return None
    s = text.strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


_DATE_RE = re.compile(r"^(xx|xxxx|\d{1,4})-(xx|\d{1,2})-(xx|\d{1,2})$")


def try_date(text: Any) -> Optional[Tuple[int, int, int]]:
    """Try to interpret a string as a yyyy-mm-dd date (xx = wildcard)."""
    if not isinstance(text, str):
        return None
    m = _DATE_RE.match(text.lower().strip())
    if not m:
        return None
    y_s, mo_s, d_s = m.groups()
    try:
        year = -1 if y_s in ("xx", "xxxx") else int(y_s)
        month = -1 if mo_s == "xx" else int(mo_s)
        day = -1 if d_s == "xx" else int(d_s)
    except ValueError:
        return None
    if year == month == day == -1:
        return None
    if month != -1 and not (1 <= month <= 12):
        return None
    if day != -1 and not (1 <= day <= 31):
        return None
    return (year, month, day)


def _safe_literal_eval(text: str) -> Any:
    """Safer replacement for eval() on caller-provided strings."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return text


# --------------------------------------------------------------------------- #
# Dtype helpers                                                               #
# --------------------------------------------------------------------------- #

def _normalize_dtype(val: Any) -> str:
    """Normalize numpy/pandas types to base Python type names."""
    dtype_name = type(val).__name__
    if dtype_name in ("int64", "int32", "int16", "int8",
                      "uint64", "uint32", "uint16", "uint8"):
        return "int"
    if dtype_name in ("float64", "float32", "float16"):
        return "float"
    if dtype_name in ("bool_", "bool"):
        return "bool"
    if dtype_name == "str_":
        return "str"
    return dtype_name


# --------------------------------------------------------------------------- #
# Scalar matching with cross-type coercion                                    #
# --------------------------------------------------------------------------- #

def _match_scalar(gt: Any, pred: Any) -> bool:
    """Match two scalar values, trying number -> date -> normalized string."""
    # Direct equality (covers ints, identical objects, etc.)
    if gt is pred:
        return True
    if type(gt) == type(pred) and not isinstance(gt, float):
        try:
            if gt == pred:
                return True
        except Exception:
            pass

    # Numeric path
    g_num = try_number(gt)
    p_num = try_number(pred)
    if g_num is not None and p_num is not None:
        if math.isclose(g_num, p_num, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL):
            return True
        # Numeric mismatch is decisive only if both sides are *unambiguously*
        # numeric. If one side came from a string that also parses as a date,
        # fall through and let the date / string paths try.
        if not (isinstance(gt, str) or isinstance(pred, str)):
            return False

    # Date path
    g_date = try_date(gt) if isinstance(gt, str) else None
    p_date = try_date(pred) if isinstance(pred, str) else None
    if g_date is not None and p_date is not None:
        return g_date == p_date

    # Fallback: normalized string compare
    return normalize(gt) == normalize(pred)


# --------------------------------------------------------------------------- #
# Collection matching                                                         #
# --------------------------------------------------------------------------- #

def _match_collection_unordered(gt_items, pred_items) -> bool:
    """Size-equal + every gt item matches some (unique) pred item."""
    gt_items = list(gt_items)
    pred_items = list(pred_items)
    if len(gt_items) != len(pred_items):
        return False
    remaining = list(pred_items)
    for g in gt_items:
        for i, p in enumerate(remaining):
            if _match_value(g, _normalize_dtype(g), p, _normalize_dtype(p)):
                remaining.pop(i)
                break
        else:
            return False
    return True


# --------------------------------------------------------------------------- #
# Core dispatch                                                               #
# --------------------------------------------------------------------------- #

def _match_value(gt_answer: Any, gt_dtype: str,
                 pred_answer: Any, pred_dtype: str) -> bool:
    """Internal recursive matcher. Public API is `evaluate`."""
    # Cross-type scalar comparisons go through _match_scalar.
    scalar_dtypes = {"str", "int", "float", "bool"}
    if gt_dtype in scalar_dtypes and pred_dtype in scalar_dtypes:
        if gt_dtype == "bool" and pred_dtype == "bool":
            return bool(gt_answer) == bool(pred_answer)
        return _match_scalar(gt_answer, pred_answer)

    # If pred is a string but gt is a structured type, try to parse pred.
    if isinstance(pred_answer, str) and gt_dtype in {"list", "tuple", "set", "dict", "pd.Series"}:
        parsed = _safe_literal_eval(pred_answer)
        if parsed is not pred_answer:
            pred_answer = parsed
            pred_dtype = _normalize_dtype(parsed)

    if isinstance(gt_answer, str) and gt_dtype in {"list", "tuple", "set", "dict", "pd.Series"}:
        parsed = _safe_literal_eval(gt_answer)
        if parsed is not gt_answer:
            gt_answer = parsed
            gt_dtype = _normalize_dtype(parsed)

    if gt_dtype == "pd.Series":
        gt_series = pd.Series(gt_answer) if isinstance(gt_answer, dict) else gt_answer
        pred_series = (pd.Series(pred_answer)
                       if isinstance(pred_answer, dict) else pred_answer)
        if not isinstance(gt_series, pd.Series) or not isinstance(pred_series, pd.Series):
            return False
        if len(gt_series) != len(pred_series):
            return False

        # Build normalized index lookup for pred.
        def _norm_key(k: Any) -> Any:
            return normalize(k) if isinstance(k, str) else k

        pred_lookup: dict = {}
        for k, v in pred_series.items():
            pred_lookup.setdefault(_norm_key(k), []).append(v)

        for idx, gt_val in gt_series.items():
            key = _norm_key(idx)
            if key not in pred_lookup or not pred_lookup[key]:
                return False
            pred_val = pred_lookup[key].pop(0)
            if not _match_value(gt_val, _normalize_dtype(gt_val),
                                pred_val, _normalize_dtype(pred_val)):
                return False
        return True

    if gt_dtype in ("list", "tuple"):
        if not hasattr(pred_answer, "__iter__"):
            return False
        return _match_collection_unordered(gt_answer, pred_answer)

    if gt_dtype == "set":
        if not hasattr(pred_answer, "__iter__"):
            return False
        # Use unordered collection matching so element-level normalization
        # (case, diacritics, numeric coercion) is honored.
        return _match_collection_unordered(gt_answer, pred_answer)

    if gt_dtype == "dict":
        if not isinstance(pred_answer, dict):
            return False
        if len(gt_answer) != len(pred_answer):
            return False
        gt_norm = {(normalize(k) if isinstance(k, str) else k): v
                   for k, v in gt_answer.items()}
        pred_norm = {(normalize(k) if isinstance(k, str) else k): v
                     for k, v in pred_answer.items()}
        for key, gt_val in gt_norm.items():
            if key not in pred_norm:
                return False
            pred_val = pred_norm[key]
            if not _match_value(gt_val, _normalize_dtype(gt_val),
                                pred_val, _normalize_dtype(pred_val)):
                return False
        return True

    # Fallback: scalar comparison using normalized strings.
    return _match_scalar(gt_answer, pred_answer)


def evaluate(gt_answer: Any, gt_dtype: str,
             pred_answer: Any, pred_dtype: str) -> bool:
    """Return True iff `pred_answer` matches `gt_answer` under `gt_dtype`.

    Stable public signature. Improvements over the previous version:
      * WTQ-style Unicode/citation/quote normalization for all string compares.
      * Numeric tolerance tightened to ~1e-6 with cross-type coercion
        (e.g. "3.0" vs 3 now matches even when dtypes disagree).
      * Native yyyy-mm-dd date matching (with `xx` wildcards).
      * `ast.literal_eval` instead of `eval` for parsing structured strings.
      * `set` comparison now uses element-wise normalized matching.
      * Dict / Series keys normalized (not just lowercased).
    """
    if pred_dtype not in ALLOWED_DTYPES:
        return False
    try:
        return _match_value(gt_answer, gt_dtype, pred_answer, pred_dtype)
    except (ValueError, TypeError, SyntaxError, AttributeError, KeyError, IndexError):
        return False
