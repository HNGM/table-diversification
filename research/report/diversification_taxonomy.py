"""Canonical taxonomy for ``diversification_type`` values.

This module is the single source of truth for grouping/renaming the raw
``diversification_type`` labels found in dataset and result JSON files into the
final reporting taxonomy. Any report, dashboard, or aggregation that needs a
"diversification type breakdown" should import :func:`map_diversification`
(and optionally :data:`GROUPED_ORDER`) from here instead of hard-coding names.

Current mapping (raw -> grouped)::

    vertical_shift              -> vertical_shift
    horizontal_shift            -> horizontal_shift
    merge_cells                 -> content_merge
    broken_rows_merge           -> content_merge
    multi_column_collapse       -> content_merge
    broken_rows_split           -> content_split
    footnote_injection          -> footnote_injection
    ocr_char_misinterpret       -> character_misinterpretation
    random_noise_chars          -> stray_characters_insertion
    date_format_corruption      -> label_mismatch   (folded into label_mismatch)
    label_mismatch              -> label_mismatch
    decimal_separator_swap     -> decimal_separator_swap   (unchanged)

Anything not listed is passed through unchanged so legacy/unknown labels stay
visible rather than silently disappearing.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional


# Raw label -> grouped label. Keep keys lowercase / snake_case to match how the
# values are written in the dataset JSON files.
DIVERSIFICATION_MAP: Dict[str, str] = {
    "vertical_shift": "vertical_shift",
    "horizontal_shift": "horizontal_shift",

    # Anything that visually merges content into fewer cells/rows.
    "merge_cells": "content_merge",
    "broken_rows_merge": "content_merge",
    "multi_column_collapse": "content_merge",

    # Splitting a single logical row into multiple.
    "broken_rows_split": "content_split",

    "footnote_injection": "footnote_injection",

    "ocr_char_misinterpret": "character_misinterpretation",
    "random_noise_chars": "stray_characters_insertion",

    # date_format_corruption is now considered a flavour of label_mismatch.
    "date_format_corruption": "label_mismatch",
    "label_mismatch": "label_mismatch",

    "decimal_separator_swap": "decimal_separator_swap",
}


# Preferred display order for the grouped types.
GROUPED_ORDER: List[str] = [
    # Structural
    "vertical_shift",
    "horizontal_shift",
    "content_merge",
    "content_split",
    "footnote_injection",
    # Semantic
    "character_misinterpretation",
    "stray_characters_insertion",
    "label_mismatch",
    "decimal_separator_swap",
]


def map_diversification(name: Optional[str]) -> str:
    """Return the grouped diversification name for a raw label.

    Unknown / missing labels are returned unchanged (or as ``"unknown"`` when
    ``name`` is falsy) so they remain visible in reports.
    """
    if not name:
        return "unknown"
    return DIVERSIFICATION_MAP.get(name, name)


def ordered_groups(present: Iterable[str]) -> List[str]:
    """Return the subset of :data:`GROUPED_ORDER` that appears in ``present``,
    followed by any extra labels (alphabetical) not in the canonical order."""
    present_set = set(present)
    ordered = [g for g in GROUPED_ORDER if g in present_set]
    extra = sorted(present_set - set(ordered))
    return ordered + extra


__all__ = [
    "DIVERSIFICATION_MAP",
    "GROUPED_ORDER",
    "map_diversification",
    "ordered_groups",
]
