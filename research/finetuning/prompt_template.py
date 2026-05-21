"""Single source of truth for the SFT prompt + target format.

Design
------
The model is trained to produce **natural, integrated reasoning** that ends
with a `Final Answer:` JSON block, e.g.:

    Looking at the entire table, the `Population` column appears vertically
    shifted: the first few cells are empty while the same number of orphan
    values appear at the bottom. After realigning the shifted rows upward,
    the populations greater than 2,000 are associated with four townships.

    Final Answer:
    ```json
    {"answer": 4, "dtype": "int"}
    ```

For an **undistorted** table the model just reasons straight through:

    Filtering rows where Status == 'Active' and counting the unique Region
    values gives three regions.

    Final Answer:
    ```json
    {"answer": 3, "dtype": "int"}
    ```

The training label is a single `reasoning` string (2-3 sentences max). For
distorted samples, the reasoning naturally weaves in **what looks wrong**,
**why** it is wrong, and **how it is mentally corrected** before the answer.
This avoids the rigid "Detection / Repair / Reasoning" sectioning, which made
the model trigger-happy about hallucinating distortions on clean tables.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

SYSTEM_PROMPT = (
    "You are an expert data analyst specializing in solving data analytics "
    "questions based on a tabular data. Your role is to analyze the data and "
    "provide accurate answers to user queries.\n"
    "\n"
    "**Your Environment:**\n"
    "- You will be provided with the relevant data context required to answer the query.\n"
    "\n"
    "**Your Responsibilities:**\n"
    "1. Carefully analyze the user's query to understand what information they need\n"
    "2. Study the table structure and content to better understand how to answer queries accurately\n"
)

USER_TEMPLATE = (
    "## Table (Markdown)\n"
    "{table_markdown}\n\n"
    "## Question\n"
    "{query}\n\n"
    "Reason in 2-3 sentences. Then output:\n"
    "Final Answer:\n"
    "```json\n"
    "{{\"answer\": ..., \"dtype\": \"...\"}}\n"
    "```"
)


def build_messages(example: Dict[str, Any]) -> list[Dict[str, str]]:
    """Build the chat messages for an example (no assistant turn)."""
    user = USER_TEMPLATE.format(
        table_markdown=example["table_markdown"].strip(),
        query=example["query"].strip(),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_target(example: Dict[str, Any]) -> str:
    """Build the assistant target string from the supervised fields.

    Accepts either:
      - New schema:  `reasoning` (single string, 2-3 sentences)
      - Legacy schema: `distortion_detection` + `repair_description`
        + `corrected_reasoning`  ->  concatenated into one flowing paragraph
    """
    answer = example["answer"]
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except Exception:
            pass
    final_json = json.dumps(
        {"answer": answer, "dtype": example["dtype"]},
        ensure_ascii=False,
    )

    reasoning = example.get("reasoning")
    if not reasoning:
        # Legacy fallback: stitch the three legacy fields together.
        parts = [
            (example.get("distortion_detection") or "").strip(),
            (example.get("repair_description") or "").strip(),
            (example.get("corrected_reasoning") or "").strip(),
        ]
        reasoning = " ".join(p for p in parts if p)

    reasoning = reasoning.strip()
    return f"{reasoning}\n\nFinal Answer:\n```json\n{final_json}\n```"


def render_for_training(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """Return {'prompt': ..., 'completion': ...} using the tokenizer's chat template.

    The prompt ends with the assistant turn opening tokens so that loss is only
    computed on the completion.
    """
    messages = build_messages(example)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    completion = build_target(example) + tokenizer.eos_token
    return {"prompt": prompt, "completion": completion}


def render_for_inference(example: Dict[str, Any], tokenizer) -> str:
    messages = build_messages(example)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Parsing the model's response back into structured fields
# ---------------------------------------------------------------------------
_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*:?\s*", re.IGNORECASE)


def parse_response(text: str) -> Dict[str, Any]:
    """Parse a generated response into reasoning + final JSON answer."""
    out: Dict[str, Any] = {
        "reasoning": "",
        "final_answer_raw": "",
        "answer": None,
        "dtype": None,
        "parse_ok": False,
    }

    # Split reasoning from the final answer block.
    m_fa = _FINAL_ANSWER_RE.search(text)
    if m_fa:
        out["reasoning"] = text[: m_fa.start()].strip()
        out["final_answer_raw"] = text[m_fa.end() :].strip()
    else:
        out["reasoning"] = text.strip()

    # Extract JSON answer (search the whole text as a fallback).
    search_in = out["final_answer_raw"] or text
    m = _JSON_BLOCK_RE.search(search_in)
    if not m:
        m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            payload = json.loads(m.group(1))
            out["answer"] = payload.get("answer")
            out["dtype"] = payload.get("dtype")
            out["parse_ok"] = True
        except json.JSONDecodeError:
            pass
    return out
