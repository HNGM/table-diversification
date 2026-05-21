"""
Dashboard to compare original vs distorted evaluation results.

Each distorted test case is shown side-by-side with the original (matched by query).
The screenshots referenced by `image_file` are displayed together with the model
answer, ground-truth answer, eval correctness and diversification/distortion
metadata.

Usage:
    python research/report/analysis/compare_dashboard.py \
        --original research/results/170526/wikitq_dataset_05172026_original_default_no_sandbox_markdown_dev-gpt-54-reasoning.json \
        --distorted research/results/170526/wikitq_dataset_05172026_disturbed_default_no_sandbox_markdown_dev-gpt-54-reasoning.json

Then open http://127.0.0.1:5000 in a browser.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template_string, request, send_file

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_eval(entry: dict[str, Any]) -> dict[str, Any]:
    evals = entry.get("eval") or []
    return evals[0] if evals else {}


def is_correct(entry: dict[str, Any]) -> bool | None:
    ev = first_eval(entry)
    return ev.get("eval") if "eval" in ev else None


def model_answer(entry: dict[str, Any]) -> Any:
    ev = first_eval(entry)
    resp = ev.get("agent_response") or {}
    if isinstance(resp, dict) and "answer" in resp:
        return resp["answer"]
    return ev.get("raw_response")


def build_app(original_path: str, distorted_path: str) -> Flask:
    original_data = load_json(original_path)
    distorted_data = load_json(distorted_path)

    # Map original entries by query (primary) and by index (fallback)
    by_query: dict[str, dict[str, Any]] = {}
    by_index: dict[str, dict[str, Any]] = {}
    for e in original_data:
        q = (e.get("query") or "").strip()
        if q and q not in by_query:
            by_query[q] = e
        idx = e.get("index")
        if idx and idx not in by_index:
            by_index[idx] = e

    # Build a list of paired records
    records: list[dict[str, Any]] = []
    for i, d in enumerate(distorted_data):
        q = (d.get("query") or "").strip()
        orig = by_query.get(q)
        if orig is None:
            # fall back: distorted index "nt-2_broken_rows_split" -> "nt-2"
            didx = d.get("index") or ""
            base_idx = didx.split("_")[0] if didx else ""
            orig = by_index.get(base_idx)

        records.append(
            {
                "i": i,
                "query": d.get("query"),
                "answer": d.get("answer"),
                "distorted_index": d.get("index"),
                "original_index": (orig or {}).get("index"),
                "diversification_type": d.get("diversification_type"),
                "distortion_type": d.get("distortion_type"),
                "type": d.get("type"),
                "dtype": d.get("dtype"),
                "distorted_eval": is_correct(d),
                "original_eval": is_correct(orig) if orig else None,
                "distorted_answer": model_answer(d),
                "original_answer": model_answer(orig) if orig else None,
                "distorted_image": d.get("image_file"),
                "original_image": (orig or {}).get("image_file"),
                "distorted_data_file": d.get("data_file"),
                "original_data_file": (orig or {}).get("data_file"),
                "_orig_entry": orig,
                "_dist_entry": d,
            }
        )

    # Aggregate stats
    total = len(records)
    dist_correct = sum(1 for r in records if r["distorted_eval"] is True)
    orig_correct = sum(1 for r in records if r["original_eval"] is True)
    regressions = sum(
        1 for r in records if r["original_eval"] is True and r["distorted_eval"] is False
    )
    improvements = sum(
        1 for r in records if r["original_eval"] is False and r["distorted_eval"] is True
    )
    missing_original = sum(1 for r in records if r["_orig_entry"] is None)

    # Per-diversification breakdown
    per_div: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in records:
        k = r["diversification_type"] or "unknown"
        per_div[k]["total"] += 1
        if r["distorted_eval"]:
            per_div[k]["correct"] += 1
    per_div_list = sorted(
        [
            {
                "name": k,
                "total": v["total"],
                "correct": v["correct"],
                "acc": (v["correct"] / v["total"]) if v["total"] else 0.0,
            }
            for k, v in per_div.items()
        ],
        key=lambda x: x["name"],
    )

    diversification_options = sorted({r["diversification_type"] or "" for r in records})
    distortion_options = sorted({r["distortion_type"] or "" for r in records})

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(
            INDEX_HTML,
            total=total,
            dist_correct=dist_correct,
            orig_correct=orig_correct,
            regressions=regressions,
            improvements=improvements,
            missing_original=missing_original,
            per_div=per_div_list,
            diversification_options=diversification_options,
            distortion_options=distortion_options,
            original_path=original_path,
            distorted_path=distorted_path,
        )

    @app.route("/api/records")
    def api_records():
        eval_filter = request.args.get("eval", "all")  # all / correct / wrong
        regr = request.args.get("regression", "all")  # all / only / none
        div = request.args.get("diversification", "")
        dist = request.args.get("distortion", "")
        q_search = (request.args.get("q") or "").lower().strip()

        out = []
        for r in records:
            if eval_filter == "correct" and r["distorted_eval"] is not True:
                continue
            if eval_filter == "wrong" and r["distorted_eval"] is True:
                continue
            if regr == "only" and not (
                r["original_eval"] is True and r["distorted_eval"] is False
            ):
                continue
            if div and (r["diversification_type"] or "") != div:
                continue
            if dist and (r["distortion_type"] or "") != dist:
                continue
            if q_search and q_search not in (r["query"] or "").lower():
                continue
            out.append(
                {
                    "i": r["i"],
                    "query": r["query"],
                    "answer": r["answer"],
                    "distorted_index": r["distorted_index"],
                    "original_index": r["original_index"],
                    "diversification_type": r["diversification_type"],
                    "distortion_type": r["distortion_type"],
                    "distorted_eval": r["distorted_eval"],
                    "original_eval": r["original_eval"],
                }
            )
        return jsonify(out)

    @app.route("/case/<int:i>")
    def case_view(i: int):
        if i < 0 or i >= len(records):
            abort(404)
        r = records[i]
        return render_template_string(
            CASE_HTML,
            r=r,
            i=i,
            prev_i=i - 1 if i > 0 else None,
            next_i=i + 1 if i < len(records) - 1 else None,
            total=total,
            json_pretty=lambda obj: json.dumps(obj, indent=2, ensure_ascii=False),
            orig_entry=r["_orig_entry"],
            dist_entry=r["_dist_entry"],
        )

    @app.route("/image")
    def image():
        rel = request.args.get("path", "")
        if not rel:
            abort(400)
        # Normalize path; only allow paths inside REPO_ROOT
        target = (REPO_ROOT / rel).resolve()
        try:
            target.relative_to(REPO_ROOT)
        except ValueError:
            abort(403)
        if not target.exists() or not target.is_file():
            abort(404)
        mime, _ = mimetypes.guess_type(str(target))
        return send_file(str(target), mimetype=mime or "application/octet-stream")

    return app


# --------------------------- Templates ---------------------------

INDEX_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Distortion Comparison Dashboard</title>
<style>
 body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; color: #222; }
 h1 { margin: 0 0 8px; }
 .meta { color: #666; font-size: 12px; margin-bottom: 12px; word-break: break-all; }
 .stats { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }
 .card { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 10px 14px; min-width: 140px; }
 .card .v { font-size: 20px; font-weight: 600; }
 .card .l { font-size: 12px; color: #555; }
 table { border-collapse: collapse; width: 100%; font-size: 13px; }
 th, td { border: 1px solid #d0d7de; padding: 4px 8px; text-align: left; vertical-align: top; }
 th { background: #f0f3f6; position: sticky; top: 0; }
 tr:hover { background: #fffbe6; }
 .ok { color: #1a7f37; font-weight: 600; }
 .bad { color: #cf222e; font-weight: 600; }
 .filters { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
 .filters input, .filters select { padding: 4px 6px; }
 .pill { display: inline-block; padding: 1px 6px; background: #eee; border-radius: 10px; font-size: 11px; }
 a { color: #0969da; text-decoration: none; }
 a:hover { text-decoration: underline; }
 details summary { cursor: pointer; font-weight: 600; }
</style>
</head>
<body>
<h1>Distortion Comparison Dashboard</h1>
<div class="meta">
  <div><b>Original:</b> {{ original_path }}</div>
  <div><b>Distorted:</b> {{ distorted_path }}</div>
</div>

<div class="stats">
  <div class="card"><div class="v">{{ total }}</div><div class="l">Distorted cases</div></div>
  <div class="card"><div class="v ok">{{ dist_correct }}</div><div class="l">Distorted correct</div></div>
  <div class="card"><div class="v">{{ orig_correct }}</div><div class="l">Original correct (matched)</div></div>
  <div class="card"><div class="v bad">{{ regressions }}</div><div class="l">Regressions (orig✓ → dist✗)</div></div>
  <div class="card"><div class="v ok">{{ improvements }}</div><div class="l">Improvements (orig✗ → dist✓)</div></div>
  <div class="card"><div class="v">{{ missing_original }}</div><div class="l">No original match</div></div>
</div>

<details>
  <summary>Per-diversification accuracy ({{ per_div|length }})</summary>
  <table style="margin-top:8px; width:auto;">
    <thead><tr><th>Diversification</th><th>Total</th><th>Correct</th><th>Accuracy</th></tr></thead>
    <tbody>
    {% for row in per_div %}
      <tr>
        <td>{{ row.name }}</td>
        <td>{{ row.total }}</td>
        <td>{{ row.correct }}</td>
        <td>{{ "%.1f%%"|format(row.acc * 100) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</details>

<h3>Cases</h3>
<div class="filters">
  <input id="q" type="text" placeholder="Search query..." style="min-width:240px;" />
  <select id="eval">
    <option value="all">All eval results</option>
    <option value="correct">Distorted correct only</option>
    <option value="wrong">Distorted wrong only</option>
  </select>
  <select id="regression">
    <option value="all">All cases</option>
    <option value="only">Regressions only</option>
  </select>
  <select id="diversification">
    <option value="">All diversifications</option>
    {% for d in diversification_options %}<option value="{{ d }}">{{ d or '(none)' }}</option>{% endfor %}
  </select>
  <select id="distortion">
    <option value="">All distortion types</option>
    {% for d in distortion_options %}<option value="{{ d }}">{{ d or '(none)' }}</option>{% endfor %}
  </select>
  <span id="count" class="pill">0</span>
</div>

<table id="tbl">
  <thead>
    <tr>
      <th>#</th>
      <th>Distorted idx</th>
      <th>Original idx</th>
      <th>Query</th>
      <th>GT answer</th>
      <th>Diversification</th>
      <th>Distortion</th>
      <th>Orig ✓</th>
      <th>Dist ✓</th>
      <th></th>
    </tr>
  </thead>
  <tbody></tbody>
</table>

<script>
async function refresh() {
  const params = new URLSearchParams({
    eval: document.getElementById('eval').value,
    regression: document.getElementById('regression').value,
    diversification: document.getElementById('diversification').value,
    distortion: document.getElementById('distortion').value,
    q: document.getElementById('q').value,
  });
  const r = await fetch('/api/records?' + params.toString());
  const data = await r.json();
  document.getElementById('count').textContent = data.length + ' rows';
  const fmt = (v) => v === true ? '<span class="ok">✓</span>' : (v === false ? '<span class="bad">✗</span>' : '—');
  const esc = (s) => (s == null ? '' : String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])));
  const tbody = document.querySelector('#tbl tbody');
  tbody.innerHTML = data.map(d => `
    <tr>
      <td>${d.i}</td>
      <td>${esc(d.distorted_index)}</td>
      <td>${esc(d.original_index)}</td>
      <td>${esc(d.query)}</td>
      <td>${esc(d.answer)}</td>
      <td>${esc(d.diversification_type)}</td>
      <td>${esc(d.distortion_type)}</td>
      <td>${fmt(d.original_eval)}</td>
      <td>${fmt(d.distorted_eval)}</td>
      <td><a href="/case/${d.i}" target="_blank">view</a></td>
    </tr>`).join('');
}
['q','eval','regression','diversification','distortion'].forEach(id => {
  document.getElementById(id).addEventListener('input', refresh);
  document.getElementById(id).addEventListener('change', refresh);
});
refresh();
</script>
</body>
</html>
"""

CASE_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Case {{ i }} — {{ r.distorted_index }}</title>
<style>
 body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; color: #222; }
 .nav { display: flex; gap: 12px; align-items: center; margin-bottom: 10px; }
 .nav a { color: #0969da; }
 .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
 .panel { border: 1px solid #d0d7de; border-radius: 6px; padding: 10px; background: #fafbfc; }
 .panel h3 { margin: 0 0 6px; }
 .kv { font-size: 13px; }
 .kv div { padding: 2px 0; }
 .kv b { display: inline-block; min-width: 130px; color: #555; }
 img { max-width: 100%; border: 1px solid #d0d7de; border-radius: 4px; background: #fff; }
 .ok { color: #1a7f37; font-weight: 600; }
 .bad { color: #cf222e; font-weight: 600; }
 pre { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 4px; padding: 8px; white-space: pre-wrap; max-height: 320px; overflow: auto; font-size: 12px; }
 .query { background: #fff8c5; border: 1px solid #d4a72c; padding: 8px 10px; border-radius: 4px; margin-bottom: 12px; }
</style>
</head>
<body>
<div class="nav">
  <a href="/">← back</a>
  {% if prev_i is not none %}<a href="/case/{{ prev_i }}">◀ prev</a>{% endif %}
  <span>Case {{ i }} / {{ total - 1 }}</span>
  {% if next_i is not none %}<a href="/case/{{ next_i }}">next ▶</a>{% endif %}
</div>

<div class="query">
  <b>Query:</b> {{ r.query }}<br/>
  <b>Ground truth:</b> {{ r.answer }} &nbsp; <span style="color:#666">(dtype: {{ r.dtype }})</span>
</div>

<div class="grid">
  <div class="panel">
    <h3>Original — <code>{{ r.original_index or 'NOT FOUND' }}</code>
      {% if r.original_eval is sameas true %}<span class="ok">✓ correct</span>
      {% elif r.original_eval is sameas false %}<span class="bad">✗ wrong</span>
      {% else %}<span>—</span>{% endif %}
    </h3>
    <div class="kv">
      <div><b>Diversification</b> {{ (orig_entry or {}).get('diversification_type') }}</div>
      <div><b>Distortion type</b> {{ (orig_entry or {}).get('distortion_type') }}</div>
      <div><b>Type</b> {{ (orig_entry or {}).get('type') }}</div>
      <div><b>Model answer</b> {{ r.original_answer }}</div>
      <div><b>Data file</b> <code>{{ r.original_data_file }}</code></div>
      <div><b>Image file</b> <code>{{ r.original_image }}</code></div>
    </div>
    {% if r.original_image %}
      <p><img src="/image?path={{ r.original_image }}" alt="original screenshot" /></p>
    {% else %}<p><i>No image</i></p>{% endif %}
    {% if orig_entry %}
      <details><summary>Raw entry</summary><pre>{{ json_pretty(orig_entry) }}</pre></details>
    {% endif %}
  </div>

  <div class="panel">
    <h3>Distorted — <code>{{ r.distorted_index }}</code>
      {% if r.distorted_eval is sameas true %}<span class="ok">✓ correct</span>
      {% elif r.distorted_eval is sameas false %}<span class="bad">✗ wrong</span>
      {% else %}<span>—</span>{% endif %}
    </h3>
    <div class="kv">
      <div><b>Diversification</b> {{ r.diversification_type }}</div>
      <div><b>Distortion type</b> {{ r.distortion_type }}</div>
      <div><b>Type</b> {{ r.type }}</div>
      <div><b>Model answer</b> {{ r.distorted_answer }}</div>
      <div><b>Data file</b> <code>{{ r.distorted_data_file }}</code></div>
      <div><b>Image file</b> <code>{{ r.distorted_image }}</code></div>
    </div>
    {% if r.distorted_image %}
      <p><img src="/image?path={{ r.distorted_image }}" alt="distorted screenshot" /></p>
    {% else %}<p><i>No image</i></p>{% endif %}
    <details><summary>Raw entry</summary><pre>{{ json_pretty(dist_entry) }}</pre></details>
  </div>
</div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original", required=True, help="Path to original results JSON")
    ap.add_argument("--distorted", required=True, help="Path to distorted results JSON")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    app = build_app(args.original, args.distorted)
    print(f"Serving dashboard at http://{args.host}:{args.port}")
    print(f"  original  = {os.path.abspath(args.original)}")
    print(f"  distorted = {os.path.abspath(args.distorted)}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
