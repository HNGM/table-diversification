"""Evaluate the finetuned adapter on the test split.

The table is provided to the model as Markdown (the `table_markdown` field of
each test record). For each example the model generates a short reasoning +
final JSON block; we parse the JSON answer and compare to ground truth.

Usage:
    python research/finetuning/evaluate_sft.py \
        --adapter research/finetuning/outputs/qwen25-1p5b-disturbed/final \
        --test    research/finetuning/data/splits/test.jsonl \
        --out     research/finetuning/outputs/qwen25-1p5b-disturbed/test_predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_template import parse_response, render_for_inference  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def normalize(v: Any) -> Any:
    """Light normalization for accuracy comparison."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v.strip().lower()
    if isinstance(v, list):
        return [normalize(x) for x in v]
    if isinstance(v, dict):
        return {str(k).strip().lower(): normalize(val) for k, val in v.items()}
    if isinstance(v, float):
        return round(v, 4)
    return v


def is_correct(pred: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    if not pred.get("parse_ok"):
        return False
    gold_ans = gold["answer"]
    if isinstance(gold_ans, str):
        try:
            gold_ans = json.loads(gold_ans)
        except Exception:
            pass
    return normalize(pred.get("answer")) == normalize(gold_ans)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=False, type=Path, default=None,
                   help="Path to PEFT adapter dir. Omit to evaluate the bare base model.")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--test", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Max generated tokens (SFT target is ~150). Lower = faster.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit loading; use full fp16 on GPU (faster, more VRAM).")
    p.add_argument("--no-merge", action="store_true",
                   help="Do NOT merge LoRA into base before generation (slower).")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate at most N examples (debugging).")
    args = p.parse_args()

    tokenizer_src = args.adapter if args.adapter is not None else args.base_model
    print(f"[eval] loading tokenizer from {tokenizer_src}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = not args.no_4bit
    bnb = None
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    print(f"[eval] loading base model {args.base_model}  (4bit={use_4bit})")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    print(f"[eval] base loaded in {time.time()-t0:.1f}s")

    if args.adapter is not None:
        print(f"[eval] attaching LoRA adapter from {args.adapter}")
        t0 = time.time()
        model = PeftModel.from_pretrained(base, str(args.adapter))

        # Merging the LoRA weights into the base gives a much faster forward pass.
        # Not possible while the base is 4-bit quantized -- in that case we keep
        # the adapter attached but still re-enable kv-cache for generation.
        if not args.no_merge and not use_4bit:
            print("[eval] merging LoRA into base for faster generation")
            model = model.merge_and_unload()
        elif not args.no_merge and use_4bit:
            print("[eval] cannot merge LoRA on 4-bit base; keeping adapter attached "
                  "(pass --no-4bit to enable merge for ~3-5x faster generation)")
        print(f"[eval] adapter ready in {time.time()-t0:.1f}s")
    else:
        print("[eval] no --adapter passed; evaluating the BASE model as-is")
        model = base

    # CRITICAL for speed: training disables kv-cache (incompatible with grad
    # checkpointing). Re-enable it for inference or generation will crawl.
    model.config.use_cache = True
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.use_cache = True

    model.eval()
    print(f"[eval] model ready in {time.time()-t0:.1f}s")

    rows = load_jsonl(args.test)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[eval] {len(rows)} test examples")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_correct = 0
    n_parsed = 0
    pbar = tqdm(rows, desc="evaluating", dynamic_ncols=True)
    with args.out.open("w", encoding="utf-8") as f_out:
        for i, ex in enumerate(pbar):
            t_ex = time.time()
            prompt = render_for_inference(ex, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-5),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            new_tokens = gen[0, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            parsed = parse_response(text)

            correct = is_correct(parsed, ex)
            n_correct += int(correct)
            n_parsed += int(parsed["parse_ok"])

            f_out.write(
                json.dumps(
                    {
                        "index": ex.get("index"),
                        "query": ex.get("query"),
                        "gold_answer": ex.get("answer"),
                        "gold_dtype": ex.get("dtype"),
                        "pred_raw": text,
                        "pred_parsed": parsed,
                        "correct": correct,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f_out.flush()  # so you can `tail -f` the output file
            done = i + 1
            pbar.set_postfix(
                acc=f"{n_correct/done:.3f}",
                parsed=f"{n_parsed/done:.2f}",
                tok=int(new_tokens.shape[0]),
                s=f"{time.time()-t_ex:.1f}",
            )

    n = len(rows)
    print(f"[eval] parse_ok = {n_parsed}/{n} ({n_parsed/n:.1%})")
    print(f"[eval] accuracy = {n_correct}/{n} ({n_correct/n:.1%})")
    print(f"[eval] predictions -> {args.out}")


if __name__ == "__main__":
    main()

