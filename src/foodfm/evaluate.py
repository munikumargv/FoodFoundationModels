"""Evaluate a (fine-tuned) model on the held-out FoodFM split.

Metrics per example kind:
- number_recall: share of numbers in the reference answer that the model reproduced
  (checks grounding of standard values such as 1300 mg calcium).
- token_f1: lexical overlap with the reference answer.
- refusal_acc (out_of_scope only): the model declined a non-nutrition question.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .dataset import read_jsonl
from .retrieval import tokenize

_NUM = re.compile(r"\d+(?:\.\d+)?")
_REFUSAL = re.compile(r"\b(only (help|answer)|can only|outside|not able to help|nutrition (questions|assistant))\b", re.I)


def number_recall(pred: str, ref: str) -> float | None:
    ref_nums = set(_NUM.findall(ref))
    if not ref_nums:
        return None
    return len(ref_nums & set(_NUM.findall(pred))) / len(ref_nums)


def token_f1(pred: str, ref: str) -> float:
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    common = sum(min(p.count(t), r.count(t)) for t in set(p) & set(r))
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def is_refusal(pred: str) -> bool:
    return bool(_REFUSAL.search(pred))


def score(predictions: list[tuple[str, str, str]]) -> dict[str, Any]:
    """predictions: list of (kind, prediction, reference)."""
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for kind, pred, ref in predictions:
        b = buckets[kind]
        b["token_f1"].append(token_f1(pred, ref))
        nr = number_recall(pred, ref)
        if nr is not None:
            b["number_recall"].append(nr)
        if kind == "out_of_scope":
            b["refusal_acc"].append(float(is_refusal(pred)))
        elif kind in ("standard_fact", "country_habit", "profile_plan"):
            b["false_refusal"].append(float(is_refusal(pred)))
    return {kind: {m: round(sum(v) / len(v), 3) for m, v in metrics.items()} | {"n": len(metrics["token_f1"])}
            for kind, metrics in sorted(buckets.items())}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate FoodFM on the held-out split")
    ap.add_argument("--model", help="model key / Hugging Face id (defaults to the adapter's base model)")
    ap.add_argument("--adapter", help="LoRA adapter directory (omit to evaluate the base model)")
    ap.add_argument("--eval-path", default="data/eval.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="outputs/eval_report.json")
    a = ap.parse_args(argv)

    from .infer import FoodFM

    fm = FoodFM(a.model, a.adapter)
    preds = []
    for ex in read_jsonl(a.eval_path)[: a.limit]:
        msgs, ref = ex["messages"][:-1], ex["messages"][-1]["content"]
        preds.append((ex["kind"], fm.generate(msgs, temperature=0), ref))
    report = {"model": fm.hf_id, "adapter": a.adapter, "metrics": score(preds)}
    print(json.dumps(report, indent=2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
