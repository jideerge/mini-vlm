#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Question-only baseline: memorize the most frequent training answer per question.

Uses no image pixels, image features, model weights, or test labels for selection.
Unknown questions fall back to the task-level most frequent training answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Allow direct execution from the repository root without importing PyTorch.
import sys
sys.path.insert(0, str(ROOT))
from evaluation.metrics import score_content, summarize_content  # noqa: E402


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def split_records(records: list[dict], splits_path: Path) -> dict[str, list[dict]]:
    ids = json.loads(splits_path.read_text(encoding="utf-8"))["ids"]
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("数据包含重复样本 id")
    split_id_sets = {part: set(ids[part]) for part in ("train", "val", "test")}
    if set(by_id) - set().union(*split_id_sets.values()):
        raise ValueError("数据包含原划分未收录的样本 id")
    selected = {part: [by_id[i] for i in ids[part] if i in by_id]
                for part in ("train", "val", "test")}
    image_sets = {part: {r["image"] for r in rows}
                  for part, rows in selected.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if image_sets[a] & image_sets[b]:
            raise ValueError(f"图片跨 {a}/{b} 泄漏")
    return selected


def most_common(counter: Counter) -> tuple[str, int]:
    """Break ties lexically, independent of JSONL row order."""
    best = max(counter.values())
    return min(answer for answer, n in counter.items() if n == best), best


class QuestionPrior:
    def __init__(self, train_records: list[dict]):
        self.by_question: dict[tuple[str, str], Counter] = defaultdict(Counter)
        self.by_task: dict[str, Counter] = defaultdict(Counter)
        for row in train_records:
            self.by_question[(row["task"], row["question"])][row["answer"]] += 1
            self.by_task[row["task"]][row["answer"]] += 1

    def predict(self, row: dict) -> tuple[str, str, int]:
        key = (row["task"], row["question"])
        if key in self.by_question:
            answer, count = most_common(self.by_question[key])
            return answer, "question", count
        if row["task"] not in self.by_task:
            raise ValueError(f"训练集没有任务 {row['task']}")
        answer, count = most_common(self.by_task[row["task"]])
        return answer, "task_fallback", count


def normalize_exact(text: str) -> str:
    # Keep the exact same rule as training.losses.normalize_answer_text.
    return re.sub(r"[\s，。、？！：；,.\?!:;]", "", text)


def summarize_details(details: list[dict]) -> dict:
    n = len(details)
    tasks = sorted({d["task"] for d in details})
    return {
        "n": n,
        "exact": sum(d["exact"] for d in details) / n if n else 0.0,
        "content": summarize_content([d["content"] for d in details]),
        "per_task": {
            task: {
                "n": len(rows),
                "exact": sum(d["exact"] for d in rows) / len(rows),
                "content": summarize_content([d["content"] for d in rows]),
            }
            for task in tasks
            for rows in [[d for d in details if d["task"] == task]]
        },
    }


def paired_comparison(details: list[dict], model_result: dict) -> dict:
    if model_result.get("split") != "test":
        raise ValueError("模型结果必须来自 test 集")
    model_details = model_result["details"]
    by_id = {d["id"]: d for d in model_details}
    if len(by_id) != len(model_details) or set(by_id) != {d["id"] for d in details}:
        raise ValueError("模型结果与基线的样本 id 不一致")
    out = {}
    for task in sorted({d["task"] for d in details}) + ["overall"]:
        rows = details if task == "overall" else [d for d in details if d["task"] == task]
        counts = Counter()
        for row in rows:
            model = by_id[row["id"]]
            if (model["task"], model["question"], model["reference"]) != (
                row["task"], row["question"], row["reference"]
            ):
                raise ValueError(f"样本 {row['id']} 的问题或答案不一致")
            counts[(bool(model["content"]["correct"]), bool(row["content"]["correct"]))] += 1
        n = len(rows)
        out[task] = {
            "n": n,
            "both_correct": counts[(True, True)],
            "model_only_correct": counts[(True, False)],
            "question_prior_only_correct": counts[(False, True)],
            "both_wrong": counts[(False, False)],
            "model_content": (counts[(True, True)] + counts[(True, False)]) / n,
            "question_prior_content": (counts[(True, True)] + counts[(False, True)]) / n,
            "model_minus_prior": (counts[(True, False)] - counts[(False, True)]) / n,
        }
    return out


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="无图的训练集问题频率基线")
    parser.add_argument("--data", default="data/processed/baseline_10k.jsonl")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--model-result", default="outputs/baseline_10k_content_v1/eval_test_per_task.json")
    parser.add_argument("--out", default="outputs/question_prior_10k/eval_test.json")
    args = parser.parse_args()
    data_path, splits_path = resolve(args.data), resolve(args.splits)
    model_path, out_path = resolve(args.model_result), resolve(args.out)
    parts = split_records(load_records(data_path), splits_path)
    prior = QuestionPrior(parts["train"])
    details = []
    for row in parts["test"]:
        answer, source, train_frequency = prior.predict(row)
        details.append({
            "id": row["id"], "task": row["task"], "question": row["question"],
            "reference": row["answer"], "predicted_answer": answer,
            "source": source, "train_frequency": train_frequency,
            "exact": normalize_exact(answer) == normalize_exact(row["answer"]),
            "content": score_content(row, answer),
        })
    result = {
        "method": "train_question_majority_then_task_fallback",
        "split": "test", "train_n": len(parts["train"]),
        "data_sha256": sha256(data_path), "splits_sha256": sha256(splits_path),
        "question_seen": sum(d["source"] == "question" for d in details),
        "task_fallback": sum(d["source"] == "task_fallback" for d in details),
        **summarize_details(details),
        "details": details,
    }
    if model_path.exists():
        model = json.loads(model_path.read_text(encoding="utf-8"))
        if model.get("data") != args.data:
            raise ValueError("模型结果的数据文件与基线不同")
        result["paired_comparison"] = paired_comparison(details, model)
        result["model_result_sha256"] = sha256(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"test={result['n']} seen_question={result['question_seen']} fallback={result['task_fallback']}")
    print(f"question prior: exact={result['exact']:.1%} content={result['content']['correct']:.1%}")
    for task, row in result["per_task"].items():
        print(f"  {task:<10} n={row['n']:>3} exact={row['exact']:.1%} content={row['content']['correct']:.1%}")
    if "paired_comparison" in result:
        print(f"model content={result['paired_comparison']['overall']['model_content']:.1%}; "
              f"model - prior={result['paired_comparison']['overall']['model_minus_prior']:+.1%}")
    print(f"result -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
