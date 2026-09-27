#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit whether every matching object pair gives the same spatial answer."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.build_voc_vqa import _load_objects  # noqa: E402


def relation_options(objects: list[dict], subject: str, other: str) -> tuple[set[str], int, int]:
    subjects = [o for o in objects if o["name"] == subject]
    others = [o for o in objects if o["name"] == other]
    directions = {"左边" if a["cx"] < b["cx"] else "右边"
                  for a in subjects for b in others if a["cx"] != b["cx"]}
    return directions, len(subjects), len(others)


def audit_record(record: dict, split: str) -> dict:
    image = Path(record["image"])
    if not image.is_file():
        raise FileNotFoundError(f"需要原始 VOC 图片旁的 XML 标注：{image}")
    objects = _load_objects({"path": str(image), "stem": image.stem})
    meta = record["meta"]
    options, n_subjects, n_others = relation_options(
        objects, meta["subject"], meta["other"])
    if len(options) > 1:
        status = "ambiguous"
    elif not options or meta["answer"] not in options:
        status = "invalid"
    else:
        status = "clear"
    return {
        "id": record["id"], "split": split, "question": record["question"],
        "label": meta["answer"], "possible_answers": sorted(options),
        "subject_instances": n_subjects, "other_instances": n_others,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="审计 VOC 空间问答的多实例歧义")
    parser.add_argument("--data", default="data/processed/baseline_10k.jsonl")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--model-result", default="outputs/baseline_10k_decode_fixed/eval_test_per_task.json")
    parser.add_argument("--prior-result", default="outputs/question_prior_10k_decode_fixed/eval_test.json")
    parser.add_argument("--out", default="report/spatial_audit.json")
    parser.add_argument("--clean-out", default="data/processed/spatial_clean_9333.jsonl")
    parser.add_argument("--smoke-out", default="data/processed/spatial_clean_smoke_1k.jsonl")
    args = parser.parse_args()
    resolve = lambda x: Path(x) if Path(x).is_absolute() else ROOT / x
    records = [json.loads(line) for line in resolve(args.data).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    split_ids = json.loads(resolve(args.splits).read_text(encoding="utf-8"))["ids"]
    by_id = {r["id"]: r for r in records}
    rows = [audit_record(by_id[sample_id], split)
            for split in ("train", "val", "test")
            for sample_id in split_ids[split]
            if by_id[sample_id]["task"] == "spatial"]
    overall = Counter(row["status"] for row in rows)
    by_split = {split: dict(Counter(row["status"] for row in rows if row["split"] == split))
                for split in ("train", "val", "test")}
    multi_instance = sum(row["subject_instances"] > 1 or row["other_instances"] > 1
                         for row in rows)
    model = {d["id"]: d for d in json.loads(resolve(args.model_result).read_text(encoding="utf-8"))["details"]}
    prior = {d["id"]: d for d in json.loads(resolve(args.prior_result).read_text(encoding="utf-8"))["details"]}
    comparison = {}
    for status in ("clear", "ambiguous", "invalid"):
        subset = [row for row in rows if row["split"] == "test" and row["status"] == status]
        if not subset:
            continue
        if any(row["id"] not in model or row["id"] not in prior for row in subset):
            raise ValueError("模型与问题先验结果缺少空间样本")
        comparison[status] = {
            "n": len(subset),
            "model_content_correct": sum(model[row["id"]]["content"]["correct"] for row in subset),
            "question_prior_content_correct": sum(prior[row["id"]]["content"]["correct"] for row in subset),
        }
    clear_ids = {row["id"] for row in rows if row["status"] == "clear"}
    clean_records = [r for r in records if r["task"] != "spatial" or r["id"] in clear_ids]
    if len(clean_records) != len(records) - overall["ambiguous"] - overall["invalid"]:
        raise AssertionError("空间清理数量与审计结果不一致")
    clean_path = resolve(args.clean_out)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in clean_records),
                          encoding="utf-8")
    quotas = Counter()
    smoke_records = []
    for record in clean_records:
        if quotas[record["task"]] < 200:
            smoke_records.append(record)
            quotas[record["task"]] += 1
    if len(smoke_records) != 1000:
        raise ValueError(f"清理后的数据不足以构造五类各200条的smoke数据：{dict(quotas)}")
    smoke_path = resolve(args.smoke_out)
    smoke_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                  for r in smoke_records), encoding="utf-8")
    result = {"n": len(rows), "status_counts": dict(overall),
              "by_split": by_split, "multi_instance": multi_instance,
              "test_comparison": comparison,
              "clean_data": {"path": args.clean_out, "n": len(clean_records),
                             "spatial_n": len(clear_ids)},
              "rows": rows}
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("spatial", len(rows), "status", dict(overall), "multi_instance", multi_instance)
    for split, counts in by_split.items():
        print(split, counts)
    for status, counts in comparison.items():
        print("test", status, counts)
    print("clean dataset ->", clean_path, len(clean_records), "spatial", len(clear_ids))
    print("smoke dataset ->", smoke_path, len(smoke_records), dict(quotas))
    print("result ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
