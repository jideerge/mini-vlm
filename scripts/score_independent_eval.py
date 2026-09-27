"""Score two frozen prediction files on the sealed four-task evaluation set.

Predictions are JSONL rows with ``id`` and ``prediction`` string fields.  This
script never loads a model or creates predictions.  It uses the project's
frozen content scorer; an unparsed response counts as incorrect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import CLASS_ZH, CONTENT_METRICS_VERSION, score_content  # noqa: E402

DATA_DIR = ROOT / "data" / "independent_eval_v1"
SCORER = ROOT / "evaluation" / "metrics.py"
TASKS = ("existence", "counting", "attribute", "listing")
N_PER_TASK = 100
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260926


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(row)
    return records


def unique_rows(rows: list[dict], path: Path) -> dict[str, dict]:
    by_id = {}
    for number, row in enumerate(rows, 1):
        identity = row.get("id")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"{path}:{number}: id must be a nonempty string")
        if identity in by_id:
            raise ValueError(f"{path}:{number}: duplicate id {identity!r}")
        by_id[identity] = row
    return by_id


def protocol_values(protocol: dict) -> dict:
    criteria = protocol.get("criteria", protocol)
    if not isinstance(criteria, dict):
        raise ValueError("protocol criteria must be an object")
    names = ("macro_gain_min", "ci_lower_gt", "per_task_regression_max",
             "candidate_floors")
    missing = [name for name in names if name not in criteria]
    if missing:
        raise ValueError(f"protocol missing frozen criteria: {missing}")
    for name in names[:3]:
        value = criteria[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"protocol {name} must be a finite number")
    floors = criteria["candidate_floors"]
    if not isinstance(floors, dict) or set(floors) != set(TASKS):
        raise ValueError(f"protocol candidate_floors must contain exactly {TASKS}")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) or not 0 <= x <= 1 for x in floors.values()):
        raise ValueError("protocol candidate_floors must be rates in [0, 1]")
    if criteria["macro_gain_min"] < 0 or criteria["per_task_regression_max"] < 0:
        raise ValueError("protocol minimum gain and maximum regression must be nonnegative")
    bootstrap = protocol.get("paired_bootstrap")
    if not isinstance(bootstrap, dict) or any(
        bootstrap.get(name) != required for name, required in (
            ("iterations", BOOTSTRAP_REPS),
            ("seed", BOOTSTRAP_SEED),
            ("resamples_per_task", N_PER_TASK),
            ("stratify_by_task", True),
        )
    ):
        raise ValueError("protocol paired_bootstrap differs from frozen procedure")
    return {name: criteria[name] for name in names}


def verify_protocol(protocol: dict, manifest: dict, questions: Path) -> dict:
    if CONTENT_METRICS_VERSION != 2 or manifest.get("scorer_version") != 2:
        raise ValueError("independent set requires content scorer v2")
    current_scorer_hash = sha256(SCORER)
    expected_scorer_hash = manifest.get("scorer_sha256")
    protocol_scorer_hash = protocol.get("scorer_sha256")
    if not expected_scorer_hash or not protocol_scorer_hash:
        raise ValueError("manifest and protocol must both contain scorer_sha256")
    if current_scorer_hash != expected_scorer_hash or current_scorer_hash != protocol_scorer_hash:
        raise ValueError("content scorer SHA-256 differs from the frozen manifest/protocol")
    question_hash = sha256(questions)
    if question_hash != manifest.get("question_sha256"):
        raise ValueError("questions.jsonl SHA-256 differs from the frozen manifest")
    if "question_sha256" in protocol and protocol["question_sha256"] != question_hash:
        raise ValueError("questions.jsonl SHA-256 differs from the frozen protocol")
    if manifest.get("n") != N_PER_TASK * len(TASKS) or manifest.get("task_counts") != dict.fromkeys(TASKS, N_PER_TASK):
        raise ValueError("manifest must specify four tasks with 100 questions each")
    baseline_checkpoint = ROOT / protocol.get("baseline_checkpoint", "")
    if (not baseline_checkpoint.is_file() or
        sha256(baseline_checkpoint) != protocol.get("baseline_checkpoint_sha256")):
        raise ValueError("baseline checkpoint differs from the frozen protocol")
    return protocol_values(protocol)


def validate_questions(rows: list[dict], path: Path) -> dict[str, dict]:
    by_id = unique_rows(rows, path)
    if len(rows) != N_PER_TASK * len(TASKS):
        raise ValueError(f"questions must contain exactly {N_PER_TASK * len(TASKS)} IDs")
    counts = Counter(row.get("task") for row in rows)
    if counts != dict.fromkeys(TASKS, N_PER_TASK):
        raise ValueError(f"questions must have 100 items in each task: {dict(counts)}")
    images = [row.get("image") for row in rows]
    if len(set(images)) != len(images) or any(not isinstance(x, str) or not x for x in images):
        raise ValueError("questions must use 400 distinct nonempty image paths")
    for row in rows:
        task, meta = row["task"], row.get("meta")
        if not isinstance(meta, dict) or not isinstance(row.get("answer"), str):
            raise ValueError(f"{row['id']}: missing meta or answer")
        if task in ("counting", "existence", "attribute") and meta.get("class") not in CLASS_ZH:
            raise ValueError(f"{row['id']}: invalid VOC target class")
        if task == "counting" and (type(meta.get("count")) is not int or meta["count"] < 1):
            raise ValueError(f"{row['id']}: invalid count")
        if task == "existence" and meta.get("label") not in (0, 1):
            raise ValueError(f"{row['id']}: invalid existence label")
        if task == "listing":
            classes = meta.get("classes")
            if (not isinstance(classes, list) or not classes or
                len(classes) != len(set(classes)) or any(c not in CLASS_ZH for c in classes)):
                raise ValueError(f"{row['id']}: invalid class list")
        if not score_content(row, row["answer"])["correct"]:
            raise ValueError(f"{row['id']}: reference answer fails frozen scorer")
    return by_id


def validate_predictions(path: Path, question_ids: set[str]) -> dict[str, str]:
    rows = read_jsonl(path)
    by_id = unique_rows(rows, path)
    missing = question_ids - by_id.keys()
    extra = by_id.keys() - question_ids
    if len(rows) != len(question_ids) or missing or extra:
        raise ValueError(f"{path}: expected exactly 400 IDs; "
                         f"missing={sorted(missing)[:5]} ({len(missing)}), "
                         f"extra={sorted(extra)[:5]} ({len(extra)})")
    for row in rows:
        if not isinstance(row.get("prediction"), str):
            raise ValueError(f"{path}: {row['id']!r} prediction must be a string")
    return {identity: row["prediction"] for identity, row in by_id.items()}


def attribute_class_count(record: dict, annotations: Path, asset: dict) -> int:
    stem = Path(record["image"]).stem
    if asset.get("stem") != stem:
        raise ValueError(f"{record['id']}: manifest annotation stem mismatch")
    xml = annotations / f"{stem}.xml"
    if sha256(xml) != asset.get("annotation_sha256"):
        raise ValueError(f"{record['id']}: annotation SHA-256 mismatch")
    classes = set()
    for obj in ET.parse(xml).getroot().findall("object"):
        if obj.find("bndbox") is not None and obj.findtext("name") in CLASS_ZH:
            classes.add(obj.findtext("name"))
    if not classes or record["meta"]["class"] not in classes:
        raise ValueError(f"{record['id']}: inconsistent attribute annotation")
    return len(classes)


def make_slices(rows: list[dict], data_dir: Path, manifest: dict) -> dict[str, list[str]]:
    assets = unique_rows(manifest.get("assets", []), data_dir / "manifest.json")
    if set(assets) != {row["id"] for row in rows}:
        raise ValueError("manifest assets must match the question IDs")
    slices = {
        "existence_positive": [], "existence_negative": [],
        "counting_one": [], "counting_two_plus": [],
        "attribute_single_class": [], "attribute_multi_class": [],
        "listing_single_class": [], "listing_multi_class": [],
    }
    for row in rows:
        task, meta, identity = row["task"], row["meta"], row["id"]
        if task == "existence":
            name = "existence_positive" if meta["label"] == 1 else "existence_negative"
        elif task == "counting":
            name = "counting_one" if meta["count"] == 1 else "counting_two_plus"
        elif task == "listing":
            name = "listing_single_class" if len(meta["classes"]) == 1 else "listing_multi_class"
        else:
            classes = attribute_class_count(row, data_dir / "annotations", assets[identity])
            name = "attribute_single_class" if classes == 1 else "attribute_multi_class"
        slices[name].append(identity)
    if any(not ids for ids in slices.values()):
        raise ValueError("each predeclared slice must contain at least one question")
    return slices


def fraction(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(sorted_values: list[float], probability: float) -> float:
    index = (len(sorted_values) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (index - lo)


def paired_bootstrap(rows: list[dict], baseline: dict, candidate: dict) -> dict:
    differences = {
        task: [int(candidate[row["id"]]["correct"]) - int(baseline[row["id"]]["correct"])
               for row in rows if row["task"] == task]
        for task in TASKS
    }
    if any(len(values) != N_PER_TASK for values in differences.values()):
        raise ValueError("paired bootstrap requires exactly 100 questions per task")
    rng = random.Random(BOOTSTRAP_SEED)
    samples = {task: [] for task in TASKS}
    macro_samples = []
    for _ in range(BOOTSTRAP_REPS):
        gains = []
        for task in TASKS:
            values = differences[task]
            gain = sum(values[rng.randrange(N_PER_TASK)]
                       for _ in range(N_PER_TASK)) / N_PER_TASK
            samples[task].append(gain)
            gains.append(gain)
        macro_samples.append(sum(gains) / len(TASKS))
    for values in (*samples.values(), macro_samples):
        values.sort()
    return {
        "method": "paired, stratified by task, resampling 100 question pairs per task",
        "replicates": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
        "confidence": 0.95, "percentile_method": "linear interpolation",
        "macro_gain_ci": [percentile(macro_samples, .025), percentile(macro_samples, .975)],
        "per_task_gain_ci": {
            task: [percentile(samples[task], .025), percentile(samples[task], .975)]
            for task in TASKS
        },
    }


def summarize(rows: list[dict], slices: dict[str, list[str]], baseline: dict,
              candidate: dict, bootstrap: dict, criteria: dict) -> dict:
    tasks = {}
    for task in TASKS:
        ids = [row["id"] for row in rows if row["task"] == task]
        b = fraction([baseline[x]["correct"] for x in ids])
        c = fraction([candidate[x]["correct"] for x in ids])
        tasks[task] = {
            "n": len(ids), "baseline": b, "candidate": c, "gain": c - b,
            "baseline_unparsed": fraction([not baseline[x]["parsed"] for x in ids]),
            "candidate_unparsed": fraction([not candidate[x]["parsed"] for x in ids]),
            "gain_ci_95": bootstrap["per_task_gain_ci"][task],
        }
    macro_b = sum(tasks[t]["baseline"] for t in TASKS) / len(TASKS)
    macro_c = sum(tasks[t]["candidate"] for t in TASKS) / len(TASKS)
    sliced = {}
    for name, ids in slices.items():
        b = fraction([baseline[x]["correct"] for x in ids])
        c = fraction([candidate[x]["correct"] for x in ids])
        sliced[name] = {"n": len(ids), "baseline": b, "candidate": c, "gain": c - b}
    listing_ids = [row["id"] for row in rows if row["task"] == "listing"]
    listing_b = sum(baseline[x]["f1"] for x in listing_ids) / len(listing_ids)
    listing_c = sum(candidate[x]["f1"] for x in listing_ids) / len(listing_ids)
    macro_gain = macro_c - macro_b
    tolerance = 1e-12
    checks = {
        "macro_gain_min": macro_gain + tolerance >= criteria["macro_gain_min"],
        "macro_gain_ci_lower_gt": bootstrap["macro_gain_ci"][0] > criteria["ci_lower_gt"],
        "per_task_regression_max": all(tasks[t]["gain"] + tolerance >=
                                       -criteria["per_task_regression_max"] for t in TASKS),
        "candidate_floors": all(tasks[t]["candidate"] + tolerance >=
                                criteria["candidate_floors"][t] for t in TASKS),
    }
    return {
        "n": len(rows), "task_order": list(TASKS), "content_metrics_version": CONTENT_METRICS_VERSION,
        "macro": {"baseline": macro_b, "candidate": macro_c,
                  "gain": macro_gain, "gain_ci_95": bootstrap["macro_gain_ci"]},
        "tasks": tasks, "slices": sliced,
        "listing_f1": {"baseline": listing_b, "candidate": listing_c,
                       "gain": listing_c - listing_b},
        "unparsed": {
            "baseline": fraction([not baseline[row["id"]]["parsed"] for row in rows]),
            "candidate": fraction([not candidate[row["id"]]["parsed"] for row in rows]),
        },
        "bootstrap": bootstrap, "criteria": criteria,
        "acceptance_checks": checks, "accepted": all(checks.values()),
    }


def score(args: argparse.Namespace) -> dict:
    questions_path = Path(args.questions)
    manifest_path = Path(args.manifest)
    protocol_path = Path(args.protocol)
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    if baseline_path.resolve() == candidate_path.resolve():
        raise ValueError("baseline and candidate prediction files must differ")
    manifest, protocol = read_json(manifest_path), read_json(protocol_path)
    if protocol.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("manifest.json SHA-256 differs from the frozen protocol")
    criteria = verify_protocol(protocol, manifest, questions_path)
    rows = read_jsonl(questions_path)
    by_id = validate_questions(rows, questions_path)
    slices = make_slices(rows, questions_path.parent, manifest)
    baseline_predictions = validate_predictions(baseline_path, set(by_id))
    candidate_predictions = validate_predictions(candidate_path, set(by_id))
    baseline_scores = {identity: score_content(row, baseline_predictions[identity])
                       for identity, row in by_id.items()}
    candidate_scores = {identity: score_content(row, candidate_predictions[identity])
                        for identity, row in by_id.items()}
    bootstrap = paired_bootstrap(rows, baseline_scores, candidate_scores)
    result = summarize(rows, slices, baseline_scores, candidate_scores, bootstrap, criteria)
    result["files"] = {
        "questions_sha256": sha256(questions_path), "manifest_sha256": sha256(manifest_path),
        "protocol_sha256": sha256(protocol_path), "scorer_sha256": sha256(SCORER),
        "baseline_predictions_sha256": sha256(baseline_path),
        "candidate_predictions_sha256": sha256(candidate_path),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="JSONL rows: id, prediction")
    parser.add_argument("--candidate", required=True, help="JSONL rows: id, prediction")
    parser.add_argument("--questions", default=DATA_DIR / "questions.jsonl", type=Path)
    parser.add_argument("--manifest", default=DATA_DIR / "manifest.json", type=Path)
    parser.add_argument("--protocol", default=DATA_DIR / "protocol.json", type=Path)
    parser.add_argument("--out", type=Path, help="optional JSON report path; refuses overwrite")
    args = parser.parse_args()
    try:
        report = score(args)
        encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.out:
            with args.out.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        print(encoded, end="")
    except (OSError, ValueError, ET.ParseError) as exc:
        parser.exit(2, f"independent evaluation error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
