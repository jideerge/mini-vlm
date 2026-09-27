"""Build a sealed, image-disjoint VOC2007 four-task evaluation set without model inference."""
from __future__ import annotations
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

from datasets.build_voc_vqa import CLASS_ZH
from datasets.voc_source import VOC_CLASSES, sample_balanced, sample_random

ROOT = Path(__file__).resolve().parents[1]
OLD_DATA = [ROOT / "data/processed" / name for name in
            ("baseline_10k.jsonl", "debug_1k.jsonl",
             "spatial_clean_9333.jsonl", "spatial_clean_smoke_1k.jsonl")]
SPLITS = ROOT / "data/processed/splits.json"
INDEX = ROOT / "data/processed/voc_index.json"
W2_CACHE_INDEX = ROOT / "data/processed/features_w2/index.json"
W2_RESULTS = ROOT / "report/figures/w2_results.json"
OUTPUT = ROOT / "data/independent_eval_v1"
SEED = 20260926
VOC_TEST = "VOCtest_06-Nov-2007"
PHASH_DISTANCE = 8
QA_EXCLUDED_STEMS = {
    "002974", "003010", "003538", "004344", "005766", "009570",
}  # Spot checks found incomplete counts or objects only inside a photo/mirror.
COUNTING_EXCLUDED_CLASSES = {"chair", "pottedplant"}  # Crowded instances are annotation-ambiguous.
LISTING_CLASSES = "、".join(CLASS_ZH[c] for c in VOC_CLASSES)
EXISTENCE_CLASSES = (
    "person", "car", "bird", "chair", "dog", "pottedplant",
    "aeroplane", "cat", "cow", "tvmonitor",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phash(path: Path) -> int:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        pixels = np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS),
                            dtype=np.float32)
    low = dctn(pixels, type=2, norm="ortho")[:8, :8]
    threshold = np.median(low.flat[1:])
    bits = (low > threshold).reshape(-1)
    bits[0] = False
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def old_images(entries: list[dict]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in OLD_DATA:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                image = Path(json.loads(line)["image"])
                paths[image.stem] = image
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    for images in splits["images"].values():
        for name in images:
            image = Path(name)
            paths[image.stem] = image
    # Week 2 CLIP diagnostics also exposed images to this project. Their
    # samples are deterministic, and the 1,000-image cache records a cross-check.
    results = json.loads(W2_RESULTS.read_text(encoding="utf-8"))
    if results["C"]["n"] != 200 or results["K"]["n"] != 400 or results["D"]["n_images"] != 200:
        raise ValueError("Week 2 sampling sizes changed; re-audit old images")
    random_pool = sample_random(entries, 1000, seed=7)
    cached = json.loads(W2_CACHE_INDEX.read_text(encoding="utf-8"))["entries"]
    if {Path(e["path"]).stem for e in cached} != {e["stem"] for e in random_pool}:
        raise ValueError("Week 2 feature cache does not match reconstructed random pool")
    week2 = (random_pool
             + sample_balanced(entries, per_class=10, single_class_only=True, seed=42)
             + sample_balanced(entries, per_class=20, single_class_only=True, seed=100))
    for entry in week2:
        image = Path(entry["path"])
        paths[image.stem] = image
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing old images: {missing[:3]}")
    return paths


def objects_for(entry: dict) -> tuple[list[dict], Path]:
    photo = Path(entry["path"])
    xml = photo.parent.parent / "Annotations" / f"{entry['stem']}.xml"
    root = ET.parse(xml).getroot()
    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))
    if width < 1 or height < 1:
        raise ValueError(f"bad image size: {xml}")
    objects = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            x1 = float(box.findtext("xmin"))
            x2 = float(box.findtext("xmax"))
            y1 = float(box.findtext("ymin"))
            y2 = float(box.findtext("ymax"))
        except (ValueError, TypeError):
            continue
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)
        name = obj.findtext("name")
        if name in VOC_CLASSES:
            objects.append({"class": name, "area": area,
                            "bad": obj.findtext("difficult") == "1" or
                                   obj.findtext("truncated") == "1"})
    return objects, xml


def stable_key(stem: str, task: str, target: str) -> str:
    return hashlib.sha256(f"{SEED}|{task}|{stem}|{target}".encode()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"sealed evaluation directory already exists: {OUTPUT}")
    index = json.loads(INDEX.read_text(encoding="utf-8"))["entries"]
    assert len({e["stem"] for e in index}) == len(index)
    old = old_images(index)
    print(f"old image stems including Week 2: {len(old)}", flush=True)
    old_hashes = set()
    old_phashes = []
    for stem, path in sorted(old.items()):
        old_hashes.add(sha256(path))
        old_phashes.append(phash(path))
    print("old image hashes ready", flush=True)
    candidates = []
    excluded = Counter()
    for entry in index:
        if entry["split"] != VOC_TEST:
            continue
        if entry["stem"] in old:
            excluded["old_stem"] += 1
            continue
        if entry["stem"] in QA_EXCLUDED_STEMS:
            excluded["manual_qa"] += 1
            continue
        photo = Path(entry["path"])
        if not photo.is_file():
            raise FileNotFoundError(photo)
        digest = sha256(photo)
        if digest in old_hashes:
            excluded["old_bytes"] += 1
            continue
        fingerprint = phash(photo)
        minimum = min((fingerprint ^ previous).bit_count()
                      for previous in old_phashes)
        if minimum <= PHASH_DISTANCE:
            excluded["old_near"] += 1
            continue
        objects, xml = objects_for(entry)
        if not objects:
            excluded["no_objects"] += 1
            continue
        candidates.append({"entry": entry, "photo": photo, "xml": xml,
                           "sha256": digest, "phash": fingerprint,
                           "objects": objects})
    print(f"VOC test candidates: {len(candidates)}, exclusions: {dict(excluded)}",
          flush=True)

    selected: list[dict] = []
    taken: set[str] = set()
    taken_hashes: set[str] = set()
    taken_phashes: list[int] = []
    class_usage: Counter = Counter()
    task_class_usage: dict[str, Counter] = {
        task: Counter() for task in ("listing", "attribute", "counting", "existence")
    }

    def choose(task: str, options: list[tuple[dict, object, tuple[str, ...]]],
               question, answer, metadata) -> None:
        choices = []
        for item, target, classes in options:
            stem = item["entry"]["stem"]
            if (stem in taken or item["sha256"] in taken_hashes or
                any((item["phash"] ^ prior).bit_count() <= PHASH_DISTANCE
                    for prior in taken_phashes)):
                continue
            diversity = sum(task_class_usage[task][c] for c in classes)
            global_diversity = sum(class_usage[c] for c in classes)
            choices.append((diversity, global_diversity,
                            stable_key(stem, task, str(target)), item, target, classes))
        if not choices:
            raise RuntimeError(f"insufficient untouched images for {task}")
        _, _, _, item, target, classes = min(choices, key=lambda x: x[:3])
        stem = item["entry"]["stem"]
        taken.add(stem)
        taken_hashes.add(item["sha256"])
        taken_phashes.append(item["phash"])
        for cls in classes:
            task_class_usage[task][cls] += 1
            class_usage[cls] += 1
        selected.append({"item": item, "task": task, "question": question(target),
                         "answer": answer(target), "meta": metadata(target)})

    def eligible_listing(item, multi: bool):
        objects = item["objects"]
        classes = tuple(sorted({obj["class"] for obj in objects}))
        return (all(not obj["bad"] and obj["area"] >= .03 for obj in objects)
                and 1 <= len(classes) <= 4 and (len(classes) >= 2) == multi)

    def eligible_attribute(item, multi: bool):
        objects = sorted(item["objects"], key=lambda x: x["area"], reverse=True)
        classes = {obj["class"] for obj in objects}
        return ((len(classes) >= 2) == multi and all(not obj["bad"] for obj in objects)
                and objects[0]["area"] >= .10
                and (len(objects) == 1 or
                     objects[0]["area"] >= 2.0 * objects[1]["area"]))

    def listing_round(n, multi):
        for _ in range(n):
            opts = [(item, tuple(sorted({o["class"] for o in item["objects"]})),
                     tuple(sorted({o["class"] for o in item["objects"]})))
                    for item in candidates if eligible_listing(item, multi)]
            choose("listing", opts,
                   lambda _: f"只考虑以下20类：{LISTING_CLASSES}。图中有哪些？",
                   lambda cs: "图中有" + "、".join(CLASS_ZH[c] for c in cs) + "。",
                   lambda cs: {"classes": list(cs)})

    def attribute_round(n, multi):
        for _ in range(n):
            opts = [(item, max(item["objects"], key=lambda x: x["area"])["class"],
                     (max(item["objects"], key=lambda x: x["area"])["class"],))
                    for item in candidates if eligible_attribute(item, multi)]
            choose("attribute", opts, lambda _: "图片主要是什么？",
                   lambda cls: f"图片主要是{CLASS_ZH[cls]}。",
                   lambda cls: {"class": cls})

    def counting_round(n, repeated):
        for _ in range(n):
            opts = []
            for item in candidates:
                counts = Counter(o["class"] for o in item["objects"])
                for cls, count in counts.items():
                    if cls in COUNTING_EXCLUDED_CLASSES:
                        continue
                    targets = [o for o in item["objects"] if o["class"] == cls]
                    if (2 <= count <= 4) == repeated and 1 <= count <= 4 and all(
                            not o["bad"] and o["area"] >= .02 for o in targets):
                        opts.append((item, (cls, count), (cls,)))
            choose("counting", opts,
                   lambda pair: f"图中一共有几个{CLASS_ZH[pair[0]]}？",
                   lambda pair: f"图中有{['零','一','两','三','四'][pair[1]]}个{CLASS_ZH[pair[0]]}。",
                   lambda pair: {"class": pair[0], "count": pair[1]})

    def existence_round(label):
        for cls in EXISTENCE_CLASSES:
            for _ in range(5):
                opts = []
                for item in candidates:
                    targets = [o for o in item["objects"] if o["class"] == cls]
                    if label == 1:
                        valid = bool(targets) and all(
                            not o["bad"] and o["area"] >= .02 for o in targets)
                    else:
                        valid = not targets
                    if valid:
                        opts.append((item, cls, (cls,)))
                choose("existence", opts,
                       lambda target: f"图中是否有{CLASS_ZH[target]}？",
                       (lambda target: f"是的，图中有{CLASS_ZH[target]}。")
                       if label else
                       (lambda target: f"图中没有{CLASS_ZH[target]}。"),
                       lambda target: {"class": target, "label": label})

    attribute_round(20, True)
    listing_round(25, True)
    counting_round(40, True)
    existence_round(1)
    listing_round(75, False)
    attribute_round(80, False)
    counting_round(60, False)
    existence_round(0)
    assert len(selected) == len(taken) == 400
    assert len(taken_hashes) == 400
    assert all((a ^ b).bit_count() > PHASH_DISTANCE
               for i, a in enumerate(taken_phashes)
               for b in taken_phashes[i + 1:])

    # Four tasks are interleaved deterministically; each photo contributes exactly one question.
    random.Random(SEED).shuffle(selected)
    OUTPUT.mkdir(parents=True)
    image_dir = OUTPUT / "images"
    annotation_dir = OUTPUT / "annotations"
    image_dir.mkdir()
    annotation_dir.mkdir()
    records = []
    assets = []
    for row in selected:
        item = row["item"]
        stem = item["entry"]["stem"]
        image = image_dir / f"{stem}.jpg"
        xml = annotation_dir / f"{stem}.xml"
        shutil.copy2(item["photo"], image)
        shutil.copy2(item["xml"], xml)
        assert sha256(image) == item["sha256"]
        record = {"id": f"holdout_v1_{stem}_{row['task']}",
                  "image": f"data/independent_eval_v1/images/{stem}.jpg",
                  "task": row["task"], "question": row["question"],
                  "answer": row["answer"], "meta": row["meta"]}
        records.append(record)
        assets.append({"id": record["id"], "stem": stem,
                       "source_split": item["entry"]["split"],
                       "image_sha256": item["sha256"],
                       "annotation_sha256": sha256(xml),
                       "phash64": f"{item['phash']:016x}"})
    jsonl = OUTPUT / "questions.jsonl"
    jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                             for r in records), encoding="utf-8")
    from evaluation.metrics import CONTENT_METRICS_VERSION, score_content
    assert CONTENT_METRICS_VERSION == 2
    for record in records:
        result = score_content(record, record["answer"])
        assert result["correct"], (record["id"], result)
    task_counts = Counter(r["task"] for r in records)
    assert task_counts == {task: 100 for task in
                           ("existence", "counting", "attribute", "listing")}
    answer_counts = {
        "existence": dict(Counter(r["meta"]["label"] for r in records
                                  if r["task"] == "existence")),
        "counting": dict(Counter(r["meta"]["count"] for r in records
                                 if r["task"] == "counting")),
        "attribute": dict(Counter(r["meta"]["class"] for r in records
                                  if r["task"] == "attribute")),
        "listing_n_classes": dict(Counter(len(r["meta"]["classes"]) for r in records
                                           if r["task"] == "listing")),
    }
    manifest = {
        "version": 1, "created_by": "scripts/build_independent_eval.py",
        "seed": SEED, "n": len(records), "task_counts": dict(task_counts),
        "answers": answer_counts, "old_image_stems": len(old),
        "source": "PASCAL VOC 2007 official test partition, unused by known previous project training and diagnostics",
        "source_exclusions": dict(excluded), "old_exact_sha256_check": True,
        "old_phash_min_distance": PHASH_DISTANCE + 1,
        "image_unique": True, "one_question_per_image": True,
        "question_sha256": sha256(jsonl),
        "source_index_sha256": sha256(INDEX),
        "old_inputs_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
                               for p in OLD_DATA + [SPLITS, W2_CACHE_INDEX, W2_RESULTS,
                                                   ROOT / "datasets/voc_source.py",
                                                   ROOT / "scripts/run_experiments.py"]},
        "scorer_version": CONTENT_METRICS_VERSION,
        "scorer_sha256": sha256(ROOT / "evaluation/metrics.py"),
        "builder_sha256": sha256(Path(__file__)),
        "assets": assets,
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    (OUTPUT / "manifest.json.sha256").write_text(
        sha256(manifest_path) + "  manifest.json\n", encoding="ascii")
    print(f"sealed {len(records)} questions / {len(assets)} images", flush=True)
    print(answer_counts, flush=True)
    print("manifest", sha256(manifest_path), flush=True)


if __name__ == "__main__":
    main()
