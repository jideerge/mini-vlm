"""Audit the frozen four-task holdout without running a model."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evaluation.metrics import CONTENT_METRICS_VERSION, score_content
from scripts.build_independent_eval import (
    COUNTING_EXCLUDED_CLASSES, INDEX, OLD_DATA, OUTPUT, PHASH_DISTANCE,
    QA_EXCLUDED_STEMS, ROOT, SPLITS, VOC_TEST,
    W2_CACHE_INDEX, W2_RESULTS, objects_for, old_images, phash, sha256,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    manifest_path = OUTPUT / "manifest.json"
    digest = (OUTPUT / "manifest.json.sha256").read_text(encoding="ascii").split()[0]
    check(sha256(manifest_path) == digest, "manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_path = OUTPUT / "protocol.json"
    protocol_digest = (OUTPUT / "protocol.json.sha256").read_text(encoding="ascii").split()[0]
    check(sha256(protocol_path) == protocol_digest, "protocol digest mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    check(protocol["manifest_sha256"] == digest, "protocol points to different manifest")
    check(manifest["n"] == 400, "unexpected benchmark size")
    check(manifest["scorer_version"] == CONTENT_METRICS_VERSION == 2,
          "scorer version mismatch")
    check(manifest["scorer_sha256"] == sha256(ROOT / "evaluation/metrics.py"),
          "scorer changed after freeze")
    check(manifest["builder_sha256"] == sha256(ROOT / "scripts/build_independent_eval.py"),
          "builder changed after freeze")
    check(manifest["source_index_sha256"] == sha256(INDEX),
          "source index changed after freeze")
    old_inputs = OLD_DATA + [SPLITS, W2_CACHE_INDEX, W2_RESULTS,
                             ROOT / "datasets/voc_source.py",
                             ROOT / "scripts/run_experiments.py"]
    for path in old_inputs:
        key = path.relative_to(ROOT).as_posix()
        check(manifest["old_inputs_sha256"][key] == sha256(path),
              f"historical input changed: {key}")

    question_path = OUTPUT / "questions.jsonl"
    check(manifest["question_sha256"] == sha256(question_path),
          "question file changed after freeze")
    check(protocol["questions_sha256"] == manifest["question_sha256"] and
          protocol["scorer_sha256"] == manifest["scorer_sha256"],
          "protocol points to different questions or scorer")
    questions = [json.loads(line) for line in
                 question_path.read_text(encoding="utf-8").splitlines() if line]
    assets = manifest["assets"]
    check(len(questions) == len(assets) == 400, "wrong row/asset count")
    check(len({q["id"] for q in questions}) == 400, "duplicate question id")
    check(len({a["stem"] for a in assets}) == 400, "duplicate image stem")
    check(Counter(q["task"] for q in questions) ==
          {task: 100 for task in ("existence", "counting", "attribute", "listing")},
          "task quota mismatch")

    index = json.loads(INDEX.read_text(encoding="utf-8"))["entries"]
    by_stem = {entry["stem"]: entry for entry in index}
    old = old_images(index)
    check(len(old) == manifest["old_image_stems"], "historical pool changed")
    old_hashes = {sha256(path) for path in old.values()}
    old_phashes = [phash(path) for path in old.values()]
    selected_hashes: set[str] = set()
    selected_phashes: list[int] = []
    minimum_old_distance = 64

    for q, asset in zip(questions, assets):
        stem = asset["stem"]
        check(q["id"] == asset["id"] and stem in q["id"],
              f"question/asset mismatch: {stem}")
        check(stem not in old, f"historical stem leaked: {stem}")
        check(stem not in QA_EXCLUDED_STEMS, f"manual QA exclusion selected: {stem}")
        check(stem in by_stem and by_stem[stem]["split"] == VOC_TEST,
              f"not from official VOC test partition: {stem}")
        image = OUTPUT / "images" / f"{stem}.jpg"
        xml = OUTPUT / "annotations" / f"{stem}.xml"
        check(image.resolve() == (ROOT / q["image"]).resolve(),
              f"bad question image path: {stem}")
        check(image.is_file() and xml.is_file(), f"missing asset: {stem}")
        digest = sha256(image)
        check(digest == asset["image_sha256"] == sha256(Path(by_stem[stem]["path"])),
              f"image digest mismatch: {stem}")
        check(sha256(xml) == asset["annotation_sha256"],
              f"annotation digest mismatch: {stem}")
        check(digest not in old_hashes and digest not in selected_hashes,
              f"duplicate image bytes: {stem}")
        fingerprint = phash(image)
        check(f"{fingerprint:016x}" == asset["phash64"],
              f"perceptual hash mismatch: {stem}")
        distance = min((fingerprint ^ previous).bit_count()
                       for previous in old_phashes)
        check(distance > PHASH_DISTANCE,
              f"near-duplicate historical image: {stem}")
        minimum_old_distance = min(minimum_old_distance, distance)
        check(all((fingerprint ^ previous).bit_count() > PHASH_DISTANCE
                  for previous in selected_phashes),
              f"near-duplicate within holdout: {stem}")
        selected_hashes.add(digest)
        selected_phashes.append(fingerprint)

        objects, source_xml = objects_for(by_stem[stem])
        check(sha256(source_xml) == asset["annotation_sha256"],
              f"source annotation changed: {stem}")
        classes = {o["class"] for o in objects}
        meta = q["meta"]
        if q["task"] == "listing":
            check(meta["classes"] == sorted(classes) and
                  all(not o["bad"] and o["area"] >= .03 for o in objects) and
                  1 <= len(classes) <= 4, f"bad listing gold: {stem}")
        elif q["task"] == "attribute":
            ordered = sorted(objects, key=lambda x: x["area"], reverse=True)
            check(meta["class"] == ordered[0]["class"] and
                  all(not o["bad"] for o in ordered) and ordered[0]["area"] >= .10 and
                  (len(ordered) == 1 or ordered[0]["area"] >= 2 * ordered[1]["area"]),
                  f"bad attribute gold: {stem}")
        elif q["task"] == "counting":
            targets = [o for o in objects if o["class"] == meta["class"]]
            check(meta["class"] not in COUNTING_EXCLUDED_CLASSES and
                  len(targets) == meta["count"] and 1 <= len(targets) <= 4 and
                  all(not o["bad"] and o["area"] >= .02 for o in targets),
                  f"bad counting gold: {stem}")
        elif q["task"] == "existence":
            targets = [o for o in objects if o["class"] == meta["class"]]
            check(meta["label"] in (0, 1) and bool(targets) == bool(meta["label"])
                  and (not targets or all(not o["bad"] and o["area"] >= .02
                                          for o in targets)),
                  f"bad existence gold: {stem}")
        check(score_content(q, q["answer"])["correct"],
              f"gold answer failed scorer: {stem}")

    check(Counter(q["meta"]["label"] for q in questions
                  if q["task"] == "existence") == {0: 50, 1: 50},
          "existence balance changed")
    check(sum(q["meta"]["count"] >= 2 for q in questions
              if q["task"] == "counting") == 40, "counting repeat quota changed")
    check(sum(len(q["meta"]["classes"]) >= 2 for q in questions
              if q["task"] == "listing") == 25, "listing multi quota changed")
    check(sum(len({o["class"] for o in objects_for(by_stem[a["stem"]])[0]}) >= 2
              for q, a in zip(questions, assets) if q["task"] == "attribute") == 20,
          "attribute multi quota changed")
    result = {"status": "PASS", "n": 400, "old_image_stems": len(old),
              "minimum_phash_distance_to_old": minimum_old_distance,
              "image_sha256_unique": len(selected_hashes),
              "questions_sha256": manifest["question_sha256"],
              "manifest_sha256": sha256(manifest_path),
              "protocol_sha256": sha256(protocol_path),
              "scorer_sha256": manifest["scorer_sha256"]}
    (OUTPUT / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
