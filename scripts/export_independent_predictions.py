"""Export one sealed, four-task holdout prediction file without scoring it.

Run once per role, after ``candidate_freeze.json`` has been written from the
existing training/validation data::

    python -B -m scripts.export_independent_predictions --role baseline
    python -B -m scripts.export_independent_predictions --role candidate

Each invocation atomically publishes a directory containing ``predictions.jsonl``
and ``provenance.json``.  No reference answer or scoring metadata is passed to
the model, and this module never computes a holdout score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DATA_DIR = ROOT / "data" / "independent_eval_v1"
PILOT_DIR = ROOT / "outputs" / "b16_four_task_pilot"
PILOT_PROTOCOL_PATH = PILOT_DIR / "protocol.json"
FREEZE_PATH = PILOT_DIR / "candidate_freeze.json"
RESULTS_PATH = PILOT_DIR / "results.json"
CONFIG_PATH = ROOT / "configs" / "model_clip_b16_qwen05.yaml"
OUTPUT_ROOT = ROOT / "outputs" / "independent_eval_v1"
SCORER_PATH = ROOT / "evaluation" / "metrics.py"
B16_WEIGHTS_SHA256 = "ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0"
TASKS = ("existence", "counting", "attribute", "listing")
N_QUESTIONS = 400
BATCH_SIZE = 4
MAX_NEW_TOKENS = 24

PRETRAINED_FILES = {
    "clip": (
        "config.json", "preprocessor_config.json", "pytorch_model.bin",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt",
    ),
    "qwen": (
        "config.json", "generation_config.json", "model.safetensors",
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    ),
}
PIPELINE_FILES = (
    "scripts/export_independent_predictions.py",
    "models/multimodal_model.py", "models/vision_encoder.py", "models/projector.py",
    "datasets/collator.py", "datasets/dataset.py", "evaluation/metrics.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sidecar_sha256(path: Path) -> str:
    expected = path.with_name(path.name + ".sha256").read_text(encoding="ascii").split()[0]
    actual = sha256(path)
    if expected != actual:
        raise ValueError(f"frozen SHA-256 mismatch: {path}")
    return actual


def project_path(raw: str, *, within: Path) -> Path:
    """Accept only project-relative paths contained in ``within``."""
    if not isinstance(raw, str) or not raw or "\\" in raw or ":" in raw:
        raise ValueError(f"unsafe project-relative path: {raw!r}")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or PureWindowsPath(raw).is_absolute() or any(
        part in ("", ".", "..") for part in raw.split("/")
    ):
        raise ValueError(f"unsafe project-relative path: {raw!r}")
    result = (ROOT / Path(*pure.parts)).resolve(strict=True)
    if not result.is_relative_to(within.resolve(strict=True)):
        raise ValueError(f"path escapes expected directory: {raw!r}")
    return result


def load_freeze(pilot_protocol_hash: str) -> tuple[dict, Path, dict]:
    if not FREEZE_PATH.is_file() or not RESULTS_PATH.is_file():
        raise FileNotFoundError("candidate_freeze.json and results.json are required before any holdout inference")
    freeze, results = json_object(FREEZE_PATH), json_object(RESULTS_PATH)
    if (freeze.get("protocol_sha256") != pilot_protocol_hash or
        results.get("protocol_sha256") != pilot_protocol_hash):
        raise ValueError("candidate freeze/results refer to a different pilot protocol")
    if not isinstance(freeze.get("selected_on"), str) or not freeze["selected_on"].strip():
        raise ValueError("candidate freeze must name its old-data selection source")
    candidate = project_path(freeze.get("checkpoint"), within=PILOT_DIR)
    if candidate.suffix != ".pt" or freeze.get("checkpoint_sha256") != sha256(candidate):
        raise ValueError("candidate checkpoint SHA-256 differs from its freeze record")
    selected = project_path(results.get("selected_checkpoint"), within=PILOT_DIR)
    if selected != candidate:
        raise ValueError("candidate freeze and pilot results select different checkpoints")
    selected_step = results.get("selected_step")
    if type(selected_step) is not int or selected_step not in (256, 512):
        raise ValueError("pilot results require an eligible step-256/512 selection")
    selected_validation = (results.get("validation") or {}).get(str(selected_step))
    if (results.get("independent_holdout_used") is not False or
        selected_step not in (results.get("eligible_steps") or []) or
        not isinstance(selected_validation, dict) or
        selected_validation.get("checkpoint_sha256") != freeze["checkpoint_sha256"]):
        raise ValueError("candidate is not the frozen old-validation pilot selection")
    return freeze, candidate, results


def load_holdout() -> tuple[dict, dict, list[dict], dict[str, str]]:
    protocol_path = DATA_DIR / "protocol.json"
    manifest_path = DATA_DIR / "manifest.json"
    questions_path = DATA_DIR / "questions.jsonl"
    protocol_hash = sidecar_sha256(protocol_path)
    manifest_hash = sidecar_sha256(manifest_path)
    protocol, manifest = json_object(protocol_path), json_object(manifest_path)
    question_hash = sha256(questions_path)
    if (protocol.get("status") != "sealed_before_model_inference" or
        protocol.get("manifest_sha256") != manifest_hash or
        protocol.get("questions_sha256") != question_hash or
        manifest.get("question_sha256") != question_hash or
        protocol.get("scorer_sha256") != sha256(SCORER_PATH) or
        manifest.get("scorer_sha256") != protocol.get("scorer_sha256") or
        protocol.get("tasks") != list(TASKS) or
        manifest.get("n") != N_QUESTIONS or
        manifest.get("task_counts") != dict.fromkeys(TASKS, N_QUESTIONS // len(TASKS))):
        raise ValueError("holdout protocol, manifest, questions, or scorer differs from frozen values")
    inference = protocol.get("inference") or {}
    if (inference.get("decoding") != "greedy" or inference.get("do_sample") is not False or
        inference.get("max_new_tokens") != MAX_NEW_TOKENS or
        inference.get("generation_batch") != BATCH_SIZE):
        raise ValueError("frozen inference settings differ from this exporter")
    rows = []
    with questions_path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank question line {number}")
            source = json.loads(line)
            if not isinstance(source, dict):
                raise ValueError(f"question line {number} is not an object")
            # Discard answer/meta immediately; neither reaches the model or
            # any generated artifact.  The whole file is still hash-checked.
            rows.append({key: source.get(key) for key in ("id", "image", "task", "question")})
    assets = manifest.get("assets")
    if len(rows) != N_QUESTIONS or not isinstance(assets, list) or len(assets) != N_QUESTIONS:
        raise ValueError("holdout requires exactly 400 questions and manifest assets")
    if len({row.get("id") for row in rows}) != N_QUESTIONS:
        raise ValueError("duplicate or missing question IDs")
    if len({row.get("image") for row in rows}) != N_QUESTIONS:
        raise ValueError("duplicate or missing question images")
    if Counter(row.get("task") for row in rows) != dict.fromkeys(TASKS, N_QUESTIONS // len(TASKS)):
        raise ValueError("holdout task allocation changed")
    images_dir = DATA_DIR / "images"
    annotations_dir = DATA_DIR / "annotations"
    clean = []
    for row, asset in zip(rows, assets):
        if not isinstance(asset, dict) or row.get("id") != asset.get("id"):
            raise ValueError("question/manifest order or ID mismatch")
        if not isinstance(row.get("question"), str) or not row["question"]:
            raise ValueError(f"invalid question: {row.get('id')}")
        stem = asset.get("stem")
        expected_rel = f"data/independent_eval_v1/images/{stem}.jpg"
        if not isinstance(stem, str) or row.get("image") != expected_rel:
            raise ValueError(f"unexpected image path: {row.get('id')}")
        image_path = project_path(expected_rel, within=images_dir)
        if image_path.parent != images_dir.resolve(strict=True):
            raise ValueError(f"image must be directly inside sealed images/: {row['id']}")
        xml_path = (annotations_dir / f"{stem}.xml").resolve(strict=True)
        if xml_path.parent != annotations_dir.resolve(strict=True):
            raise ValueError(f"annotation must be directly inside sealed annotations/: {row['id']}")
        if sha256(image_path) != asset.get("image_sha256") or \
           sha256(xml_path) != asset.get("annotation_sha256"):
            raise ValueError(f"sealed image/XML SHA-256 mismatch: {row['id']}")
        # The model-facing record intentionally contains no gold answer or meta.
        clean.append({"id": row["id"], "image": image_path, "question": row["question"]})
    hashes = {"protocol_sha256": protocol_hash, "manifest_sha256": manifest_hash,
              "questions_sha256": question_hash}
    return protocol, manifest, clean, hashes


def verify_pretrained() -> dict[str, str]:
    clip = ROOT / "checkpoints" / "pretrained" / "clip-vit-base-patch16"
    qwen = ROOT / "checkpoints" / "pretrained" / "qwen2.5-0.5b-instruct"
    roots = {"clip": clip, "qwen": qwen}
    hashes = {}
    for family, names in PRETRAINED_FILES.items():
        for name in names:
            path = roots[family] / name
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(f"missing pretrained asset: {path}")
            hashes[f"{family}/{name}"] = sha256(path)
    if hashes["clip/pytorch_model.bin"] != B16_WEIGHTS_SHA256:
        raise ValueError("CLIP B/16 weights differ from the pinned official download")
    clip_cfg = json_object(clip / "config.json")["vision_config"]
    qwen_cfg = json_object(qwen / "config.json")
    if (clip_cfg.get("image_size"), clip_cfg.get("patch_size"), clip_cfg.get("hidden_size")) != (224, 16, 768):
        raise ValueError("CLIP is not the expected ViT-B/16")
    if (qwen_cfg.get("model_type"), qwen_cfg.get("hidden_size"),
        qwen_cfg.get("num_hidden_layers")) != ("qwen2", 896, 24):
        raise ValueError("Qwen is not the expected 0.5B architecture")
    return hashes


def verify_model_config() -> tuple[object, str]:
    from models.multimodal_model import config_from_yaml

    cfg = config_from_yaml(CONFIG_PATH)
    if (Path(cfg.clip_path).resolve() != (ROOT / "checkpoints/pretrained/clip-vit-base-patch16").resolve() or
        Path(cfg.llm_path).resolve() != (ROOT / "checkpoints/pretrained/qwen2.5-0.5b-instruct").resolve() or
        cfg.strategy != "projector_only" or cfg.num_image_token != 196 or
        cfg.projector_depth != 2 or cfg.projector_hidden != 2048 or
        cfg.max_seq_len != 768 or not cfg.freeze_vision or not cfg.freeze_llm):
        raise ValueError("model configuration differs from the fixed B/16 projector-only architecture")
    cfg.device = "cuda"
    return cfg, sha256(CONFIG_PATH)


def load_projector(model: object, checkpoint: Path, expected_step: int | None) -> int:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("projector"), dict):
        raise ValueError(f"invalid projector checkpoint: {checkpoint}")
    if payload.get("lora") or payload.get("trainable_extra"):
        raise ValueError("checkpoint has extra trainable weights incompatible with fixed projector-only evaluation")
    step = payload.get("global_step")
    if type(step) is not int or step < 0 or (expected_step is not None and step != expected_step):
        raise ValueError("checkpoint step differs from frozen pilot selection")
    model.projector.load_state_dict(payload["projector"], strict=True)
    return step


def generate_predictions(model: object, records: list[dict]) -> tuple[list[dict], str]:
    import torch
    from PIL import Image

    from datasets.collator import MultipleImageCollator, pad_to_square
    from datasets.dataset import JsonlVLDataset

    collator = MultipleImageCollator(
        processor=model.processor, tokenizer=model.tokenizer,
        max_length=model.cfg.max_seq_len, num_image_token=model.num_image_token,
    )
    if model.vision is None or model.processor is None:
        raise ValueError("B/16 vision encoder and processor must both be loaded")
    model.eval()
    # Preserve the low-memory path used by the historical local B/16 checks:
    # encode each image once, keep raw CLIP features on CPU, and release the
    # vision tower's GPU memory before autoregressive Qwen generation.
    feature_cache = {}
    with torch.no_grad():
        for start in range(0, len(records), BATCH_SIZE):
            chunk = records[start:start + BATCH_SIZE]
            pictures = []
            try:
                for rec in chunk:
                    with Image.open(rec["image"]) as source:
                        pictures.append(source.convert("RGB"))
                pixels = model.processor(
                    images=[pad_to_square(picture) for picture in pictures],
                    return_tensors="pt",
                )["pixel_values"].to(model.device_)
                features = model.vision(pixels).detach().cpu()
                if features.size(0) != len(chunk) or not torch.isfinite(features).all():
                    raise ValueError("invalid frozen CLIP features")
                for rec, feature in zip(chunk, features):
                    feature_cache[rec["id"]] = feature
            finally:
                for picture in pictures:
                    picture.close()
            if len(feature_cache) % 40 == 0:
                print(f"{len(feature_cache)}/{len(records)} images encoded")
    del pixels, features
    model.vision.cpu()
    torch.cuda.empty_cache()
    if len(feature_cache) != N_QUESTIONS:
        raise ValueError("frozen CLIP encoding did not cover exactly 400 images")

    outputs: list[dict] = []
    prompt_digest = hashlib.sha256()
    with torch.no_grad():
        for start in range(0, len(records), BATCH_SIZE):
            chunk = records[start:start + BATCH_SIZE]
            pictures = []
            try:
                for rec in chunk:
                    with Image.open(rec["image"]) as source:
                        pictures.append(source.convert("RGB"))
                batch = collator([
                    {"image": picture,
                     "question": JsonlVLDataset.build_question(rec["question"]),
                     "answer": ""}
                    for rec, picture in zip(chunk, pictures)
                ])
                ids, mask = batch["input_ids"], batch["attention_mask"]
                eos = int(model.tokenizer.eos_token_id)
                for i, rec in enumerate(chunk):
                    answer_start = int(batch["answer_starts"][i])
                    valid = int(mask[i].sum().item())
                    tail = ids[i, answer_start:valid].tolist()
                    if tail != [eos]:
                        raise ValueError(f"gold-free prompt invariant failed: {rec['id']}")
                    prefix = ids[i, :answer_start].tolist()
                    prompt_digest.update(json.dumps(
                        [rec["id"], prefix], separators=(",", ":")
                    ).encode("utf-8"))
                    prompt_digest.update(b"\n")
                raw_features = torch.stack(
                    [feature_cache[rec["id"]] for rec in chunk]
                ).to(model.device_, dtype=next(model.projector.parameters()).dtype)
                texts = model.generate_text(
                    input_ids=ids, attention_mask=mask,
                    visual_features=raw_features,
                    answer_start=torch.tensor(batch["answer_starts"], device=model.device_),
                    do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                    micro_batch=BATCH_SIZE,
                )
                if len(texts) != len(chunk) or any(not isinstance(x, str) for x in texts):
                    raise ValueError("generation returned an incomplete batch")
                outputs.extend({"id": rec["id"], "prediction": text}
                               for rec, text in zip(chunk, texts))
            finally:
                for picture in pictures:
                    picture.close()
            if len(outputs) % 40 == 0 or len(outputs) == len(records):
                print(f"{len(outputs)}/{len(records)} predictions generated")
    if len(outputs) != N_QUESTIONS or len({row["id"] for row in outputs}) != N_QUESTIONS:
        raise ValueError("generation did not produce exactly 400 unique predictions")
    return outputs, prompt_digest.hexdigest()


def publish(role: str, predictions: list[dict], provenance: dict) -> Path:
    """Publish both files together through an atomic directory rename."""
    import shutil

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_ROOT / role
    if destination.exists():
        raise FileExistsError(f"role already exported; refusing to overwrite: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{role}.partial-", dir=OUTPUT_ROOT))
    if not stage.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        raise RuntimeError("staging directory escaped the project outputs directory")
    try:
        predictions_path = stage / "predictions.jsonl"
        with predictions_path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        provenance["predictions_sha256"] = sha256(predictions_path)
        with (stage / "provenance.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(provenance, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(stage, destination)
    finally:
        if stage.exists():
            if not stage.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
                raise RuntimeError("refusing to clean a staging directory outside outputs")
            shutil.rmtree(stage)
    return destination / "predictions.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("baseline", "candidate"))
    args = parser.parse_args()
    role = args.role
    final_dir = OUTPUT_ROOT / role
    if final_dir.exists():
        parser.error(f"{final_dir} already exists; the one-time export cannot be overwritten")

    # Complete all freeze, benchmark and input checks before loading a model.
    # The candidate must be frozen before opening the question file at all.
    pilot_protocol_hash = sha256(PILOT_PROTOCOL_PATH)
    freeze, candidate, pilot_results = load_freeze(pilot_protocol_hash)
    frozen_protocol_hash = sidecar_sha256(DATA_DIR / "protocol.json")
    protocol, manifest, records, holdout_hashes = load_holdout()
    if holdout_hashes["protocol_sha256"] != frozen_protocol_hash:
        raise ValueError("holdout protocol changed after candidate freeze check")
    pilot_protocol = json_object(PILOT_PROTOCOL_PATH)
    if (pilot_protocol.get("independent_holdout_used") is not False or
        pilot_protocol.get("source_sha256") != protocol.get("baseline_checkpoint_sha256") or
        pilot_results.get("source_sha256_after") != pilot_protocol.get("source_sha256") or
        pilot_protocol.get("model_config_sha256") != sha256(CONFIG_PATH) or
        pilot_protocol.get("scorer_sha256") != sha256(SCORER_PATH)):
        raise ValueError("pilot source/config/scorer provenance differs from the frozen baseline")
    baseline = project_path(protocol.get("baseline_checkpoint"), within=ROOT / "outputs")
    if sha256(baseline) != protocol.get("baseline_checkpoint_sha256"):
        raise ValueError("baseline checkpoint differs from the frozen protocol")
    checkpoint = baseline if role == "baseline" else candidate
    checkpoint_sha = sha256(checkpoint)
    pretrained_hashes = verify_pretrained()
    cfg, config_hash = verify_model_config()
    pipeline_hashes = {name: sha256(ROOT / name) for name in PIPELINE_FILES}
    input_hashes = {
        **holdout_hashes,
        "candidate_freeze_sha256": sha256(FREEZE_PATH),
        "pilot_results_sha256": sha256(RESULTS_PATH),
        "pilot_protocol_sha256": pilot_protocol_hash,
        "baseline_checkpoint_sha256": sha256(baseline),
        "candidate_checkpoint_sha256": sha256(candidate),
        "config_sha256": config_hash,
    }

    import torch
    from models.multimodal_model import MiniVLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the sealed one-time evaluation")
    environment = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "transformers": version("transformers"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    print(f"Loading {role} checkpoint; 400 sealed questions will be generated once")
    model = MiniVLM(cfg)
    checkpoint_step = load_projector(
        model, checkpoint, pilot_results["selected_step"] if role == "candidate" else None
    )
    predictions, prompt_hash = generate_predictions(model, records)

    # A run is invalid if an input was modified while generation was underway.
    if (sha256(DATA_DIR / "protocol.json") != input_hashes["protocol_sha256"] or
        sha256(DATA_DIR / "manifest.json") != input_hashes["manifest_sha256"] or
        sha256(DATA_DIR / "questions.jsonl") != input_hashes["questions_sha256"] or
        sha256(FREEZE_PATH) != input_hashes["candidate_freeze_sha256"] or
        sha256(RESULTS_PATH) != input_hashes["pilot_results_sha256"] or
        sha256(PILOT_PROTOCOL_PATH) != input_hashes["pilot_protocol_sha256"] or
        sha256(baseline) != input_hashes["baseline_checkpoint_sha256"] or
        sha256(candidate) != input_hashes["candidate_checkpoint_sha256"] or
        sha256(CONFIG_PATH) != input_hashes["config_sha256"] or
        sha256(checkpoint) != checkpoint_sha or
        any(sha256(rec["image"]) != asset["image_sha256"]
            for rec, asset in zip(records, manifest["assets"])) or
        any(sha256(DATA_DIR / "annotations" / f"{asset['stem']}.xml") != asset["annotation_sha256"]
            for asset in manifest["assets"]) or
        verify_pretrained() != pretrained_hashes or
        any(sha256(ROOT / name) != expected for name, expected in pipeline_hashes.items())):
        raise ValueError("an input changed during generation; refusing to publish predictions")
    other = "candidate" if role == "baseline" else "baseline"
    counterpart = OUTPUT_ROOT / other / "provenance.json"
    if counterpart.parent.exists() and not counterpart.is_file():
        raise ValueError("other role has an incomplete export directory")
    if counterpart.is_file():
        previous = json_object(counterpart)
        other_predictions = counterpart.parent / "predictions.jsonl"
        if (previous.get("status") != "complete" or
            not other_predictions.is_file() or
            sha256(other_predictions) != previous.get("predictions_sha256") or
            previous.get("holdout") != holdout_hashes or
            previous.get("candidate_freeze_sha256") != input_hashes["candidate_freeze_sha256"] or
            previous.get("pilot_results_sha256") != input_hashes["pilot_results_sha256"] or
            previous.get("pilot_protocol_sha256") != input_hashes["pilot_protocol_sha256"] or
            previous.get("config_sha256") != config_hash or
            previous.get("pretrained_assets_sha256") != pretrained_hashes or
            previous.get("pipeline_files_sha256") != pipeline_hashes or
            previous.get("prompt_tokens_sha256") != prompt_hash or
            previous.get("environment") != environment or
            previous.get("decoding") != {"method": "greedy", "do_sample": False,
                                         "max_new_tokens": MAX_NEW_TOKENS, "batch_size": BATCH_SIZE}):
            raise ValueError("baseline/candidate inference pipelines differ; refusing publication")
    provenance = {
        "status": "complete", "role": role, "n": N_QUESTIONS,
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": checkpoint_sha, "checkpoint_step": checkpoint_step,
        "candidate_freeze_sha256": input_hashes["candidate_freeze_sha256"],
        "pilot_results_sha256": input_hashes["pilot_results_sha256"],
        "pilot_protocol_sha256": input_hashes["pilot_protocol_sha256"],
        "candidate_selected_on": freeze["selected_on"],
        "holdout": holdout_hashes,
        "asset_image_sha256": [asset["image_sha256"] for asset in manifest["assets"]],
        "config_sha256": config_hash,
        "pretrained_assets_sha256": pretrained_hashes,
        "pipeline_files_sha256": pipeline_hashes,
        "prompt_tokens_sha256": prompt_hash,
        "decoding": {"method": "greedy", "do_sample": False,
                     "max_new_tokens": MAX_NEW_TOKENS, "batch_size": BATCH_SIZE},
        "environment": environment,
    }
    path = publish(role, predictions, provenance)
    print(f"Complete, immutable export: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
