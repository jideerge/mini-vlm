"""Local B/16 four-task continuation, selecting only on the old validation split.

The independent_eval_v1 holdout is deliberately absent from this script.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from scripts.evaluate_spatial_flip import (
    ROOT, MiniVLM, MultipleImageCollator, Trainer, TrainingConfig,
    config_from_yaml, load_one, load_records, sha256, split_records, torch,
)

SOURCE = ROOT / "outputs/b16_spatial_clean_cloud/checkpoint_best.pt"
DATA = ROOT / "data/processed/spatial_clean_9333.jsonl"
SPLITS = ROOT / "data/processed/splits.json"
CONFIG = ROOT / "configs/model_clip_b16_qwen05.yaml"
OUT = ROOT / "outputs/b16_four_task_pilot"
TASKS = ("existence", "counting", "attribute", "listing")
SEED = 42
PER_TASK = 512
STEPS = (0, 256, 512)
MAX_STEPS = STEPS[-1]
STRATUM_QUOTAS = {
    "existence": {"negative": 256, "positive": 256},
    "counting": {"one": 307, "two_plus": 205},
    "attribute": {"single": 410, "multi": 102},
    "listing": {"single": 384, "multi": 128},
}


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def rates(result: dict) -> dict[str, float]:
    content = result["content_per_task"]
    if set(content) != set(TASKS):
        raise ValueError(f"wrong validation tasks: {list(content)}")
    return {task: content[task]["correct"] for task in TASKS}


def macro(values: dict[str, float]) -> float:
    return sum(values[t] for t in TASKS) / len(TASKS)


def main() -> None:
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA GPU is required")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError(f"pilot already exists: {OUT}")
    source_hash = sha256(SOURCE)
    if source_hash != "2a4ce406e54cdb4798b3afa1af3ef2dd7ed59ebd62a7b65b5921e7f155c6d575":
        raise ValueError("source checkpoint differs from the frozen B/16 baseline")
    payload = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if payload.get("global_step") != 840 or "lora" in payload or "trainable_extra" in payload:
        raise ValueError("expected the original projector-only step-840 checkpoint")

    parts = split_records(load_records(DATA), SPLITS)
    voc_index = {entry["stem"]: entry for entry in json.loads(
        (ROOT / "data/processed/voc_index.json").read_text(encoding="utf-8"))["entries"]}

    def stratum(record: dict) -> str:
        task, meta = record["task"], record["meta"]
        if task == "existence":
            return "positive" if meta["label"] == 1 else "negative"
        if task == "counting":
            return "two_plus" if meta["count"] >= 2 else "one"
        if task == "listing":
            return "multi" if len(meta["classes"]) >= 2 else "single"
        classes = voc_index[Path(record["image"]).stem]["classes"]
        return "multi" if len(classes) >= 2 else "single"

    selected = []
    for offset, task in enumerate(TASKS):
        for sub_index, (group, quota) in enumerate(STRATUM_QUOTAS[task].items()):
            pool = sorted((r for r in parts["train"] if r["task"] == task
                           and stratum(r) == group), key=lambda r: r["id"])
            random.Random(SEED + offset * 10 + sub_index).shuffle(pool)
            if len(pool) < quota:
                raise ValueError(f"insufficient {task}/{group} examples")
            selected.extend(pool[:quota])
    val = [r for r in parts["val"] if r["task"] in TASKS]
    if len(selected) != 4 * PER_TASK or len(val) != 399:
        raise ValueError("unexpected four-task train/validation size")
    if {r["image"] for r in selected} & {r["image"] for r in val}:
        raise ValueError("train/validation image leak")
    if len({r["id"] for r in selected + val}) != len(selected) + len(val):
        raise ValueError("duplicate train/validation question ID")

    protocol = {
        "name": "B16 four-task local continuation", "created_on": "2026-09-27",
        "seed": SEED, "tasks": list(TASKS), "per_task_train": PER_TASK,
        "stratum_quotas": STRATUM_QUOTAS,
        "train_ids": [r["id"] for r in selected], "validation_ids": [r["id"] for r in val],
        "source_checkpoint": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": source_hash, "data_sha256": sha256(DATA),
        "splits_sha256": sha256(SPLITS), "model_config_sha256": sha256(CONFIG),
        "scorer_sha256": sha256(ROOT / "evaluation/metrics.py"),
        "optimization": {"projector_only": True, "steps": MAX_STEPS,
                         "batch_size": 2, "grad_accum": 4,
                         "projector_lr": 2e-4, "warmup_steps": 16,
                         "candidate_steps": list(STEPS)},
        "selection": "On 399 old validation questions, require macro content gain >=5pp and no task regression >3pp versus step0. Among eligible step256/512 choose highest macro accuracy; ties prefer earlier step. If none qualifies, retain baseline and do not open the independent holdout.",
        "validation_generation": {"greedy": True, "max_new_tokens": 24,
                                  "batch_size": 4, "micro_batch": 4},
        "independent_holdout_used": False,
    }
    write(OUT / "protocol.json", protocol)

    config = config_from_yaml(CONFIG)
    config.device = "cuda"
    model = MiniVLM(config)
    model.projector.load_state_dict(payload["projector"], strict=True)
    if any(not name.startswith("projector.") for name, p in model.named_parameters()
           if p.requires_grad):
        raise ValueError("pilot must train only the projector")
    collator = MultipleImageCollator(
        processor=model.processor, tokenizer=model.tokenizer,
        max_length=config.max_seq_len, num_image_token=model.num_image_token)
    tcfg = TrainingConfig(device="cuda", batch_size=2, grad_accum_steps=4,
                          max_steps=MAX_STEPS, projector_lr=2e-4,
                          warmup_steps=16, eval_every=256, log_every=32,
                          cache_features_on_gpu=False, save_best=False,
                          output_dir=str(OUT))
    trainer = Trainer(model, collator, selected, val, load_one, tcfg)
    # The frozen encoder is run once per distinct image. Its CPU feature cache
    # avoids repeated CLIP passes and leaves GPU memory for Qwen + projector.
    unique = list({r["image"]: r for r in selected + val}.values())
    started = time.perf_counter()
    trainer.precompute_features(unique)
    if len(trainer.feature_cache) != len(unique):
        raise ValueError("incomplete visual feature cache")
    model.vision.cpu()
    torch.cuda.empty_cache()

    def evaluate_old_validation() -> dict:
        trainer.cfg.batch_size = 4
        try:
            result = trainer.evaluate_generation(
                val, max_new_tokens=24, micro_batch=4, detail_limit=None)
        finally:
            trainer.cfg.batch_size = 2
        return result

    baseline = evaluate_old_validation()
    baseline_rates = rates(baseline)
    baseline_correct = sum(int(d["content"]["correct"]) for d in baseline["details"])
    if baseline_correct != 324:
        raise ValueError(f"baseline old-validation score changed: {baseline_correct}/399")
    write(OUT / "validation_step0.json", baseline)
    print(f"BASELINE {baseline_correct}/399 macro={macro(baseline_rates):.4f}", flush=True)

    def on_log(message: str) -> None:
        print(message, flush=True)
        if "val_loss=" in message:
            model.llm.eval()
            model.vision.eval()
            if trainer.global_step == 256:
                trainer.save_checkpoint(OUT / "checkpoint_step256.pt")

    trainer.train(log_fn=on_log)
    validation = {0: {"rates": baseline_rates,
                      "macro": macro(baseline_rates), "correct": baseline_correct}}
    for step, checkpoint in ((256, OUT / "checkpoint_step256.pt"),
                             (512, OUT / "checkpoint_last.pt")):
        candidate = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if candidate.get("global_step") != step or "lora" in candidate:
            raise ValueError(f"unexpected step-{step} checkpoint")
        model.projector.load_state_dict(candidate["projector"], strict=True)
        result = evaluate_old_validation()
        result_rates = rates(result)
        correct = sum(int(d["content"]["correct"]) for d in result["details"])
        validation[step] = {"rates": result_rates, "macro": macro(result_rates),
                            "correct": correct, "checkpoint_sha256": sha256(checkpoint)}
        write(OUT / f"validation_step{step}.json", result)
        print(f"STEP {step} {correct}/399 macro={macro(result_rates):.4f}", flush=True)

    eligible = [step for step in (256, 512)
                if validation[step]["macro"] >= validation[0]["macro"] + .05 - 1e-12
                and all(validation[step]["rates"][task] >= baseline_rates[task] - .03 - 1e-12
                        for task in TASKS)]
    selected_step = min(eligible, key=lambda step: (-validation[step]["macro"], step)) if eligible else 0
    result = {"protocol_sha256": sha256(OUT / "protocol.json"),
              "source_sha256_after": sha256(SOURCE),
              "training_elapsed_s": time.perf_counter() - started,
              "validation": validation, "eligible_steps": eligible,
              "selected_step": selected_step,
              "selected_checkpoint": (str((OUT / f"checkpoint_step{selected_step}.pt").relative_to(ROOT)).replace("\\", "/")
                                      if selected_step == 256 else
                                      str((OUT / "checkpoint_last.pt").relative_to(ROOT)).replace("\\", "/")
                                      if selected_step == 512 else None),
              "independent_holdout_used": False}
    if result["source_sha256_after"] != source_hash:
        raise ValueError("source checkpoint changed during local pilot")
    write(OUT / "results.json", result)
    if selected_step:
        freeze = {"checkpoint": result["selected_checkpoint"],
                  "checkpoint_sha256": validation[selected_step]["checkpoint_sha256"],
                  "selected_on": "old validation 399, not independent holdout",
                  "protocol_sha256": result["protocol_sha256"]}
        write(OUT / "candidate_freeze.json", freeze)
    print("DONE selected_step", selected_step, "holdout untouched", flush=True)


if __name__ == "__main__":
    main()
