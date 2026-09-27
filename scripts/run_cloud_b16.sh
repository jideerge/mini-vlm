#!/usr/bin/env bash
# Run from an extracted B/16 cloud bundle on an NVIDIA Linux host.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export MINIVLM_IMAGE_ROOT="${MINIVLM_IMAGE_ROOT:-$ROOT/images}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export MPLBACKEND=Agg
mkdir -p outputs/b16_spatial_clean_cloud report/figures
# Optional shared-storage backup, including on a failed or interrupted run.
backup_results() {
  if [[ -n "${MINIVLM_PERSIST_ROOT:-}" ]]; then
    mkdir -p "$MINIVLM_PERSIST_ROOT"
    for folder in outputs/b16_spatial_clean_cloud outputs/b16_spatial_clean_cloud_smoke outputs/b16_spatial_clean_cloud_eval; do
      if [[ -d "$folder" ]]; then cp -a "$folder" "$MINIVLM_PERSIST_ROOT/"; fi
    done
  fi
}
trap backup_results EXIT
printf 'cloud run start: %s\n' "$(date -Is)"
"$PYTHON_BIN" -B scripts/verify_cloud_inputs.py
"$PYTHON_BIN" -B -m unittest discover -s tests -p 'test_*.py'
"$PYTHON_BIN" -B scripts/train.py --config configs/model_clip_b16_qwen05.yaml \
  --data data/processed/spatial_clean_9333.jsonl \
  --out outputs/b16_spatial_clean_cloud_preflight --dry-run --no-gpu-cache
"$PYTHON_BIN" -B -u scripts/train.py --config configs/model_clip_b16_qwen05.yaml \
  --data data/processed/spatial_clean_smoke_1k.jsonl \
  --out outputs/b16_spatial_clean_cloud_smoke --steps 50 --batch 4 --accum 4 \
  --warmup 5 --eval-every 25 --log-every 10 --gen-samples 4 --no-gpu-cache \
  2>&1 | tee outputs/b16_spatial_clean_cloud/smoke.log
"$PYTHON_BIN" -B - <<'PY_GATE'
import json
from pathlib import Path
h=json.loads(Path('outputs/b16_spatial_clean_cloud_smoke/history.json').read_text())
vals=[x['val_loss'] for x in h if x.get('val_loss') is not None]
assert len(vals)>=2 and vals[-1]<vals[0],f'smoke loss did not decrease: {vals}'
print('smoke validation loss:',vals)
PY_GATE
timeout --signal=TERM --kill-after=30s 210m "$PYTHON_BIN" -B -u scripts/train.py --config configs/model_clip_b16_qwen05.yaml \
  --data data/processed/spatial_clean_9333.jsonl \
  --out outputs/b16_spatial_clean_cloud --steps 1200 --batch 4 --accum 8 \
  --lr 1e-3 --no-gpu-cache \
  2>&1 | tee outputs/b16_spatial_clean_cloud/train.log
"$PYTHON_BIN" -B -u scripts/evaluate.py --config configs/model_clip_b16_qwen05.yaml \
  --data data/processed/spatial_clean_9333.jsonl \
  --out outputs/b16_spatial_clean_cloud_eval \
  --ckpt "$ROOT/outputs/b16_spatial_clean_cloud/checkpoint_best.pt" --split test \
  2>&1 | tee outputs/b16_spatial_clean_cloud/evaluate.log
printf 'cloud run end: %s\n' "$(date -Is)"
printf '%s\n' 'Done. Download outputs/b16_spatial_clean_cloud, outputs/b16_spatial_clean_cloud_eval, and outputs/b16_spatial_clean_cloud_smoke before stopping the instance.'

