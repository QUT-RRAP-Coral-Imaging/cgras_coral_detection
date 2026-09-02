#!/bin/bash -l

#PBS -N train_yolov8_scales
#PBS -l select=1:ncpus=8:ngpus=1:mem=64GB:gpu_id=A100
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -j oe

set -euo pipefail

cd "$PBS_O_WORKDIR"
pwd

REPO_ROOT="${REPO_ROOT:-Corals/cgras_settler_counter}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/segmenter/config/cslics_2025.yaml}"
MODELS="${MODELS:-yolov8n.pt yolov8s.pt yolov8m.pt yolov8l.pt yolov8x.pt}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/analysis/scale_runs}"
BASE_SEED="${BASE_SEED:-42}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_GROUP="yolov8_scales_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/${RUN_GROUP}_summary.log"
METRICS_CSV="${OUTPUT_DIR}/${RUN_GROUP}_metrics.csv"
PLOTS_DIR="${OUTPUT_DIR}/${RUN_GROUP}_plots"

if [ -f /home/wardlewo/miniforge3/bin/activate ]; then
  source /home/wardlewo/miniforge3/bin/activate cgras
elif [ -f /home/wardlewo/mambaforge/bin/activate ]; then
  source /home/wardlewo/mambaforge/bin/activate cgras
else
  echo "Could not find conda activate script for known users"
  exit 1
fi

if [ ! -f "$BASE_CONFIG" ]; then
  echo "Missing base config: $BASE_CONFIG"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Run group: ${RUN_GROUP}" | tee "$LOG_FILE"
echo "Base config: ${BASE_CONFIG}" | tee -a "$LOG_FILE"
echo "Models: ${MODELS}" | tee -a "$LOG_FILE"
echo "Base seed: ${BASE_SEED}" | tee -a "$LOG_FILE"
echo "Eval split: ${EVAL_SPLIT}" | tee -a "$LOG_FILE"
echo "Output dir: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"
echo "Metrics CSV: ${METRICS_CSV}" | tee -a "$LOG_FILE"
echo "Plots dir: ${PLOTS_DIR}" | tee -a "$LOG_FILE"

echo "model_size,run_name,model_path,best_weights,precision,recall,map50,map50_95" > "$METRICS_CSV"

which python | tee -a "$LOG_FILE"
nvidia-smi | tee -a "$LOG_FILE" || true

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "Initial CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$LOG_FILE"
  if [[ "$CUDA_VISIBLE_DEVICES" == GPU-* ]] || [[ "$CUDA_VISIBLE_DEVICES" == MIG-* ]]; then
    echo "Detected UUID-form CUDA_VISIBLE_DEVICES; normalizing to index 0 for PyTorch compatibility." | tee -a "$LOG_FILE"
    export CUDA_VISIBLE_DEVICES=0
  fi
fi

python - <<'PY' | tee -a "$LOG_FILE"
import os
import torch
print(f"Post-normalization CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
PY

if ! python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)
PY
then
  echo "ERROR: PyTorch still cannot see a CUDA device after CUDA_VISIBLE_DEVICES normalization." | tee -a "$LOG_FILE"
  echo "Likely environment issue (e.g., torch/cuda mismatch). Aborting before training." | tee -a "$LOG_FILE"
  exit 1
fi

for model_name in $MODELS; do
  model_label="${model_name%.pt}"
  model_idx="${MODEL_INDEX:-0}"
  RUN_SEED=$((BASE_SEED + model_idx))
  RUN_TAG="${RUN_GROUP}_${model_label}"
  TMP_CONFIG="${OUTPUT_DIR}/${RUN_TAG}.yaml"

  echo "==================================================" | tee -a "$LOG_FILE"
  echo "[Model ${model_name}] Preparing config and passing via --config: ${TMP_CONFIG} (seed=${RUN_SEED})" | tee -a "$LOG_FILE"

  python - "$BASE_CONFIG" "$TMP_CONFIG" "$RUN_TAG" "$model_name" "$RUN_SEED" <<'PY'
import pathlib
import sys
import yaml

base_cfg = pathlib.Path(sys.argv[1])
out_cfg = pathlib.Path(sys.argv[2])
run_tag = sys.argv[3]
model_name = sys.argv[4]
run_seed = int(sys.argv[5])

with base_cfg.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["name"] = run_tag
cfg["deterministic"] = False
cfg["seed"] = run_seed

base_model = str(cfg.get("model_path", "")).strip()
if not base_model:
    raise ValueError("model_path missing from base config")

cfg["model_path"] = model_name

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  echo "[Model ${model_name}] Training..." | tee -a "$LOG_FILE"
  python -u "${REPO_ROOT}/segmenter/scripts/train.py" --config "$TMP_CONFIG" 2>&1 | tee "${OUTPUT_DIR}/${RUN_TAG}.train.log"

  echo "[Model ${model_name}] Evaluating best checkpoint..." | tee -a "$LOG_FILE"
  python - "$TMP_CONFIG" "$model_name" "$EVAL_SPLIT" "$METRICS_CSV" <<'PY'
import csv
import pathlib
import sys
import yaml
from ultralytics import YOLO

cfg_path = pathlib.Path(sys.argv[1])
model_name = sys.argv[2]
split = sys.argv[3]
metrics_csv = pathlib.Path(sys.argv[4])

with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

project = pathlib.Path(str(cfg["project"]))
run_name = str(cfg["name"])
best_weights = project / run_name / "weights" / "best.pt"

if not best_weights.exists():
    raise FileNotFoundError(f"best.pt not found at: {best_weights}")

model = YOLO(str(best_weights))
metrics = model.val(data=cfg["yaml_path"], split=split, plots=False, verbose=False)

precision = 0.0
recall = 0.0
map50 = 0.0
map50_95 = 0.0

if hasattr(metrics, "box") and metrics.box is not None:
    precision = float(getattr(metrics.box, "mp", 0.0))
    recall = float(getattr(metrics.box, "mr", 0.0))
    map50 = float(getattr(metrics.box, "map50", 0.0))
    map50_95 = float(getattr(metrics.box, "map", 0.0))
elif hasattr(metrics, "seg") and metrics.seg is not None:
    precision = float(getattr(metrics.seg, "mp", 0.0))
    recall = float(getattr(metrics.seg, "mr", 0.0))
    map50 = float(getattr(metrics.seg, "map50", 0.0))
    map50_95 = float(getattr(metrics.seg, "map", 0.0))
elif hasattr(metrics, "results_dict") and metrics.results_dict is not None:
    d = metrics.results_dict
    precision = float(d.get("metrics/precision(B)", d.get("metrics/precision(M)", 0.0)))
    recall = float(d.get("metrics/recall(B)", d.get("metrics/recall(M)", 0.0)))
    map50 = float(d.get("metrics/mAP50(B)", d.get("metrics/mAP50(M)", 0.0)))
    map50_95 = float(d.get("metrics/mAP50-95(B)", d.get("metrics/mAP50-95(M)", 0.0)))

row = [
  model_name,
    run_name,
    str(cfg.get("model_path", "")),
    str(best_weights),
    f"{precision:.6f}",
    f"{recall:.6f}",
    f"{map50:.6f}",
    f"{map50_95:.6f}",
]

with metrics_csv.open("a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(row)

print(f"Wrote metrics row for {model_name}")
PY

  echo "[Model ${model_name}] Completed." | tee -a "$LOG_FILE"
  MODEL_INDEX=$((model_idx + 1))
done

echo "==================================================" | tee -a "$LOG_FILE"
echo "All models completed: ${MODELS}" | tee -a "$LOG_FILE"
echo "Metrics CSV written to: ${METRICS_CSV}" | tee -a "$LOG_FILE"

echo "Generating plots from metrics CSV..." | tee -a "$LOG_FILE"
python -u "${REPO_ROOT}/analysis/plot_model_scale_results.py" \
  --csv "$METRICS_CSV" \
  --out-dir "$PLOTS_DIR" 2>&1 | tee -a "$LOG_FILE"

echo "Plots written to: ${PLOTS_DIR}" | tee -a "$LOG_FILE"
echo "Summary log: ${LOG_FILE}" | tee -a "$LOG_FILE"

conda deactivate
echo "scale comparison job done"