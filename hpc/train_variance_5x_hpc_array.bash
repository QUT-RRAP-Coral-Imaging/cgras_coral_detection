#!/bin/bash -l

#PBS -N train_variance_5x_array
#PBS -l select=1:ncpus=8:ngpus=1:mem=64GB:gpu_id=A100
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -J 1-5
#PBS -j oe

set -euo pipefail

cd "$PBS_O_WORKDIR"
pwd

REPO_ROOT="${REPO_ROOT:-Corals/cgras_settler_counter}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/segmenter/config/cgras/amag140.yaml}"
TRAIN_TEMPLATE="${TRAIN_TEMPLATE:-${REPO_ROOT}/segmenter/config/cgras/amag_train.yaml}"
RUNS="${RUNS:-5}"
RUN_INDEX="${PBS_ARRAY_INDEX:-${RUN_INDEX:-1}}"
BASE_SEED="${BASE_SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/analysis/cgras/variance_runs}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_GROUP="cgras_variance_${TIMESTAMP}"
RUN_TAG="${RUN_GROUP}_run${RUN_INDEX}"
TMP_CONFIG="${OUTPUT_DIR}/${RUN_TAG}.yaml"
RUN_SEED=$((BASE_SEED + RUN_INDEX - 1))
LOG_FILE="${OUTPUT_DIR}/${RUN_TAG}.log"
DRY_RUN="${DRY_RUN:-0}"

if ! [[ "$RUN_INDEX" =~ ^[0-9]+$ ]]; then
  echo "RUN_INDEX must be an integer; got: $RUN_INDEX"
  exit 1
fi

if [ "$RUN_INDEX" -lt 1 ] || [ "$RUN_INDEX" -gt "$RUNS" ]; then
  echo "RUN_INDEX ${RUN_INDEX} is outside the allowed range [1, ${RUNS}]"
  exit 1
fi

if [ -f /home/wardlewo/miniforge3/bin/activate ]; then
  source /home/wardlewo/miniforge3/bin/activate cgras
elif [ -f /home/wardlewo/mambaforge/bin/activate ]; then
  source /home/wardlewo/mambaforge/bin/activate cgras
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate cgras
else
  echo "Could not find conda activate script or conda command; please activate the 'cgras' environment manually"
  exit 1
fi

if [ ! -f "$BASE_CONFIG" ]; then
  echo "Missing base config: $BASE_CONFIG"
  exit 1
fi

if [ ! -f "$TRAIN_TEMPLATE" ]; then
  echo "Missing training template: $TRAIN_TEMPLATE"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
: > "$LOG_FILE"
echo "Run group: ${RUN_GROUP}" | tee "$LOG_FILE"
echo "Array index: ${RUN_INDEX}/${RUNS}" | tee -a "$LOG_FILE"
echo "Base config: ${BASE_CONFIG}" | tee -a "$LOG_FILE"
echo "Train template: ${TRAIN_TEMPLATE}" | tee -a "$LOG_FILE"
echo "Output dir: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"
echo "Seed: ${RUN_SEED}" | tee -a "$LOG_FILE"

which python | tee -a "$LOG_FILE"
nvidia-smi | tee -a "$LOG_FILE" || true

python - "$BASE_CONFIG" "$TRAIN_TEMPLATE" "$TMP_CONFIG" "$RUN_TAG" "$RUN_SEED" <<'PY'
import pathlib
import sys
import yaml

base_cfg = pathlib.Path(sys.argv[1])
train_template = pathlib.Path(sys.argv[2])
out_cfg = pathlib.Path(sys.argv[3])
run_tag = sys.argv[4]
run_seed = int(sys.argv[5])

with base_cfg.open("r", encoding="utf-8") as f:
    base_data = yaml.safe_load(f)

with train_template.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if not isinstance(cfg, dict):
    raise SystemExit(f"Invalid training template structure in {train_template}")

if not isinstance(base_data, dict):
    raise SystemExit(f"Invalid base config structure in {base_cfg}")

cfg.update(base_data)
cfg["name"] = run_tag
cfg["deterministic"] = False
cfg["seed"] = run_seed

if "model_path" not in cfg or not str(cfg["model_path"]).strip():
    raise SystemExit(f"model_path missing after merging template {train_template}")

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run enabled; skipping training." | tee -a "$LOG_FILE"
  echo "Generated config: ${TMP_CONFIG}" | tee -a "$LOG_FILE"
else
  echo "Starting training for ${RUN_TAG}" | tee -a "$LOG_FILE"
  python -u - "$TMP_CONFIG" <<'PY' 2>&1 | tee "${OUTPUT_DIR}/${RUN_TAG}.train.log"
import os
import sys
import yaml
from ultralytics import YOLO

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model_path = os.path.expanduser(cfg["model_path"])
seed = int(cfg.get("seed", 0))
deterministic = bool(cfg.get("deterministic", False))

print(f"Using seed={seed}, deterministic={deterministic}, config={cfg_path}")

model = YOLO(model_path)
model.info()

model.train(
    data=cfg["yaml_path"],
    task=cfg["task"],
    device=cfg["device"],
    epochs=cfg["epochs"],
    batch=cfg["batch_size"],
    project=cfg["project"],
    name=cfg["name"],
    classes=cfg["classes"],
    workers=cfg["workers"],
    patience=cfg["patience"],
    pretrained=cfg["pretrained"],
    save_period=cfg["save_period"],
    seed=seed,
    deterministic=deterministic,
    overlap_mask=cfg["mask_overlap"],
    imgsz=cfg["image_size"],
    scale=cfg["scale"],
    flipud=cfg["flipud"],
    fliplr=cfg["fliplr"],
)
PY
fi

if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
  conda deactivate
fi

echo "Variance job completed for ${RUN_TAG}" | tee -a "$LOG_FILE"
