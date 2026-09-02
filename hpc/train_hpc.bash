    #!/bin/bash -l

    #PBS -N train_model
    #PBS -l select=1:ncpus=8:ngpus=1:mem=64GB:gpu_id=A100
    #PBS -l walltime=24:00:00
    #PBS -m abe
    #PBS -j oe

    set -euo pipefail

    cd "$PBS_O_WORKDIR"
    pwd

    # You can override these with qsub -v KEY=value,...
    REPO_ROOT="${REPO_ROOT:-Corals/cgras_settler_counter}"
    TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_ROOT}/segmenter/scripts/train.py}"
    DEFAULT_CONFIG="${DEFAULT_CONFIG:-${REPO_ROOT}/segmenter/config/cslics/cslics_2025.yaml}"
    TRAIN_CONFIG="${1:-${TRAIN_CONFIG:-$DEFAULT_CONFIG}}"
    CONDA_ENV="${CONDA_ENV:-cgras}"
    LOG_DIR="${LOG_DIR:-${REPO_ROOT}/analysis/hpc_logs}"
    RUN_TAG="${RUN_TAG:-train_$(date +%Y%m%d_%H%M%S)}"
    LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"
    RESOLVED_CONFIG="${LOG_DIR}/${RUN_TAG}.resolved_config.yaml"
    RESOLVED_DATASET_YAML="${LOG_DIR}/${RUN_TAG}.resolved_dataset.yaml"

    if [ -f "${HOME}/miniforge3/bin/activate" ]; then
        source "${HOME}/miniforge3/bin/activate" "$CONDA_ENV"
    elif [ -f "${HOME}/mambaforge/bin/activate" ]; then
        source "${HOME}/mambaforge/bin/activate" "$CONDA_ENV"
    elif [ -f /home/wardlewo/miniforge3/bin/activate ]; then
        source /home/wardlewo/miniforge3/bin/activate "$CONDA_ENV"
    elif [ -f /home/wardlewo/mambaforge/bin/activate ]; then
        source /home/wardlewo/mambaforge/bin/activate "$CONDA_ENV"
    else
        echo "Could not find conda activate script"
        exit 1
    fi

    if [ ! -f "$TRAIN_SCRIPT" ]; then
        echo "Missing training script: $TRAIN_SCRIPT"
        exit 1
    fi

    if [ ! -f "$TRAIN_CONFIG" ]; then
        echo "Missing train config: $TRAIN_CONFIG"
        exit 1
    fi

    mkdir -p "$LOG_DIR"

    # Normalize common path issues (for example missing leading slash on mnt paths)
    python - "$TRAIN_CONFIG" "$RESOLVED_CONFIG" "$PBS_O_WORKDIR" "$RESOLVED_DATASET_YAML" <<'PY'
import pathlib
import sys
import yaml

in_cfg = pathlib.Path(sys.argv[1])
out_cfg = pathlib.Path(sys.argv[2])
workdir = pathlib.Path(sys.argv[3]).resolve()
out_dataset_yaml = pathlib.Path(sys.argv[4])

with in_cfg.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if not isinstance(cfg, dict):
    raise SystemExit(f"Invalid config structure in {in_cfg}")

changes = []

def normalize_path_value(value: str) -> str:
    text = value.strip()
    if text.startswith("mnt/"):
        return "/" + text
    return text

for key in ("yaml_path", "project", "model_path"):
    raw = cfg.get(key)
    if isinstance(raw, str):
        fixed = normalize_path_value(raw)
        if fixed != raw:
            cfg[key] = fixed
            changes.append((key, raw, fixed))

yaml_path = str(cfg.get("yaml_path", "")).strip()
if yaml_path:
    looks_like_file_path = yaml_path.startswith("/") or yaml_path.startswith("./") or yaml_path.startswith("../") or "/" in yaml_path
    if looks_like_file_path:
        candidate = pathlib.Path(yaml_path)
        if not candidate.is_absolute():
            candidate = (workdir / candidate).resolve()
            cfg["yaml_path"] = str(candidate)
        if not candidate.exists():
            raise SystemExit(f"Dataset yaml not found: {candidate}")
        if candidate.suffix.lower() in {".yaml", ".yml"}:
            with candidate.open("r", encoding="utf-8") as f:
                dataset_cfg = yaml.safe_load(f)

            if isinstance(dataset_cfg, dict):
                dataset_changed = False

                for key in ("path", "train", "val", "test"):
                    raw = dataset_cfg.get(key)
                    if isinstance(raw, str):
                        fixed = normalize_path_value(raw)
                        if fixed != raw:
                            dataset_cfg[key] = fixed
                            dataset_changed = True
                            print(f"Normalized dataset {key}: {raw} -> {fixed}")
                    elif isinstance(raw, list):
                        fixed_list = []
                        item_changed = False
                        for item in raw:
                            if isinstance(item, str):
                                fixed_item = normalize_path_value(item)
                                if fixed_item != item:
                                    item_changed = True
                                fixed_list.append(fixed_item)
                            else:
                                fixed_list.append(item)
                        if item_changed:
                            dataset_cfg[key] = fixed_list
                            dataset_changed = True
                            print(f"Normalized dataset list field: {key}")

                if dataset_changed:
                    out_dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
                    with out_dataset_yaml.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(dataset_cfg, f, sort_keys=False)
                    cfg["yaml_path"] = str(out_dataset_yaml)
                    print(f"Wrote resolved dataset yaml: {out_dataset_yaml}")

                dataset_root = None
                dataset_root_raw = dataset_cfg.get("path")
                if isinstance(dataset_root_raw, str) and dataset_root_raw.strip():
                    root_candidate = pathlib.Path(dataset_root_raw.strip())
                    if not root_candidate.is_absolute():
                        root_candidate = (candidate.parent / root_candidate).resolve()
                    dataset_root = root_candidate

                for split_key in ("train", "val", "test"):
                    split_raw = dataset_cfg.get(split_key)
                    split_items = split_raw if isinstance(split_raw, list) else [split_raw]
                    for item in split_items:
                        if not isinstance(item, str) or not item.strip():
                            continue
                        split_path = pathlib.Path(item.strip())
                        if not split_path.is_absolute() and dataset_root is not None:
                            split_path = dataset_root / split_path
                        elif not split_path.is_absolute():
                            split_path = (candidate.parent / split_path).resolve()
                        if not split_path.exists():
                            raise SystemExit(
                                f"Dataset split path not found ({split_key}): {split_path}"
                            )

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

for key, before, after in changes:
    print(f"Normalized {key}: {before} -> {after}")
PY

    echo "Run tag: ${RUN_TAG}" | tee "$LOG_FILE"
    echo "Repo root: ${REPO_ROOT}" | tee -a "$LOG_FILE"
    echo "Conda env: ${CONDA_ENV}" | tee -a "$LOG_FILE"
    echo "Training script: ${TRAIN_SCRIPT}" | tee -a "$LOG_FILE"
    echo "Training config: ${TRAIN_CONFIG}" | tee -a "$LOG_FILE"
    echo "Resolved config: ${RESOLVED_CONFIG}" | tee -a "$LOG_FILE"
    echo "Resolved dataset yaml: ${RESOLVED_DATASET_YAML}" | tee -a "$LOG_FILE"
    echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"

    which python | tee -a "$LOG_FILE"
    nvidia-smi | tee -a "$LOG_FILE" || true

    echo "Starting training..." | tee -a "$LOG_FILE"
    python -u "$TRAIN_SCRIPT" --config "$RESOLVED_CONFIG" 2>&1 | tee -a "$LOG_FILE"

    conda deactivate
    echo "Training job completed: ${RUN_TAG}" | tee -a "$LOG_FILE"