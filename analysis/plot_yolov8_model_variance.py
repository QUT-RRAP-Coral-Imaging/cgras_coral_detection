#!/usr/bin/env python3
"""
Plot variance across YOLOv8 model scales from per-run results.csv files.

Expected run folder naming from hpc/train_yolov8_scales_hpc.bash:
  yolov8_scales_<timestamp>_yolov8n
  yolov8_scales_<timestamp>_yolov8s
  yolov8_scales_<timestamp>_yolov8m
  yolov8_scales_<timestamp>_yolov8l
  yolov8_scales_<timestamp>_yolov8x

Examples:
  python analysis/plot_yolov8_model_variance.py \
    --base-dir /home/java/hpc-home/cslics_detection \
    --group yolov8_scales_20260407_101500 \
    --output-dir /home/java/Desktop/temp
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
MODEL_INDEX = {name: idx for idx, name in enumerate(MODEL_ORDER)}

METRIC_CANDIDATES = {
    "precision": ["metrics/precision(B)", "metrics/precision(M)", "metrics/precision"],
    "recall": ["metrics/recall(B)", "metrics/recall(M)", "metrics/recall"],
    "map50": ["metrics/mAP50(B)", "metrics/mAP50(M)", "metrics/mAP50"],
    "map50_95": ["metrics/mAP50-95(B)", "metrics/mAP50-95(M)", "metrics/mAP50-95"],
}

RUN_RE = re.compile(r"^(?P<group>yolov8_scales_\d{8}_\d{6})_(?P<model>yolov8[nslmx])$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot variance across YOLOv8 model scales from results.csv files")
    parser.add_argument("--base-dir", required=True, help="Directory containing yolov8_scales_*_yolov8*/results.csv")
    parser.add_argument("--group", required=True, help="Group prefix, e.g. yolov8_scales_20260407_101500")
    parser.add_argument("--output-dir", default=None, help="Directory for output plots (default: <base-dir>/<group>_model_variance)")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["precision", "recall", "map50", "map50_95"],
        help="Metrics to plot (precision recall map50 map50_95)",
    )
    return parser.parse_args()


def discover_model_results(base_dir: Path, group: str) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []

    for run_dir in sorted(base_dir.glob(f"{group}_yolov8*")):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        model = match.group("model")
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            items.append((model, results_csv))

    items.sort(key=lambda item: MODEL_INDEX.get(item[0], 999))
    return items


def read_results_csv(csv_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], set[str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    if "epoch" in fieldnames:
        epochs = np.array([float((row.get("epoch") or idx)) for idx, row in enumerate(rows)], dtype=float)
    else:
        epochs = np.array([float(i) for i in range(len(rows))], dtype=float)

    data: dict[str, np.ndarray] = {}
    for col in fieldnames:
        parsed = []
        has_numeric = False
        for row in rows:
            raw = (row.get(col) or "").strip()
            if raw == "":
                parsed.append(np.nan)
                continue
            try:
                parsed.append(float(raw))
                has_numeric = True
            except ValueError:
                parsed.append(np.nan)
        if has_numeric:
            data[col] = np.array(parsed, dtype=float)

    return epochs, data, set(fieldnames)


def resolve_metric(metric_key: str, shared_columns: set[str]) -> str | None:
    for candidate in METRIC_CANDIDATES.get(metric_key, []):
        if candidate in shared_columns:
            return candidate
    return None


def align_epochs(
    model_data: list[tuple[str, np.ndarray, dict[str, np.ndarray]]],
    metric_column: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    common_epochs = sorted(set.intersection(*[set(epochs.tolist()) for _, epochs, _ in model_data]))
    if not common_epochs:
        raise ValueError(f"No common epochs found for {metric_column}")

    labels: list[str] = []
    values_by_model: list[list[float]] = []
    for model, epochs, metrics in model_data:
        idx = {ep: i for i, ep in enumerate(epochs.tolist())}
        values = [float(metrics[metric_column][idx[ep]]) for ep in common_epochs]
        labels.append(model)
        values_by_model.append(values)

    return np.array(common_epochs, dtype=float), np.array(values_by_model, dtype=float), labels


def plot_metric_variance(
    group: str,
    metric_key: str,
    metric_column: str,
    epochs: np.ndarray,
    values_by_model: np.ndarray,
    labels: list[str],
    output_dir: Path,
) -> Path:
    mean_values = np.nanmean(values_by_model, axis=0)
    std_values = np.nanstd(values_by_model, axis=0)

    plt.figure(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 3)))

    plt.fill_between(
        epochs,
        mean_values - std_values,
        mean_values + std_values,
        color="gray",
        alpha=0.18,
        label="±1 std across models",
        zorder=1,
    )

    for i, label in enumerate(labels):
        plt.plot(
            epochs,
            values_by_model[i],
            label=label,
            color=colors[i % len(colors)],
            linewidth=2.0,
            alpha=0.95,
            zorder=3,
        )

    plt.plot(epochs, mean_values, color="black", linestyle="--", linewidth=2.2, label="mean", zorder=4)
    plt.title(f"{group} - {metric_key} variance across YOLOv8 scales")
    plt.xlabel("Epoch")
    plt.ylabel(metric_column)
    plt.grid(alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()

    out = output_dir / f"{group}_{metric_key}_model_variance.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_final_metric_bars(
    group: str,
    metric_key: str,
    labels: list[str],
    final_values: list[float],
    output_dir: Path,
) -> Path:
    x = np.arange(len(labels))
    plt.figure(figsize=(9, 5))
    bars = plt.bar(x, final_values, alpha=0.85)

    for bar, value in zip(bars, final_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    spread_std = float(np.nanstd(final_values))
    spread_range = float(np.nanmax(final_values) - np.nanmin(final_values))

    plt.xticks(x, labels)
    plt.ylim(0, max(1.0, max(final_values) * 1.12 if final_values else 1.0))
    plt.ylabel("Final epoch value")
    plt.title(f"{group} - final {metric_key} by model\nstd={spread_std:.4f}, range={spread_range:.4f}")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    out = output_dir / f"{group}_{metric_key}_final_by_model.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (base_dir / f"{args.group}_model_variance")
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_model_results(base_dir, args.group)
    if len(discovered) < 2:
        raise SystemExit("Need at least 2 model results.csv files to measure model variance.")

    print(f"Group: {args.group}")
    print(f"Models found: {[model for model, _ in discovered]}")

    model_data: list[tuple[str, np.ndarray, dict[str, np.ndarray]]] = []
    shared_columns: set[str] | None = None
    for model, csv_path in discovered:
        epochs, data, columns = read_results_csv(csv_path)
        model_data.append((model, epochs, data))
        shared_columns = columns if shared_columns is None else (shared_columns & columns)

    assert shared_columns is not None

    outputs: list[Path] = []
    for metric_key in args.metrics:
        metric_column = resolve_metric(metric_key, shared_columns)
        if metric_column is None:
            print(f"Skipping metric '{metric_key}' (not present in all model CSVs)")
            continue

        epochs, values_by_model, labels = align_epochs(model_data, metric_column)
        out = plot_metric_variance(
            group=args.group,
            metric_key=metric_key,
            metric_column=metric_column,
            epochs=epochs,
            values_by_model=values_by_model,
            labels=labels,
            output_dir=output_dir,
        )
        outputs.append(out)
        print(f"Saved: {out}")

        final_values = [float(series[-1]) for series in values_by_model]
        out_final = plot_final_metric_bars(args.group, metric_key, labels, final_values, output_dir)
        outputs.append(out_final)
        print(f"Saved: {out_final}")

    print("Done.")
    for out in outputs:
        print(f"  - {out}")


if __name__ == "__main__":
    main()
