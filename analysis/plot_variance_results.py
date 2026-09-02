#!/usr/bin/env python3
"""
Plot variance across repeated training runs from YOLO-style results.csv files.

Examples:
  python analysis/plot_variance_results.py \
      --base-dir /home/java/hpc-home/cslics_detection \
      --group variance_20260331_122902 \
      --output-dir /home/java/Desktop/temp
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_DIR_RE = re.compile(r"^(?P<group>.+)_run(?P<run>\d+)$")
SEED_FOLD_RE = re.compile(r"^(?P<group>.+?)_seed_(?P<seed>\d+)_fold[_-]?(?P<fold>\d+)$")

DEFAULT_METRICS = {
    "precision": ["metrics/precision(B)", "metrics/precision(M)", "metrics/precision"],
    "recall": ["metrics/recall(B)", "metrics/recall(M)", "metrics/recall"],
    "map50": ["metrics/mAP50(B)", "metrics/mAP50(M)", "metrics/mAP50"],
    "map50_95": ["metrics/mAP50-95(B)", "metrics/mAP50-95(M)", "metrics/mAP50-95"],
}


def parse_run_dir(dir_name: str) -> tuple[str, int] | None:
    match = RUN_DIR_RE.match(dir_name)
    if match:
        return match.group("group"), int(match.group("run"))

    match = SEED_FOLD_RE.match(dir_name)
    if match:
        group = match.group("group")
        fold = int(match.group("fold"))
        return group, fold

    return None


def group_matches(filter_group: str | None, group: str) -> bool:
    if filter_group is None:
        return True
    return group == filter_group or group.startswith(filter_group)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot variance of repeated training runs from results.csv files")
    parser.add_argument("--base-dir", required=True, help="Directory containing variance_*_run*/results.csv")
    parser.add_argument("--group", default=None, help="Run group prefix, e.g. variance_20260331_122902")
    parser.add_argument("--output-dir", default=None, help="Directory to save plots")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["precision", "recall", "map50", "map50_95"],
        help="Metric keys to plot (precision recall map50 map50_95)",
    )
    return parser.parse_args()


def discover_results(base_dir: Path, group: str | None) -> list[tuple[int, Path, str]]:
    results = []

    if base_dir.is_file() and base_dir.name == "results.csv":
        run_dir = base_dir.parent
        parsed = parse_run_dir(run_dir.name)
        if not parsed:
            raise ValueError(f"Parent folder '{run_dir.name}' does not match expected run/fold pattern")
        grp, run = parsed
        if not group_matches(group, grp):
            return []
        return [(run, base_dir, grp)]

    if base_dir.is_dir() and (base_dir / "results.csv").exists():
        parsed = parse_run_dir(base_dir.name)
        if parsed:
            inferred_group, _ = parsed
            group = group or inferred_group
            search_root = base_dir.parent
        else:
            search_root = base_dir
    else:
        search_root = base_dir

    for entry in sorted(search_root.iterdir()):
        if not entry.is_dir():
            continue
        parsed = parse_run_dir(entry.name)
        if not parsed:
            continue
        grp, run = parsed
        if not group_matches(group, grp):
            continue
        results_csv = entry / "results.csv"
        if results_csv.exists():
            results.append((run, results_csv, grp))

    results.sort(key=lambda x: x[0])
    return results


def read_results_csv(csv_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except PermissionError as exc:
        raise PermissionError(f"Cannot read file (permission denied): {csv_path}") from exc

    if not rows:
        raise ValueError(f"No data rows in {csv_path}")

    epochs = []
    if "epoch" in fieldnames:
        for idx, row in enumerate(rows):
            value = row.get("epoch", "").strip()
            if value == "":
                epochs.append(float(idx))
            else:
                epochs.append(float(value))
    else:
        epochs = [float(i) for i in range(len(rows))]

    data: dict[str, np.ndarray] = {}
    for name in fieldnames:
        values = []
        has_numeric = False
        for row in rows:
            raw = (row.get(name) or "").strip()
            if raw == "":
                values.append(np.nan)
                continue
            try:
                parsed = float(raw)
                values.append(parsed)
                has_numeric = True
            except ValueError:
                values.append(np.nan)
        if has_numeric:
            data[name] = np.array(values, dtype=float)

    return np.array(epochs, dtype=float), data, fieldnames


def resolve_metric_name(metric_key: str, available_columns: set[str]) -> str | None:
    for candidate in DEFAULT_METRICS.get(metric_key, []):
        if candidate in available_columns:
            return candidate
    return None


def align_runs(
    run_data: list[tuple[int, np.ndarray, dict[str, np.ndarray]]],
    metric_column: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    run_ids = [run_id for run_id, _, _ in run_data]
    
    # Find the maximum epoch range across all runs
    all_epochs = []
    for _, epochs, _ in run_data:
        all_epochs.extend(epochs.tolist())
    
    if not all_epochs:
        raise ValueError(f"No epochs found for metric column '{metric_column}'")
    
    # Create common epoch range from min to max
    min_epoch = min(all_epochs)
    max_epoch = max(all_epochs)
    common_epochs = sorted(set(all_epochs))  # Use all unique epochs across all runs
    
    values_per_run = []
    for _, epochs, metrics in run_data:
        idx_by_epoch = {ep: idx for idx, ep in enumerate(epochs.tolist())}
        metric_values = metrics[metric_column]
        # Pad with NaN for epochs not in this run
        values = [metric_values[idx_by_epoch[ep]] if ep in idx_by_epoch else np.nan for ep in common_epochs]
        values_per_run.append(values)

    return np.array(common_epochs, dtype=float), np.array(values_per_run, dtype=float), run_ids


def plot_metric_variance(
    group: str,
    metric_key: str,
    metric_column: str,
    epochs: np.ndarray,
    run_values: np.ndarray,
    run_ids: list[int],
    output_dir: Path,
) -> Path:
    mean_values = np.nanmean(run_values, axis=0)
    std_values = np.nanstd(run_values, axis=0)

    plt.figure(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(run_ids), 3)))

    plt.fill_between(
        epochs,
        mean_values - std_values,
        mean_values + std_values,
        alpha=0.15,
        color="gray",
        label="±1 std",
        zorder=1,
    )

    for i, run_id in enumerate(run_ids):
        plt.plot(
            epochs,
            run_values[i],
            alpha=0.9,
            linewidth=2.0,
            color=colors[i % len(colors)],
            label=f"run{run_id}",
            zorder=3,
        )

    plt.plot(epochs, mean_values, linewidth=2.2, color="black", linestyle="--", label="mean", zorder=4)

    plt.title(f"{group} - {metric_key} variance across runs")
    plt.xlabel("Epoch")
    plt.ylabel(metric_column)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()

    out_file = output_dir / f"{group}_{metric_key}_variance.png"
    plt.savefig(out_file, dpi=180)
    plt.close()
    return out_file


def plot_final_summary(
    group: str,
    summary: dict[str, list[float]],
    output_dir: Path,
) -> Path:
    metric_names = list(summary.keys())
    means = [float(np.nanmean(summary[m])) for m in metric_names]
    stds = [float(np.nanstd(summary[m])) for m in metric_names]

    x = np.arange(len(metric_names))
    plt.figure(figsize=(9, 5))
    plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7)

    for idx, metric in enumerate(metric_names):
        y = summary[metric]
        jitter = np.linspace(-0.12, 0.12, len(y)) if len(y) > 1 else np.array([0.0])
        plt.scatter(np.full(len(y), x[idx]) + jitter, y, s=25)

    plt.xticks(x, metric_names)
    plt.ylabel("Final epoch metric value")
    plt.title(f"{group} - final epoch variance")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()

    out_file = output_dir / f"{group}_final_metrics_variance.png"
    plt.savefig(out_file, dpi=180)
    plt.close()
    return out_file


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (base_dir / "variance_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_results(base_dir, args.group)
    if not discovered:
        raise SystemExit("No matching results.csv files found. Check --base-dir and --group.")

    groups = defaultdict(list)
    for run_id, csv_path, group in discovered:
        groups[group].append((run_id, csv_path))

    all_outputs: list[Path] = []

    for group, entries in sorted(groups.items()):
        run_data = []
        shared_columns: set[str] | None = None

        print(f"\nGroup: {group}")
        print(f"Runs found: {[run_id for run_id, _ in entries]}")

        for run_id, csv_path in entries:
            try:
                epochs, data, fieldnames = read_results_csv(csv_path)
            except PermissionError as exc:
                print(f"Skipping run{run_id}: {exc}")
                continue
            run_data.append((run_id, epochs, data))
            cols = set(fieldnames)
            shared_columns = cols if shared_columns is None else (shared_columns & cols)

        if len(run_data) < 2:
            print("Need at least 2 readable runs to compute variance; skipping this group.")
            continue
         

        assert shared_columns is not None

        final_summary: dict[str, list[float]] = {}
    
        for metric_key in args.metrics:
            metric_column = resolve_metric_name(metric_key, shared_columns)
            if metric_column is None:
                print(f"Skipping metric '{metric_key}' (column not found in all runs)")
                continue

            epochs, run_values, run_ids = align_runs(run_data, metric_column)
            out = plot_metric_variance(
                group=group,
                metric_key=metric_key,
                metric_column=metric_column,
                epochs=epochs,
                run_values=run_values,
                run_ids=run_ids,
                output_dir=output_dir,
            )
            all_outputs.append(out)
            print(f"Saved: {out}")

            final_vals = [float(v[-1]) for v in run_values]
            final_summary[metric_key] = final_vals

        if final_summary:
            out = plot_final_summary(group=group, summary=final_summary, output_dir=output_dir)
            all_outputs.append(out)
            print(f"Saved: {out}")
        else:
            print("No requested metrics were available; final summary not generated.")
    # print for each run the final metric values in a table

    print("\nDone.")
    print("Generated plot files:")
    for out in all_outputs:
        print(f"  - {out}")


if __name__ == "__main__":
    main()
