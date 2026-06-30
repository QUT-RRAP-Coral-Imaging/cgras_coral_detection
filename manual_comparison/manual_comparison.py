#!/usr/bin/env python3
"""
manual_comparison.py

Compares manual coral counts against CGRAS AI counts across all matched
tile-timepoint pairs, producing per-pair figures and an aggregate report.

Usage:
    python manual_comparison.py \
        --manual  /path/to/202505_...wholetiles_split.xlsx \
        --ai_dir  /path/to/ai_counts/ \
        --output  /path/to/output/

The script handles partial overlap between manual and AI datasets: only
pairs where both a manual sheet and a matching AI sheet exist are analysed.
Unmatched manual sheets are reported but not treated as errors.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe for servers and HPC
import matplotlib.pyplot as plt
import numpy as np

from .data_loader import find_matched_pairs, load_matrices
from .metrics import compute_all_metrics
from .visualisation import (
    plot_difference_heatmap,
    plot_scatter,
    plot_bland_altman,
    plot_histogram,
    plot_combined_histogram,
    plot_combined_histogram_by_species,
    plot_combined_scatter,
    plot_combined_bland_altman,
    plot_spatial_average_error,
    plot_time_histories,
    plot_time_history_single,
    plot_growth_trend,
)
from .report import build_summary_table, save_summary_table, print_summary_table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare manual coral counts with CGRAS AI counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manual", required=True, type=Path,
        help="Path to the manual count Excel file (wholetiles_split.xlsx).",
    )
    parser.add_argument(
        "--ai_dir", required=True, type=Path,
        help="Directory containing the AI count *_data.xlsx files.",
    )
    parser.add_argument(
        "--output", default=Path("output"), type=Path,
        help="Root directory for all output figures and tables.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively (in addition to saving).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-pair analysis
# ---------------------------------------------------------------------------

def _analyse_pair(pair, manual_file: Path, output_dir: Path) -> list[dict]:
    """
    Run the full comparison for one ComparisonPair.
    Returns a list of result dicts (one for alive, optionally one for dead).
    """
    results = []
    pair_dir = output_dir / "per_pair" / pair.label

    for count_type in ("alive", "dead"):
        m_arr = pair.manual_alive if count_type == "alive" else pair.manual_dead
        a_arr = pair.ai_alive     if count_type == "alive" else pair.ai_dead
        if m_arr is None or a_arr is None:
            continue

        # Metrics
        mets = compute_all_metrics(m_arr, a_arr)

        # Per-pair figures
        title_base = f"{pair.label} [{count_type}]"

        plot_difference_heatmap(
            m_arr, a_arr, title=f"Difference heatmap — {title_base}",
            output_path=pair_dir / f"{count_type}_heatmap.png",
        )
        plot_scatter(
            m_arr, a_arr, title=f"Scatter — {title_base}",
            output_path=pair_dir / f"{count_type}_scatter.png",
        )
        plot_bland_altman(
            m_arr, a_arr, title=f"Bland-Altman — {title_base}",
            output_path=pair_dir / f"{count_type}_bland_altman.png",
        )
        plot_histogram(
            m_arr, a_arr, title=f"Difference histogram — {title_base}",
            output_path=pair_dir / f"{count_type}_histogram.png",
        )

        print(f"  [{count_type}]  manual={mets['total_manual']:.0f}  "
              f"AI={mets['total_ai']:.0f}  "
              f"bias={mets['ba_bias']:+.2f}  "
              f"r={mets['pearson_r']:.3f}  "
              f"CCC={mets['lins_ccc']:.3f}  "
              f"±1={100*mets['agree_within_1']:.1f}%  "
              f"Wilcoxon p={mets['wilcoxon_p']:.4f}")

        results.append({
            "tile_id":    pair.tile_id,
            "tile_group": pair.tile_group,
            "species":    pair.species,
            "season":     pair.season,
            "date":       pair.date.strftime("%Y-%m-%d"),
            "count_type": count_type,
            **mets,
        })

    return results


# ---------------------------------------------------------------------------
# Aggregate figures
# ---------------------------------------------------------------------------

def _aggregate_figures(pairs, output_dir: Path) -> None:
    agg_dir = output_dir / "aggregate"

    for count_type in ("alive", "dead"):
        active = [p for p in pairs
                  if (p.manual_alive if count_type == "alive" else p.manual_dead) is not None
                  and (p.ai_alive     if count_type == "alive" else p.ai_dead)     is not None]
        if not active:
            continue

        plot_combined_histogram(
            active, count_type=count_type,
            output_path=agg_dir / f"combined_histogram_{count_type}.png",
        )
        if count_type == "alive":
            plot_combined_histogram_by_species(
                active, count_type=count_type,
                output_path=agg_dir / f"combined_histogram_{count_type}_by_species.png",
            )
        plot_combined_scatter(
            active, count_type=count_type,
            output_path=agg_dir / f"combined_scatter_{count_type}.png",
        )
        plot_combined_bland_altman(
            active, count_type=count_type,
            output_path=agg_dir / f"combined_bland_altman_{count_type}.png",
        )

        # Spatial average error: all pairs, then by group
        plot_spatial_average_error(
            active, count_type=count_type,
            output_path=agg_dir / f"spatial_avg_error_{count_type}_all.png",
        )
        for group in ("T05", "Tile ID"):
            fig = plot_spatial_average_error(
                active, count_type=count_type, group_filter=group,
                output_path=agg_dir / f"spatial_avg_error_{count_type}_{group.replace(' ', '_')}.png",
            )
            if fig is None:
                # No pairs for this group; output file not written (expected)
                pass

    plot_time_histories(pairs, output_path=agg_dir / "time_histories.png")

    # Per-tile time history: individual PNG + JSON
    by_tile: dict[str, list] = defaultdict(list)
    for p in pairs:
        if p.manual_alive is not None and p.ai_alive is not None:
            by_tile[p.tile_id].append(p)

    for tile_id, tile_pairs in sorted(by_tile.items()):
        tile_pairs = sorted(tile_pairs, key=lambda p: p.date)
        safe_id = tile_id.replace("/", "_").replace(" ", "_")

        plot_time_history_single(
            tile_pairs,
            output_path=agg_dir / f"time_histories_{safe_id}.png",
        )

        t0 = tile_pairs[0].date
        n_cells = int(tile_pairs[0].manual_alive.size)
        first = tile_pairs[0]
        json_data = {
            "tile_id":    tile_id,
            "tile_label": first.ai_file.stem.removesuffix("_data"),
            "species":    first.species,
            "season":     first.season,
            "n_cells":    n_cells,
            "timepoints": [
                {
                    "date":             p.date.strftime("%Y-%m-%d"),
                    "days_since_first": int((p.date - t0).days),
                    "manual_total":     float(p.manual_alive.sum()),
                    "ai_total":         float(p.ai_alive.sum()),
                    "manual_err":       float(p.manual_alive.std() * np.sqrt(n_cells)),
                    "ai_err":           float(p.ai_alive.std()     * np.sqrt(n_cells)),
                }
                for p in tile_pairs
            ],
        }
        json_path = agg_dir / f"time_histories_{safe_id}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as fh:
            json.dump(json_data, fh, indent=2)
        print(f"  Saved time history: {json_path.name}")

    plot_growth_trend(pairs, output_path=agg_dir / "growth_trend.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if not args.manual.exists():
        print(f"ERROR: manual file not found: {args.manual}", file=sys.stderr)
        sys.exit(1)
    if not args.ai_dir.is_dir():
        print(f"ERROR: AI directory not found: {args.ai_dir}", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    # ---- Match pairs ----
    print("Matching manual sheets to AI count files...")
    pairs, unmatched = find_matched_pairs(args.manual, args.ai_dir)

    if unmatched:
        print(f"\n{len(unmatched)} manual sheet(s) have no AI match:")
        for msg in unmatched:
            print(f"  - {msg}")

    if not pairs:
        print("No matched pairs found. Nothing to analyse.")
        sys.exit(0)

    print(f"\nAnalysing {len(pairs)} matched pair(s)...\n")

    # ---- Load matrices and run per-pair analysis ----
    all_results = []
    for pair in pairs:
        print(f"  {pair.label}  ({pair.manual_sheet})")
        load_matrices(pair, args.manual)
        results = _analyse_pair(pair, args.manual, args.output)
        all_results.extend(results)

    # ---- Aggregate figures ----
    print("\nGenerating aggregate figures...")
    _aggregate_figures(pairs, args.output)

    # ---- Summary table ----
    df = build_summary_table(all_results)
    save_summary_table(df, args.output / "summary_table.csv")
    print_summary_table(df)

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()

    print(f"\nAll outputs saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
