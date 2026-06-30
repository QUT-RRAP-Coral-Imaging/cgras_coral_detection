#!/usr/bin/env python3
"""
report.py

Builds and saves the summary statistics table across all comparison pairs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_DISPLAY_COLUMNS = [
    ("tile_id",          "Tile"),
    ("tile_group",       "Group"),
    ("species",          "Species"),
    ("season",           "Season"),
    ("date",             "Date"),
    ("count_type",       "Count type"),
    ("total_manual",     "Manual total"),
    ("total_ai",         "AI total"),
    ("pct_diff",         "Diff (%)"),
    ("ba_bias",          "Bias"),
    ("ba_sd",            "SD diff"),
    ("ba_loa_lower",     "LoA lower"),
    ("ba_loa_upper",     "LoA upper"),
    ("ba_prop_bias_r",   "Prop. bias r"),
    ("ba_prop_bias_p",   "Prop. bias p"),
    ("pearson_r",        "Pearson r"),
    ("pearson_p",        "Pearson p"),
    ("spearman_r",       "Spearman r"),
    ("lins_ccc",         "Lin's CCC"),
    ("mae",              "MAE"),
    ("agree_exact",      "Exact agree"),
    ("agree_within_1",   "±1 agree"),
    ("agree_within_2",   "±2 agree"),
    ("rel_error_mean",   "Rel. error mean"),
    ("rel_error_std",    "Rel. error SD"),
    ("wilcoxon_p",       "Wilcoxon p"),
]


def build_summary_table(results: list[dict]) -> pd.DataFrame:
    """
    Assemble a summary DataFrame from a list of result dicts.

    Each dict must contain keys from _DISPLAY_COLUMNS plus the raw metric keys
    from metrics.compute_all_metrics().
    """
    rows = []
    for r in results:
        row = {}
        for key, _ in _DISPLAY_COLUMNS:
            val = r.get(key, np.nan)
            row[key] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.rename(columns={k: v for k, v in _DISPLAY_COLUMNS})

    # Format percentage and agreement columns for readability
    for col in ["Exact agree", "±1 agree", "±2 agree"]:
        if col in df.columns:
            df[col] = (df[col] * 100).round(1).astype(str) + "%"

    for col in ["Diff (%)", "Rel. error mean", "Rel. error SD"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    for col in ["Pearson r", "Spearman r", "Lin's CCC", "Bias",
                "SD diff", "LoA lower", "LoA upper", "MAE",
                "Prop. bias r"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(3)

    for col in ["Pearson p", "Spearman p", "Wilcoxon p", "Prop. bias p"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    return df


def save_summary_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Summary table saved → {path}")


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a compact console version of the summary table."""
    brief_cols = [
        "Tile", "Group", "Date", "Count type",
        "Manual total", "AI total", "Diff (%)",
        "Bias", "LoA lower", "LoA upper",
        "Pearson r", "Lin's CCC",
        "Exact agree", "±1 agree",
        "Wilcoxon p",
    ]
    cols = [c for c in brief_cols if c in df.columns]
    print("\n" + "=" * 120)
    print("COMPARISON SUMMARY")
    print("=" * 120)
    print(df[cols].to_string(index=False))
    print("=" * 120 + "\n")
