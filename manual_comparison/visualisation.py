#!/usr/bin/env python3
"""
visualisation.py

All figure-generation functions for the manual vs AI count comparison.
Each function returns a matplotlib Figure and optionally saves it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Optional[Path], dpi: int = 150) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _lim(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(np.nanmax(a), np.nanmax(b))) * 1.05


# ---------------------------------------------------------------------------
# Per-pair figures
# ---------------------------------------------------------------------------

def plot_difference_heatmap(
    manual: np.ndarray,
    ai: np.ndarray,
    title: str = "",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    20×20 heatmap of (manual − AI) per cell.
    Zero-difference cells are left blank to reduce visual clutter.
    """
    diff = (manual - ai).astype(float)
    annot = diff.copy().astype(object)
    annot[diff == 0] = ""

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(diff, cmap="coolwarm", center=0, annot=annot, fmt="", ax=ax,
                cbar_kws={"label": "Difference (Manual − AI)"})
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_scatter(
    manual: np.ndarray,
    ai: np.ndarray,
    title: str = "",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Cell-level scatter with 1:1 line and OLS regression line.
    Annotates R² and Lin's CCC from metrics module.
    """
    from .metrics import lins_ccc

    m = manual.flatten().astype(float)
    a = ai.flatten().astype(float)

    slope, intercept, r_val, p_val, _ = stats.linregress(m, a)
    ccc = lins_ccc(manual, ai)
    lim = _lim(m, a)
    x_line = np.linspace(0, lim, 200)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(m, a, alpha=0.30, s=18, color="steelblue", zorder=2)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.4, label="1:1", zorder=3)
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=1.4,
            label=f"OLS  y = {slope:.2f}x + {intercept:.2f}", zorder=3)

    ax.set_xlim(left=0, right=lim)
    ax.set_ylim(bottom=0, top=lim)
    ax.set_xlabel("Manual count (per tab)")
    ax.set_ylabel("AI count (per tab)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.text(0.05, 0.95,
            f"R² = {r_val**2:.3f}\nCCC = {ccc:.3f}\nn = {len(m)}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_bland_altman(
    manual: np.ndarray,
    ai: np.ndarray,
    title: str = "",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bland-Altman plot of (manual − AI) vs (manual + AI)/2.
    A regression line is overlaid when proportional bias is significant (p < 0.05).
    """
    m = manual.flatten().astype(float)
    a = ai.flatten().astype(float)
    diff = m - a
    mean_val = (m + a) / 2

    bias = np.mean(diff)
    sd   = np.std(diff, ddof=1)
    loa_u = bias + 1.96 * sd
    loa_l = bias - 1.96 * sd

    try:
        prop_r, prop_p = stats.pearsonr(mean_val, diff)
    except Exception:
        prop_r, prop_p = np.nan, np.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_val, diff, alpha=0.25, s=16, color="steelblue", zorder=2)
    ax.axhline(bias,  color="black", linewidth=1.6, label=f"Bias = {bias:.2f}")
    ax.axhline(loa_u, color="red",   linewidth=1.2, linestyle="--",
               label=f"+1.96 SD = {loa_u:.2f}")
    ax.axhline(loa_l, color="red",   linewidth=1.2, linestyle="--",
               label=f"−1.96 SD = {loa_l:.2f}")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")

    # Overlay proportional-bias regression if significant
    if prop_p < 0.05:
        sl, ic, *_ = stats.linregress(mean_val, diff)
        x_ = np.linspace(mean_val.min(), mean_val.max(), 200)
        ax.plot(x_, sl * x_ + ic, "g-", linewidth=1.2,
                label=f"Prop. bias (r={prop_r:.2f}, p={prop_p:.3f})")

    ax.set_xlabel("Mean of manual and AI counts")
    ax.set_ylabel("Difference (Manual − AI)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_histogram(
    manual: np.ndarray,
    ai: np.ndarray,
    title: str = "",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Histogram of per-cell (manual − AI) differences with mean marked."""
    diff = (manual - ai).astype(float).flatten()
    min_d, max_d = int(np.floor(diff.min())), int(np.ceil(diff.max()))
    bins = np.arange(min_d - 0.5, max_d + 1.5, 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(diff, bins=bins, edgecolor="black", alpha=0.75, color="steelblue")
    ax.axvline(np.mean(diff), color="red", linewidth=1.5, linestyle="--",
               label=f"Mean = {np.mean(diff):.2f}")
    ax.axvline(0, color="black", linewidth=0.9, linestyle=":")
    ax.set_xlabel("Difference (Manual − AI)")
    ax.set_ylabel("Frequency")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)

    stats_txt = (f"SD = {np.std(diff, ddof=1):.2f}\n"
                 f"n = {len(diff)}\n"
                 f"Range [{min_d}, {max_d}]")
    ax.text(0.97, 0.95, stats_txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    plt.tight_layout()
    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Aggregate figures
# ---------------------------------------------------------------------------

def plot_combined_histogram(
    pairs,
    count_type: str = "alive",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Pooled histogram of per-cell (manual − AI) differences across all pairs,
    with each tile's contribution shown as a stacked colour segment.

    Display label is derived from the AI filename stem
    (e.g. "2025Dec-982091078941976"), giving one entry per tile.
    """
    def _tile_label(p) -> str:
        return p.ai_file.stem.removesuffix("_data")

    # Accumulate diffs per tile label, preserving insertion order
    tile_diffs: dict[str, list] = {}
    for p in pairs:
        m_arr = p.manual_alive if count_type == "alive" else p.manual_dead
        a_arr = p.ai_alive     if count_type == "alive" else p.ai_dead
        if m_arr is None or a_arr is None:
            continue
        d = (m_arr.astype(float) - a_arr.astype(float)).flatten()
        tile_diffs.setdefault(_tile_label(p), []).append(d)

    if not tile_diffs:
        return None

    all_diffs = np.concatenate([d for dlist in tile_diffs.values() for d in dlist])
    min_d = int(np.floor(all_diffs.min()))
    max_d = int(np.ceil(all_diffs.max()))
    bins = np.arange(min_d - 0.5, max_d + 1.5, 1)
    mean_d = float(np.mean(all_diffs))
    sd_d   = float(np.std(all_diffs, ddof=1))

    colours = [f"C{i}" for i in range(len(tile_diffs))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(bins) - 1)

    for (tile_label, dlist), colour in zip(tile_diffs.items(), colours):
        gd = np.concatenate(dlist)
        counts, _ = np.histogram(gd, bins=bins)
        n_timepoints = len(dlist)
        ax.bar(bins[:-1] + 0.5, counts, width=1.0, bottom=bottom,
               color=colour, alpha=0.75, edgecolor="black", linewidth=0.5,
               label=f"{tile_label} ({n_timepoints} timepoint{'s' if n_timepoints != 1 else ''})")
        bottom += counts

    ax.axvline(mean_d, color="red", linewidth=1.6, linestyle="--",
               label=f"Mean = {mean_d:.2f}")
    ax.axvline(0, color="black", linewidth=0.9, linestyle=":")
    ax.set_xlabel("Difference (Manual − AI) per tab", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(
        f"Pooled per-tab difference histogram — {count_type} counts\n"
        f"all tile-timepoints (n = {len(all_diffs):,} tabs)",
        fontsize=11,
    )
    ax.legend(fontsize=9)

    stats_txt = (f"Mean = {mean_d:.2f}\n"
                 f"SD = {sd_d:.2f}\n"
                 f"n tabs = {len(all_diffs):,}\n"
                 f"Range [{min_d}, {max_d}]")
    ax.text(0.97, 0.95, stats_txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_combined_histogram_by_species(
    pairs,
    count_type: str = "alive",
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Multi-panel histogram: one subplot per coral species showing the pooled
    per-cell (manual − AI) differences with each tile's contribution stacked.

    Tile display label is derived from the AI filename stem
    (e.g. "2025Dec-982091078941976"). A shared x-axis range is used across
    all panels so species can be compared directly.
    """
    def _tile_label(p) -> str:
        return p.ai_file.stem.removesuffix("_data")

    # Accumulate diffs: species → tile_label → [diff arrays]
    species_data: dict[str, dict[str, list]] = {}
    for p in pairs:
        m_arr = p.manual_alive if count_type == "alive" else p.manual_dead
        a_arr = p.ai_alive     if count_type == "alive" else p.ai_dead
        if m_arr is None or a_arr is None:
            continue
        sp = (p.species or "unknown").title()
        tl = _tile_label(p)
        d  = (m_arr.astype(float) - a_arr.astype(float)).flatten()
        species_data.setdefault(sp, {}).setdefault(tl, []).append(d)

    if not species_data:
        return None

    # Global bin range so panels share the same x-axis
    all_diffs = np.concatenate([
        d
        for tile_diffs in species_data.values()
        for dlist in tile_diffs.values()
        for d in dlist
    ])
    global_min = int(np.floor(all_diffs.min()))
    global_max = int(np.ceil(all_diffs.max()))
    bins = np.arange(global_min - 0.5, global_max + 1.5, 1)

    # Assign a consistent colour per tile across all panels
    all_tile_labels = sorted({tl for td in species_data.values() for tl in td})
    tile_colours = {tl: f"C{i}" for i, tl in enumerate(all_tile_labels)}

    species_list = sorted(species_data.keys())
    n = len(species_list)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, sp in zip(axes, species_list):
        tile_diffs = species_data[sp]
        sp_diffs = np.concatenate([d for dlist in tile_diffs.values() for d in dlist])
        mean_d = float(np.mean(sp_diffs))
        sd_d   = float(np.std(sp_diffs, ddof=1))

        bottom = np.zeros(len(bins) - 1)
        for tile_label, dlist in tile_diffs.items():
            gd = np.concatenate(dlist)
            counts, _ = np.histogram(gd, bins=bins)
            n_tp = len(dlist)
            ax.bar(bins[:-1] + 0.5, counts, width=1.0, bottom=bottom,
                   color=tile_colours[tile_label], alpha=0.75,
                   edgecolor="black", linewidth=0.5,
                   label=f"{tile_label} ({n_tp} tp{'s' if n_tp != 1 else ''})")
            bottom += counts

        ax.axvline(mean_d, color="red", linewidth=1.6, linestyle="--",
                   label=f"Mean = {mean_d:.2f}")
        ax.axvline(0, color="black", linewidth=0.9, linestyle=":")
        ax.set_xlabel("Difference (Manual − AI) per tab", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"{sp}\n(n = {len(sp_diffs):,} tabs)", fontsize=10)
        ax.legend(fontsize=8)

        stats_txt = (f"Mean = {mean_d:.2f}\n"
                     f"SD = {sd_d:.2f}\n"
                     f"n tabs = {len(sp_diffs):,}")
        ax.text(0.97, 0.95, stats_txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    fig.suptitle(
        f"Per-tab difference histogram by species — {count_type} counts",
        fontsize=12,
    )
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_combined_scatter(
    pairs,               # list[ComparisonPair]
    count_type: str = "alive",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Pooled scatter of all cell-level counts across all pairs,
    coloured by tile group (T05 vs Tile ID).
    """
    from .metrics import lins_ccc

    group_colours = {"T05": "steelblue", "Tile ID": "darkorange"}
    group_m: dict[str, list] = {"T05": [], "Tile ID": []}
    group_a: dict[str, list] = {"T05": [], "Tile ID": []}

    for p in pairs:
        m_arr = p.manual_alive if count_type == "alive" else p.manual_dead
        a_arr = p.ai_alive     if count_type == "alive" else p.ai_dead
        if m_arr is None or a_arr is None:
            continue
        group_m[p.tile_group].append(m_arr.flatten())
        group_a[p.tile_group].append(a_arr.flatten())

    all_m = np.concatenate([v for vals in group_m.values() for v in vals])
    all_a = np.concatenate([v for vals in group_a.values() for v in vals])
    lim = _lim(all_m, all_a)

    slope, intercept, r_val, _, _ = stats.linregress(all_m, all_a)
    ccc = lins_ccc(all_m, all_a)

    fig, ax = plt.subplots(figsize=(7, 6))
    for group, colour in group_colours.items():
        if not group_m[group]:
            continue
        gm = np.concatenate(group_m[group])
        ga = np.concatenate(group_a[group])
        ax.scatter(gm, ga, alpha=0.20, s=12, color=colour, label=group, zorder=2)

    x_line = np.linspace(0, lim, 200)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.4, label="1:1", zorder=3)
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=1.4,
            label=f"OLS  y = {slope:.2f}x + {intercept:.2f}", zorder=3)

    ax.set_xlim(left=0, right=lim)
    ax.set_ylim(bottom=0, top=lim)
    ax.set_xlabel("Manual count (per tab)")
    ax.set_ylabel("AI count (per tab)")
    ax.set_title(f"All pairs — {count_type} counts (n={len(all_m)} tabs)", fontsize=11)
    ax.legend(fontsize=8)
    ax.text(0.05, 0.95,
            f"R² = {r_val**2:.3f}\nCCC = {ccc:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_combined_bland_altman(
    pairs,
    count_type: str = "alive",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Pooled Bland-Altman across all pairs, coloured by tile group.
    """
    group_colours = {"T05": "steelblue", "Tile ID": "darkorange"}
    all_diff, all_mean = [], []
    group_diff: dict[str, list] = {"T05": [], "Tile ID": []}

    for p in pairs:
        m_arr = p.manual_alive if count_type == "alive" else p.manual_dead
        a_arr = p.ai_alive     if count_type == "alive" else p.ai_dead
        if m_arr is None or a_arr is None:
            continue
        d = m_arr.flatten().astype(float) - a_arr.flatten().astype(float)
        mv = (m_arr.flatten().astype(float) + a_arr.flatten().astype(float)) / 2
        all_diff.append(d)
        all_mean.append(mv)
        group_diff[p.tile_group].append(d)

    all_diff = np.concatenate(all_diff)
    all_mean = np.concatenate(all_mean)
    bias  = np.mean(all_diff)
    sd    = np.std(all_diff, ddof=1)
    loa_u = bias + 1.96 * sd
    loa_l = bias - 1.96 * sd

    prop_r, prop_p = stats.pearsonr(all_mean, all_diff)

    fig, ax = plt.subplots(figsize=(8, 5))
    for group, colour in group_colours.items():
        if not group_diff[group]:
            continue
        gd = np.concatenate(group_diff[group])
        gm_idx = [i for i, p in enumerate(pairs) if p.tile_group == group
                  and (p.manual_alive if count_type == "alive" else p.manual_dead) is not None]
        gm_vals = np.concatenate([
            ((p.manual_alive if count_type == "alive" else p.manual_dead).flatten() +
             (p.ai_alive     if count_type == "alive" else p.ai_dead).flatten()) / 2
            for p in pairs if p.tile_group == group
            and (p.manual_alive if count_type == "alive" else p.manual_dead) is not None
        ])
        ax.scatter(gm_vals, gd, alpha=0.20, s=12, color=colour, label=group, zorder=2)

    ax.axhline(bias,  color="black", linewidth=1.6, label=f"Bias = {bias:.2f}")
    ax.axhline(loa_u, color="red",   linewidth=1.2, linestyle="--",
               label=f"+1.96 SD = {loa_u:.2f}")
    ax.axhline(loa_l, color="red",   linewidth=1.2, linestyle="--",
               label=f"−1.96 SD = {loa_l:.2f}")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")

    if prop_p < 0.05:
        sl, ic, *_ = stats.linregress(all_mean, all_diff)
        x_ = np.linspace(all_mean.min(), all_mean.max(), 200)
        ax.plot(x_, sl * x_ + ic, "g-", linewidth=1.2,
                label=f"Prop. bias (r={prop_r:.2f}, p={prop_p:.3f})")

    ax.set_xlabel("Mean of manual and AI counts")
    ax.set_ylabel("Difference (Manual − AI)")
    ax.set_title(f"Pooled Bland-Altman — {count_type} counts", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_spatial_average_error(
    pairs,
    count_type: str = "alive",
    group_filter: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Average (manual − AI) difference matrix across multiple pairs,
    revealing systematic spatial biases in the AI model.
    """
    diff_matrices = []
    for p in pairs:
        if group_filter and p.tile_group != group_filter:
            continue
        m_arr = p.manual_alive if count_type == "alive" else p.manual_dead
        a_arr = p.ai_alive     if count_type == "alive" else p.ai_dead
        if m_arr is None or a_arr is None:
            continue
        diff_matrices.append(m_arr.astype(float) - a_arr.astype(float))

    if not diff_matrices:
        return None

    avg = np.mean(diff_matrices, axis=0)
    group_label = group_filter if group_filter else "all groups"
    n = len(diff_matrices)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(avg, cmap="coolwarm", center=0, annot=True, fmt=".1f", ax=ax,
                cbar_kws={"label": "Mean difference (Manual − AI)"})
    ax.set_title(
        f"Spatial average error — {count_type} counts\n"
        f"{group_label}, n={n} tile-timepoints",
        fontsize=11,
    )
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_time_histories(
    pairs,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    One subplot per tile: total alive manual vs AI counts over time.

    X-axis: days since first count for that tile (proxy for settlement age).
    Y-axis: total alive coral count summed across the 20x20 grid.
    """
    from collections import defaultdict

    by_tile: dict[str, list] = defaultdict(list)
    for p in pairs:
        if p.manual_alive is not None and p.ai_alive is not None:
            by_tile[p.tile_id].append(p)

    tiles = sorted(by_tile.keys())
    n = len(tiles)
    if n == 0:
        return None

    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for idx, tile_id in enumerate(tiles):
        ax = axes[idx // ncols][idx % ncols]
        tile_pairs = sorted(by_tile[tile_id], key=lambda p: p.date)
        t0 = tile_pairs[0].date
        days = [(p.date - t0).days for p in tile_pairs]

        man_totals = [float(p.manual_alive.sum()) for p in tile_pairs]
        ai_totals  = [float(p.ai_alive.sum())     for p in tile_pairs]

        # Error bars (spatial SD propagated to tile total) removed: they reflect
        # within-tile spatial heterogeneity, not repeated-measurement precision,
        # and would be misleading in a scientific paper context.
        # n_cells = tile_pairs[0].manual_alive.size  # 400
        # man_errs = [float(p.manual_alive.std() * np.sqrt(n_cells)) for p in tile_pairs]
        # ai_errs  = [float(p.ai_alive.std()     * np.sqrt(n_cells)) for p in tile_pairs]

        short = tile_id[-8:] if len(tile_id) > 8 else tile_id

        ax.plot(days, man_totals, "o-",  color="steelblue",
                label="Manual", linewidth=1.8, markersize=7, zorder=3)
        ax.plot(days, ai_totals,  "s--", color="darkorange",
                label="AI",     linewidth=1.5, markersize=7, alpha=0.85, zorder=2)

        ax.set_xlabel("Days since first count", fontsize=10)
        ax.set_ylabel("Total alive coral count", fontsize=10)
        ax.set_title(f"Tile {short}", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(left=-2)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=9)

    # Hide unused axes in the grid
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "Manual vs AI alive coral counts over time",
        fontsize=12,
    )
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_time_history_single(
    tile_pairs: list,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Time history figure for a single tile: total alive manual vs AI counts over time.
    """
    tile_pairs = sorted(tile_pairs, key=lambda p: p.date)
    t0 = tile_pairs[0].date

    days       = [(p.date - t0).days for p in tile_pairs]
    man_totals = [float(p.manual_alive.sum()) for p in tile_pairs]
    ai_totals  = [float(p.ai_alive.sum())     for p in tile_pairs]

    # Error bars (spatial SD propagated to tile total) removed: they reflect
    # within-tile spatial heterogeneity, not repeated-measurement precision,
    # and would be misleading in a scientific paper context.
    # n_cells  = tile_pairs[0].manual_alive.size
    # man_errs = [float(p.manual_alive.std() * np.sqrt(n_cells)) for p in tile_pairs]
    # ai_errs  = [float(p.ai_alive.std()     * np.sqrt(n_cells)) for p in tile_pairs]

    tile_label = tile_pairs[0].ai_file.stem.removesuffix("_data")
    species    = (tile_pairs[0].species or "").title()
    season     = tile_pairs[0].season or ""
    meta_line  = "  ·  ".join(filter(None, [species, season]))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(days, man_totals, "o-",  color="steelblue",
            label="Manual", linewidth=1.8, markersize=7, zorder=3)
    ax.plot(days, ai_totals,  "s--", color="darkorange",
            label="AI",     linewidth=1.5, markersize=7, alpha=0.85, zorder=2)

    ax.set_xlabel("Days since first count", fontsize=10)
    ax.set_ylabel("Total alive coral count", fontsize=10)
    title = f"Tile {tile_label}"
    if meta_line:
        title += f"\n{meta_line}"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(left=-2)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_growth_trend(
    pairs,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Total alive count per tile over time for manual and AI,
    showing whether the AI tracks colony growth trajectories.
    """
    from collections import defaultdict

    by_tile: dict[str, list] = defaultdict(list)
    for p in pairs:
        if p.manual_alive is not None and p.ai_alive is not None:
            by_tile[p.tile_id].append(p)

    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (tile_id, tile_pairs) in enumerate(sorted(by_tile.items())):
        tile_pairs = sorted(tile_pairs, key=lambda p: p.date)
        dates = [p.date for p in tile_pairs]
        man_totals = [float(p.manual_alive.sum()) for p in tile_pairs]
        ai_totals  = [float(p.ai_alive.sum())     for p in tile_pairs]

        colour = cmap(i % 10)
        short = tile_id[-8:] if len(tile_id) > 8 else tile_id
        ax.plot(dates, man_totals, "o-",  color=colour, linewidth=1.6,
                label=f"{short} manual")
        ax.plot(dates, ai_totals,  "s--", color=colour, linewidth=1.4,
                alpha=0.7, label=f"{short} AI")

    ax.set_xlabel("Capture date")
    ax.set_ylabel("Total coral count per tile")
    ax.set_title("Colony growth trajectory: Manual vs AI counts", fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.autofmt_xdate(rotation=25)
    plt.tight_layout()
    _save(fig, output_path)
    return fig
