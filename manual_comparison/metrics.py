#!/usr/bin/env python3
"""
metrics.py

Statistical metrics for comparing two 20×20 count matrices.
All functions accept flat or 2-D numpy arrays.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Lin's Concordance Correlation Coefficient.
    Measures both precision (correlation) and accuracy (proximity to 1:1 line).
    Range [-1, 1]; 1 = perfect agreement.
    """
    x, y = x.flatten().astype(float), y.flatten().astype(float)
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x  = np.var(x, ddof=0)
    var_y  = np.var(y, ddof=0)
    covar  = np.cov(x, y, ddof=0)[0, 1]
    denom  = var_x + var_y + (mean_x - mean_y) ** 2
    return float(2 * covar / denom) if denom > 0 else 0.0


def bland_altman_stats(manual: np.ndarray, ai: np.ndarray) -> dict:
    """
    Bias and limits of agreement (Bland & Altman 1986).
    Also tests for proportional bias (correlation of difference with mean).
    """
    d = manual.flatten().astype(float) - ai.flatten().astype(float)
    m = (manual.flatten().astype(float) + ai.flatten().astype(float)) / 2
    bias = float(np.mean(d))
    sd   = float(np.std(d, ddof=1))

    try:
        prop_r, prop_p = stats.pearsonr(m, d)
    except Exception:
        prop_r, prop_p = np.nan, np.nan

    return {
        "bias":      bias,
        "sd_diff":   sd,
        "loa_lower": bias - 1.96 * sd,
        "loa_upper": bias + 1.96 * sd,
        "prop_bias_r": float(prop_r),
        "prop_bias_p": float(prop_p),
    }


def agreement_rates(manual: np.ndarray, ai: np.ndarray,
                    tolerances: tuple[int, ...] = (0, 1, 2)) -> dict[int, float]:
    """Fraction of cells where |manual − AI| ≤ tolerance."""
    diff = np.abs(manual.flatten().astype(float) - ai.flatten().astype(float))
    n = len(diff)
    return {t: float(np.sum(diff <= t) / n) for t in tolerances}


def relative_errors(manual: np.ndarray, ai: np.ndarray) -> np.ndarray:
    """Per-cell (AI − manual) / manual on non-zero manual cells."""
    m = manual.flatten().astype(float)
    a = ai.flatten().astype(float)
    mask = m > 0
    if not np.any(mask):
        return np.array([])
    return (a[mask] - m[mask]) / m[mask]


def compute_all_metrics(manual: np.ndarray, ai: np.ndarray) -> dict:
    """
    Compute the full set of comparison metrics between two 20×20 matrices.

    Returns a flat dict suitable for building a summary table row.
    """
    m = manual.flatten().astype(float)
    a = ai.flatten().astype(float)
    diff = m - a
    abs_diff = np.abs(diff)

    # Correlation
    try:
        pearson_r, pearson_p = stats.pearsonr(m, a)
    except Exception:
        pearson_r, pearson_p = np.nan, np.nan

    try:
        spearman_r, spearman_p = stats.spearmanr(m, a)
    except Exception:
        spearman_r, spearman_p = np.nan, np.nan

    # Wilcoxon signed-rank: are cell-level differences systematically non-zero?
    try:
        wilcox_stat, wilcox_p = stats.wilcoxon(diff)
    except ValueError:
        # All differences are zero or too few non-zero values
        wilcox_stat, wilcox_p = np.nan, 1.0

    total_m = float(np.sum(m))
    total_a = float(np.sum(a))
    pct_diff = 100.0 * (total_m - total_a) / total_m if total_m > 0 else np.nan

    ba = bland_altman_stats(manual, ai)
    agree = agreement_rates(manual, ai)
    rel_err = relative_errors(manual, ai)

    return {
        "total_manual":     total_m,
        "total_ai":         total_a,
        "pct_diff":         pct_diff,
        "mean_diff":        float(np.mean(diff)),
        "sd_diff":          float(np.std(diff, ddof=1)),
        "mae":              float(np.mean(abs_diff)),
        "max_abs_diff":     float(np.max(abs_diff)),
        "pearson_r":        float(pearson_r),
        "pearson_p":        float(pearson_p),
        "spearman_r":       float(spearman_r),
        "spearman_p":       float(spearman_p),
        "lins_ccc":         lins_ccc(manual, ai),
        "ba_bias":          ba["bias"],
        "ba_sd":            ba["sd_diff"],
        "ba_loa_lower":     ba["loa_lower"],
        "ba_loa_upper":     ba["loa_upper"],
        "ba_prop_bias_r":   ba["prop_bias_r"],
        "ba_prop_bias_p":   ba["prop_bias_p"],
        "agree_exact":      agree[0],
        "agree_within_1":   agree[1],
        "agree_within_2":   agree[2],
        "rel_error_mean":   float(np.mean(rel_err)) if len(rel_err) > 0 else np.nan,
        "rel_error_std":    float(np.std(rel_err))  if len(rel_err) > 0 else np.nan,
        "wilcoxon_stat":    float(wilcox_stat) if not np.isnan(wilcox_stat) else np.nan,
        "wilcoxon_p":       float(wilcox_p),
    }
