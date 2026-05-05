"""Validation helpers for permutation-test experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PermutationEquivalence:
    max_abs_stat_diff: float
    max_abs_p_diff: float
    mean_abs_p_diff: float
    status: str


def summarize_equivalence(
    reference_null: np.ndarray,
    reference_p: np.ndarray,
    candidate_null: np.ndarray,
    candidate_p: np.ndarray,
    *,
    abs_tol: float = 1e-10,
) -> PermutationEquivalence:
    max_abs_stat_diff = float(np.max(np.abs(np.asarray(reference_null) - np.asarray(candidate_null))))
    p_diff = np.abs(np.asarray(reference_p) - np.asarray(candidate_p))
    max_abs_p_diff = float(np.max(p_diff))
    mean_abs_p_diff = float(np.mean(p_diff))
    status = "pass" if max_abs_stat_diff <= abs_tol and max_abs_p_diff <= abs_tol else "fail"
    return PermutationEquivalence(max_abs_stat_diff, max_abs_p_diff, mean_abs_p_diff, status)


def calibration_summary(p_values: np.ndarray, *, alpha: float = 0.05) -> dict[str, float]:
    p_values = np.asarray(p_values)
    sorted_p = np.sort(p_values)
    n = sorted_p.size
    if n == 0:
        return {"mean_p": float("nan"), "median_p": float("nan"), "prop_below_alpha": float("nan"), "ks_uniform": float("nan")}
    empirical = np.arange(1, n + 1) / n
    ks_uniform = float(np.max(np.maximum(np.abs(empirical - sorted_p), np.abs((np.arange(n) / n) - sorted_p))))
    return {
        "mean_p": float(np.mean(p_values)),
        "median_p": float(np.median(p_values)),
        "prop_below_alpha": float(np.mean(p_values <= alpha)),
        "ks_uniform": ks_uniform,
    }


def power_summary(p_values: np.ndarray, signal_mask: np.ndarray, *, alpha: float = 0.05) -> dict[str, float]:
    p_values = np.asarray(p_values)
    signal_mask = np.asarray(signal_mask, dtype=bool)
    null_mask = ~signal_mask
    signal_power = float(np.mean(p_values[signal_mask] <= alpha)) if np.any(signal_mask) else float("nan")
    null_fpr = float(np.mean(p_values[null_mask] <= alpha)) if np.any(null_mask) else float("nan")
    return {
        "signal_power": signal_power,
        "null_false_positive_rate": null_fpr,
        "discoveries": float(np.sum(p_values <= alpha)),
    }
