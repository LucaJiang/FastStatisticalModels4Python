"""Helpers for k-means statistical validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import adjusted_rand_score


@dataclass(frozen=True)
class KMeansValidation:
    ari_true: float
    ari_vs_reference: float
    inertia_abs_diff: float
    inertia_rel_diff: float
    empty_clusters: int
    numerical_failure: bool
    correctness_status: str


def summarize_kmeans_result(
    labels: np.ndarray,
    true_labels: np.ndarray,
    inertia: float,
    k: int,
    reference_labels: np.ndarray | None = None,
    reference_inertia: float | None = None,
    rel_tol: float = 1e-8,
) -> KMeansValidation:
    empty = int(k - np.unique(labels).size)
    numerical_failure = not bool(np.isfinite(inertia))
    ari_true = float(adjusted_rand_score(true_labels, labels))

    if reference_labels is None or reference_inertia is None:
        ari_ref = 1.0
        abs_diff = 0.0
        rel_diff = 0.0
        status = "pass" if not numerical_failure else "fail"
    else:
        ari_ref = float(adjusted_rand_score(reference_labels, labels))
        abs_diff = float(abs(inertia - reference_inertia))
        denom = max(abs(reference_inertia), 1e-12)
        rel_diff = float(abs_diff / denom)
        status = "pass" if (not numerical_failure and math.isfinite(rel_diff) and rel_diff <= rel_tol) else "fail"

    return KMeansValidation(
        ari_true=ari_true,
        ari_vs_reference=ari_ref,
        inertia_abs_diff=abs_diff,
        inertia_rel_diff=rel_diff,
        empty_clusters=empty,
        numerical_failure=numerical_failure,
        correctness_status=status,
    )
