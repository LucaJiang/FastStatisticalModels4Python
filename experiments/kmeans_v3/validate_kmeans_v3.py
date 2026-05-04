#!/usr/bin/env python3
"""Validate k-means v3 implementations on small shared scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_generation import KMeansScenario, initial_centroids, make_gaussian_mixture
from kmeans_numpy_broadcast import kmeans_numpy_broadcast
from kmeans_numpy_matmul import kmeans_numpy_matmul
from kmeans_reference import kmeans_reference


def ari_score(true_labels: np.ndarray, pred_labels: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import adjusted_rand_score
    except Exception:
        return None
    return float(adjusted_rand_score(true_labels, pred_labels))


def validate(include_jax: bool = False, include_numba: bool = False) -> dict:
    scenario = KMeansScenario(n=400, d=6, k=4, separation=3.0, seed=11)
    x, true_labels, _ = make_gaussian_mixture(scenario)
    init = initial_centroids(x, scenario.k, seed=99)
    impls = {
        "reference": kmeans_reference,
        "numpy_broadcast": kmeans_numpy_broadcast,
        "numpy_matmul": kmeans_numpy_matmul,
    }
    if include_numba:
        from kmeans_numba import kmeans_numba

        impls["numba"] = kmeans_numba
    if include_jax:
        from kmeans_jax import kmeans_jax

        impls["jax"] = kmeans_jax
    results = {}
    ref_inertia = None
    for name, fn in impls.items():
        centroids, labels, inertia, n_iter, empty = fn(x, init, max_iter=12)
        if ref_inertia is None:
            ref_inertia = inertia
        rel = abs(inertia - ref_inertia) / max(1.0, abs(ref_inertia))
        results[name] = {
            "inertia": float(inertia),
            "relative_inertia_delta": float(rel),
            "n_iter": int(n_iter),
            "empty_clusters": int(empty),
            "ari": ari_score(true_labels, labels),
            "status": "pass" if rel < 1e-6 or name == "jax" and rel < 5e-3 else "check",
            "centroid_checksum": float(np.sum(centroids)),
        }
    return {"scenario": scenario.__dict__, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-jax", action="store_true")
    parser.add_argument("--include-numba", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    data = validate(include_jax=args.include_jax, include_numba=args.include_numba)
    text = json.dumps(data, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
