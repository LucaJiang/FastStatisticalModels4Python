#!/usr/bin/env python3
"""Check local k-means implementations against sklearn with fixed init."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_gen import make_cluster_data  # noqa: E402
from kmeans_numpy import kmeans_numpy_smart  # noqa: E402


def _initial_centroids(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    return X[idx].astype(np.float64, copy=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--n-features", type=int, default=10)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--centers", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    X, _ = make_cluster_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        centers=args.centers,
        random_state=args.seed,
    )
    init = _initial_centroids(X, args.k, args.seed)

    _, _, ours_inertia, ours_iter = kmeans_numpy_smart(
        X, args.k, args.max_iter, init
    )
    sklearn_model = KMeans(
        n_clusters=args.k,
        init=init.copy(),
        n_init=1,
        max_iter=args.max_iter,
        tol=0.0,
        algorithm="lloyd",
        random_state=args.seed,
    ).fit(X)

    sklearn_inertia = float(sklearn_model.inertia_)
    abs_diff = abs(float(ours_inertia) - sklearn_inertia)
    rel_diff = abs_diff / max(1.0, abs(sklearn_inertia))
    passed = rel_diff <= args.rtol

    payload = {
        "passed": passed,
        "ours_inertia": float(ours_inertia),
        "sklearn_inertia": sklearn_inertia,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "ours_iter": int(ours_iter),
        "sklearn_iter": int(sklearn_model.n_iter_),
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "k": args.k,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "rtol": args.rtol,
    }
    print(json.dumps(payload, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
