#!/usr/bin/env python3
"""Sweep k-means by shelling out to ``bench_kmeans.py`` once per impl.

Running each impl in a fresh subprocess matters less here than for the
permutation test (no 2 GiB allocations), but we still do it so cold
times reflect a fresh JIT compile rather than a warmed-up kernel from
a previous step.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def run_single(python: str, impl: str, n: int, n_features: int, k: int,
               centers: int, max_iter: int, seed: int, warmup: int,
               repeat: int, loops_max_n: int, out_dir: Path) -> Path:
    out_json = out_dir / f"kmeans__{impl}__N{n}.json"
    cmd = [
        python,
        str(_HERE / "bench_kmeans.py"),
        "--n-samples", str(n),
        "--n-features", str(n_features),
        "--k", str(k),
        "--centers", str(centers),
        "--max-iter", str(max_iter),
        "--seed", str(seed),
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--loops-max-n", str(loops_max_n),
        "--impl", impl,
        "--output-json", str(out_json),
    ]
    print(f"  $ {' '.join(cmd[-7:])}", file=sys.stderr)
    subprocess.run(cmd, check=True, stdout=sys.stderr, stderr=sys.stderr)
    return out_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--n-list", type=int, nargs="+",
                   default=[10_000, 100_000, 500_000, 1_000_000])
    p.add_argument("--loops-n", type=int, default=2000)
    p.add_argument("--n-features", type=int, default=10)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--skip-jax", action="store_true")
    p.add_argument(
        "--impls",
        nargs="+",
        choices=["numpy_naive", "numpy_smart", "numba", "jax"],
        default=None,
        help="Main implementations to run. Defaults to NumPy naive/smart, Numba, and JAX unless --skip-jax is set.",
    )
    p.add_argument(
        "--skip-loops",
        action="store_true",
        help="Do not append the small-N pure-Python loops point.",
    )
    p.add_argument(
        "--max-numpy-naive-n",
        type=int,
        default=None,
        help="Skip numpy_naive above this N. Useful for high-dimensional sweeps where the broadcast temp would be too large.",
    )
    args = p.parse_args()

    impls_main = args.impls or ["numpy_naive", "numpy_smart", "numba", "jax"]
    if args.skip_jax:
        impls_main = [i for i in impls_main if i != "jax"]

    subprocess_dir = args.output_csv.parent / "_subprocess_km"
    subprocess_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    env_seen: dict | None = None

    for n in args.n_list:
        print(f"\n=== N = {n:,} ===", file=sys.stderr)
        for impl in impls_main:
            if (
                impl == "numpy_naive"
                and args.max_numpy_naive_n is not None
                and n > args.max_numpy_naive_n
            ):
                print(
                    f"  -> {impl} skipped above N={args.max_numpy_naive_n:,}",
                    file=sys.stderr,
                )
                continue
            print(f"  → {impl}", file=sys.stderr)
            try:
                path = run_single(
                    args.python, impl, n, args.n_features, args.k, args.k,
                    args.max_iter, args.seed, args.warmup, args.repeat,
                    args.loops_n, subprocess_dir,
                )
            except subprocess.CalledProcessError as e:
                print(f"    (failed, exit {e.returncode})", file=sys.stderr)
                continue
            payload = json.loads(path.read_text())
            if env_seen is None:
                env_seen = payload.get("env")
            all_rows.extend(payload["rows"])

    if not args.skip_loops:
        # Loops at small N, one subprocess
        print(f"\n=== loops @ N={args.loops_n} ===", file=sys.stderr)
        path = run_single(
            args.python, "loops", args.loops_n, args.n_features, args.k, args.k,
            args.max_iter, args.seed, 1, 2, args.loops_n, subprocess_dir,
        )
        payload = json.loads(path.read_text())
        all_rows.extend(payload["rows"])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        fields = list(all_rows[0].keys())
        with args.output_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in all_rows:
                r2 = row.copy()
                r2["extra"] = json.dumps(r2.get("extra") or {})
                w.writerow(r2)
        print(f"\nWrote {args.output_csv}", file=sys.stderr)

    if args.output_json:
        with args.output_json.open("w") as f:
            json.dump({"env": env_seen, "rows": all_rows}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
