#!/usr/bin/env python3
"""Sweep permutation test by shelling out to ``bench_permtest.py`` once
per implementation.

Running each implementation in its own subprocess prevents the big
``numpy_batched`` allocation (~2 GiB at R=10 000) from poisoning the
timings of impls that come after it in the same process. That single
choice made our measured times move by >5× for the lightweight impls;
it's the kind of thing that's easy to miss in a shared notebook.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def run_single(python: str, impl: str, n1: int, n2: int, r: int,
               seed: int, warmup: int, repeat: int,
               max_workers: int | None, out_dir: Path) -> Path:
    out_json = out_dir / f"perm__{impl}__R{r}.json"
    cmd = [
        python,
        str(_HERE / "bench_permtest.py"),
        "--n1", str(n1), "--n2", str(n2),
        "--r", str(r), "--seed", str(seed),
        "--warmup", str(warmup), "--repeat", str(repeat),
        "--impl", impl,
        "--output-json", str(out_json),
    ]
    if max_workers is not None:
        cmd += ["--max-workers", str(max_workers)]
    print(f"  $ {' '.join(cmd[-6:])}", file=sys.stderr)
    subprocess.run(cmd, check=True, stdout=sys.stderr, stderr=sys.stderr)
    return out_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--n1", type=int, default=5000)
    p.add_argument("--n2", type=int, default=5000)
    p.add_argument("--r-list", type=int, nargs="+", default=[500, 2000, 10_000])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--skip-jax", action="store_true")
    p.add_argument("--impls", nargs="+", default=None)
    args = p.parse_args()

    impls = args.impls or [
        "numpy_naive",
        "numpy_trick",
        "multiprocessing",
        "threads",
        "numba",
        "numpy_batched",
        "numpy_trick_batched",
        "jax_perm",
        "jax_trick",
    ]
    if args.skip_jax:
        impls = [i for i in impls if not i.startswith("jax")]

    subprocess_dir = args.output_csv.parent / "_subprocess"
    subprocess_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    env_seen: dict | None = None

    for r in args.r_list:
        print(f"\n=== R = {r:,} ===", file=sys.stderr)
        for impl in impls:
            print(f"  → {impl}", file=sys.stderr)
            try:
                path = run_single(
                    args.python, impl,
                    args.n1, args.n2, r, args.seed,
                    args.warmup, args.repeat, args.max_workers,
                    subprocess_dir,
                )
            except subprocess.CalledProcessError as e:
                print(f"    (failed, exit {e.returncode})", file=sys.stderr)
                continue
            payload = json.loads(path.read_text())
            if env_seen is None:
                env_seen = payload.get("env")
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
