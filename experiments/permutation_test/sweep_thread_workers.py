#!/usr/bin/env python3
"""Sweep ThreadPool permutation performance across Python runtimes.

The talk uses this to compare the standard GIL build and the local
free-threaded Python 3.14 build while keeping the benchmark code and
problem size identical.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _python_probe(python: str) -> dict:
    code = (
        "import json, sys, platform; "
        "print(json.dumps({"
        "'python': sys.version.replace('\\n', ' '), "
        "'gil_enabled': getattr(sys, '_is_gil_enabled', lambda: None)(), "
        "'jit_available': getattr(getattr(sys, '_jit', None), 'is_available', lambda: None)(), "
        "'platform': platform.platform()"
        "}))"
    )
    out = subprocess.check_output([python, "-c", code], text=True)
    return json.loads(out)


def run_single(
    python: str,
    env_label: str,
    n1: int,
    n2: int,
    r: int,
    seed: int,
    warmup: int,
    repeat: int,
    workers: int,
    out_dir: Path,
) -> dict:
    out_json = out_dir / f"threads__{env_label}__w{workers}.json"
    cmd = [
        python,
        str(_HERE / "bench_permtest.py"),
        "--n1",
        str(n1),
        "--n2",
        str(n2),
        "--r",
        str(r),
        "--seed",
        str(seed),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--impl",
        "threads",
        "--max-workers",
        str(workers),
        "--output-json",
        str(out_json),
    ]
    print(f"  $ {env_label} workers={workers}", file=sys.stderr)
    subprocess.run(cmd, check=True, stdout=sys.stderr, stderr=sys.stderr)
    payload = json.loads(out_json.read_text())
    row = payload["rows"][0]
    row["env_label"] = env_label
    row["python_executable"] = python
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--py312-python", type=Path, required=True)
    p.add_argument("--py314t-python", type=Path, required=True)
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--n1", type=int, default=5000)
    p.add_argument("--n2", type=int, default=5000)
    p.add_argument("--r", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    runtimes = {
        "py312-gil": str(args.py312_python),
        "py314t-nogil": str(args.py314t_python),
    }
    probes = {label: _python_probe(python) for label, python in runtimes.items()}

    subprocess_dir = args.output_csv.parent / "_subprocess_threads"
    subprocess_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for label, python in runtimes.items():
        print(f"\n=== {label} ===", file=sys.stderr)
        for workers in args.workers:
            row = run_single(
                python,
                label,
                args.n1,
                args.n2,
                args.r,
                args.seed,
                args.warmup,
                args.repeat,
                workers,
                subprocess_dir,
            )
            row.update(
                {
                    "gil_enabled": probes[label].get("gil_enabled"),
                    "jit_available": probes[label].get("jit_available"),
                    "python_version": probes[label].get("python"),
                }
            )
            rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(rows[0].keys())
        with args.output_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                r2 = row.copy()
                r2["extra"] = json.dumps(r2.get("extra") or {})
                w.writerow(r2)
        print(f"\nWrote {args.output_csv}", file=sys.stderr)

    if args.output_json:
        with args.output_json.open("w") as f:
            json.dump({"env": probes, "rows": rows}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
