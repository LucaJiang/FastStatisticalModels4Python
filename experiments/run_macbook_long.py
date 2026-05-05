"""Orchestrate the MacBook Air long evidence run."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

from experiments.common.env_report import build_report


ROOT = Path("experiments/results/macbook_air_long")


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float_values(rows: list[dict[str, str]], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        try:
            vals.append(float(row[key]))
        except Exception:
            pass
    return vals


def _status_counts(rows: list[dict[str, str]], key: str = "correctness_status") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get(key) or row.get("status") or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


def _mean_by(rows: list[dict[str, str]], group_key: str, value_key: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        if not row.get(group_key):
            continue
        try:
            value = float(row[value_key])
        except Exception:
            continue
        groups.setdefault(row[group_key], []).append(value)
    return {key: round(fmean(values), 4) for key, values in sorted(groups.items()) if values}


def _mode_paths(root: Path, mode: str) -> dict[str, Path]:
    if mode == "long":
        preferred = {
            "kmeans": root / "kmeans_long_correctness.csv",
            "perm_eq": root / "permutation_long_equivalence.csv",
            "perm_cal": root / "permutation_long_calibration.csv",
            "perm_power": root / "permutation_long_power.csv",
            "perm_runtime": root / "permutation_long_runtime.csv",
        }
    else:
        preferred = {
            "kmeans": root / "kmeans_correctness.csv",
            "perm_eq": root / "permutation_equivalence.csv",
            "perm_cal": root / "permutation_calibration.csv",
            "perm_power": root / "permutation_power.csv",
            "perm_runtime": root / "permutation_quick_runtime.csv",
        }
    fallback = {
        "kmeans": root / "kmeans_correctness.csv",
        "perm_eq": root / "permutation_equivalence.csv",
        "perm_cal": root / "permutation_calibration.csv",
        "perm_power": root / "permutation_power.csv",
        "perm_runtime": root / "permutation_quick_runtime.csv",
    }
    return {key: path if path.exists() else fallback[key] for key, path in preferred.items()}


def write_summary(root: Path, mode: str) -> None:
    env = json.loads((root / "env.json").read_text()) if (root / "env.json").exists() else {}
    paths = _mode_paths(root, mode)
    kmeans = _rows(paths["kmeans"])
    perm_eq = _rows(paths["perm_eq"])
    perm_cal = _rows(paths["perm_cal"])
    perm_power = _rows(paths["perm_power"])
    perm_runtime = _rows(paths["perm_runtime"])

    kmeans_pass = [row for row in kmeans if row.get("correctness_status") == "pass"]
    runtime_by_impl = _mean_by(kmeans_pass, "implementation", "warm_median_s")
    ari_by_sep = _mean_by([row for row in kmeans_pass if row.get("implementation") in {"reference", "numpy_matmul"}], "separation", "ari_true")
    cal_props = _float_values(perm_cal, "prop_below_alpha")
    power_by_delta = _mean_by([row for row in perm_power if row.get("status") == "pass"], "delta", "signal_power")

    lines = [
        "# MacBook Air evidence summary",
        "",
        f"Run directory: `{root}`",
        f"Mode: `{mode}`",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Environment",
        "",
        f"- Python: {str(env.get('python_version', 'unknown')).split('|')[0].strip()}",
        f"- NumPy: {env.get('packages', {}).get('numpy', 'unknown')}",
        f"- Numba: {env.get('packages', {}).get('numba', 'unknown')}",
        f"- JAX: {env.get('packages', {}).get('jax', 'unknown')} ({env.get('jax', {}).get('jax_backend', 'unknown')})",
        f"- CPU cores: {env.get('cpu', {}).get('cpu_count_logical', 'unknown')}",
        f"- RAM GB: {env.get('ram_gb', 'unknown')}",
        "",
        "## k-means",
        "",
        f"- Rows: {len(kmeans)}",
        f"- Status counts: `{_status_counts(kmeans)}`",
        f"- Mean warm runtime by implementation: `{runtime_by_impl}`",
        f"- Mean ARI by separation: `{ari_by_sep}`",
        "",
        "## permutation test",
        "",
        f"- Equivalence rows: {len(perm_eq)}, status counts: `{_status_counts(perm_eq)}`",
        f"- Calibration rows: {len(perm_cal)}, status counts: `{_status_counts(perm_cal)}`",
        f"- Power rows: {len(perm_power)}, status counts: `{_status_counts(perm_power)}`",
        f"- Runtime rows: {len(perm_runtime)}",
        f"- Mean null `p <= 0.05`: {round(fmean(cal_props), 4) if cal_props else 'NA'}",
        f"- Mean power by delta: `{power_by_delta}`",
        "",
        "## Figures",
        "",
        "- `figures/kmeans_ari_heatmap.png`",
        "- `figures/kmeans_reference_equivalence.png`",
        "- `figures/kmeans_runtime_scaling.png`",
        "- `figures/kmeans_memory_scaling.png`",
        "- `figures/permutation_null_calibration.png`",
        "- `figures/permutation_power_quick.png`",
        "- `figures/permutation_runtime_heatmap.png`",
        "- `figures/permutation_equivalence.png`",
    ]
    (root / "LOCAL_LONG_SUMMARY.md").write_text("\n".join(lines) + "\n")

    notes = [
        "# Slides data notes",
        "",
        "Use these local long-run figures as MacBook Air evidence only. Do not label them as server CPU or GPU results.",
        "",
        "- k-means ARI and failure surface: cite `kmeans_ari_heatmap.png`.",
        "- k-means implementation preservation: cite `kmeans_reference_equivalence.png` and report only pass rows.",
        "- k-means local performance: cite `kmeans_runtime_scaling.png` and `kmeans_memory_scaling.png`; skipped broadcast rows are memory-risk skips, not zero timings.",
        "- permutation calibration: cite `permutation_null_calibration.png` and mean null `p <= 0.05` from `LOCAL_LONG_SUMMARY.md`.",
        "- permutation power: cite `permutation_power_quick.png`; it is a quick power curve, not a clinical operating-characteristic study.",
        "- JAX rows in this tier are CPU-only.",
    ]
    (root / "slides_data_notes.md").write_text("\n".join(notes) + "\n")

    devex = [
        "# Developer experience notes",
        "",
        "- Long runs append to CSV and can be resumed without rerunning completed scenario/implementation rows.",
        "- Rows that would allocate unsafe broadcast or full null matrices are recorded as `skipped_memory_risk`.",
        "- Per-scenario exceptions are recorded as `fail` rows with `notes`; they do not abort the whole run.",
        "- JAX is labeled CPU-only in this MacBook tier.",
    ]
    (root / "developer_experience_notes.md").write_text("\n".join(devex) + "\n")


def update_latest(root: Path) -> None:
    latest = ROOT / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_dir() and not latest.is_symlink():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    shutil.copytree(root, latest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full", "long"], default="long")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = ROOT / stamp
    else:
        root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        env = build_report("macbook_air_long", machine_name="macbook_air")
        (root / "env.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
        _run([sys.executable, "-m", "experiments.kmeans.run_mac_validation", "--mode", args.mode, "--output-dir", str(root), "--checkpoint-every", str(args.checkpoint_every), "--repeat", str(args.repeat), "--max-iter", str(args.max_iter)])
        _run([sys.executable, "-m", "experiments.permutation.run_mac_validation", "--mode", args.mode, "--output-dir", str(root), "--checkpoint-every", str(args.checkpoint_every)])

    _run([sys.executable, "-m", "experiments.kmeans.run_mac_validation", "--mode", args.mode, "--output-dir", str(root), "--regenerate-plots-only"])
    _run([sys.executable, "-m", "experiments.permutation.run_mac_validation", "--mode", args.mode, "--output-dir", str(root), "--regenerate-plots-only"])
    write_summary(root, args.mode)
    update_latest(root)
    print(root)


if __name__ == "__main__":
    main()
