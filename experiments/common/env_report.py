#!/usr/bin/env python3
"""Environment report for environment-tiered experiments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any


def run_cmd(cmd: list[str], timeout: int = 20) -> dict[str, Any]:
    if shutil.which(cmd[0]) is None:
        return {"available": False, "cmd": cmd, "stdout": "", "stderr": ""}
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except Exception as exc:
        return {"available": False, "cmd": cmd, "error": repr(exc), "stdout": "", "stderr": ""}
    return {
        "available": proc.returncode == 0,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def mod_version(name: str) -> str | None:
    try:
        mod = __import__(name)
    except Exception:
        return None
    return getattr(mod, "__version__", None)


def jax_info() -> dict[str, Any]:
    try:
        import jax
    except Exception as exc:
        return {"available": False, "error": repr(exc), "cuda_visible": False}
    try:
        devices = [str(d) for d in jax.devices()]
    except Exception as exc:
        devices = [f"error: {exc!r}"]
    backend = None
    try:
        backend = jax.default_backend()
    except Exception as exc:
        backend = f"error: {exc!r}"
    return {
        "available": True,
        "jax_version": getattr(jax, "__version__", None),
        "jaxlib_version": getattr(jax.lib, "__version__", None),
        "default_backend": backend,
        "devices": devices,
        "cuda_visible": any("cuda" in d.lower() or "gpu" in d.lower() for d in devices) or backend == "gpu",
    }


def python_info() -> dict[str, Any]:
    gil_enabled = None
    if hasattr(sys, "_is_gil_enabled"):
        try:
            gil_enabled = bool(sys._is_gil_enabled())  # type: ignore[attr-defined]
        except Exception:
            gil_enabled = None
    jit: dict[str, Any] = {"present": hasattr(sys, "_jit")}
    if hasattr(sys, "_jit"):
        try:
            jit["available"] = bool(sys._jit.is_available())  # type: ignore[attr-defined]
            jit["enabled"] = bool(sys._jit.is_enabled())  # type: ignore[attr-defined]
        except Exception as exc:
            jit["error"] = repr(exc)
    return {
        "executable": sys.executable,
        "version": sys.version.replace("\n", " "),
        "version_info": list(sys.version_info[:5]),
        "py_gil_disabled_config": sysconfig.get_config_var("Py_GIL_DISABLED"),
        "gil_enabled": gil_enabled,
        "jit": jit,
    }


def report(environment_tier: str) -> dict[str, Any]:
    return {
        "environment_tier": environment_tier,
        "machine_name": platform.node(),
        "python": python_info(),
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "packages": {
            name: mod_version(name)
            for name in ("numpy", "scipy", "numba", "sklearn", "matplotlib", "psutil", "pandas", "pyarrow", "jax")
        },
        "threading_env": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "XLA_FLAGS")
        },
        "jax": jax_info(),
        "commands": {
            "nvidia_smi": run_cmd(["nvidia-smi"]),
            "free_h": run_cmd(["free", "-h"]),
            "lscpu": run_cmd(["lscpu"]),
            "df": run_cmd(["df", "-h", "/home/wjiang49", "/home/wjiang49/conda_envs"]),
            "conda_list_explicit": run_cmd(["conda", "list", "-p", sys.prefix], timeout=60),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", required=True, choices=["linux_server_cpu", "linux_server_a100"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = report(args.tier)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
