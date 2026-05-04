#!/usr/bin/env python3
"""Capture benchmark environment metadata for the v3 experiments."""

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


def _run(cmd: list[str]) -> dict[str, Any]:
    if shutil.which(cmd[0]) is None:
        return {"available": False, "cmd": cmd, "stdout": "", "stderr": ""}
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=20, check=False)
    except Exception as exc:
        return {"available": False, "cmd": cmd, "error": repr(exc), "stdout": "", "stderr": ""}
    return {
        "available": proc.returncode == 0,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _module_version(name: str) -> str | None:
    try:
        mod = __import__(name)
    except Exception:
        return None
    return getattr(mod, "__version__", None)


def _jax_info() -> dict[str, Any]:
    try:
        import jax
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    try:
        devices = [str(d) for d in jax.devices()]
    except Exception as exc:
        devices = [f"error: {exc!r}"]
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
    }


def _python_runtime_info() -> dict[str, Any]:
    gil_enabled = None
    if hasattr(sys, "_is_gil_enabled"):
        try:
            gil_enabled = bool(sys._is_gil_enabled())  # type: ignore[attr-defined]
        except Exception:
            gil_enabled = None
    jit_info: dict[str, Any] = {"present": hasattr(sys, "_jit")}
    if hasattr(sys, "_jit"):
        try:
            jit_info["available"] = bool(sys._jit.is_available())  # type: ignore[attr-defined]
            jit_info["enabled"] = bool(sys._jit.is_enabled())  # type: ignore[attr-defined]
        except Exception as exc:
            jit_info["error"] = repr(exc)
    return {
        "version": sys.version.replace("\n", " "),
        "version_info": list(sys.version_info[:5]),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "py_gil_disabled_config": sysconfig.get_config_var("Py_GIL_DISABLED"),
        "gil_enabled": gil_enabled,
        "jit": jit_info,
    }


def capture() -> dict[str, Any]:
    return {
        "python": _python_runtime_info(),
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "packages": {
            name: _module_version(name)
            for name in (
                "numpy",
                "scipy",
                "numba",
                "sklearn",
                "matplotlib",
                "psutil",
                "pandas",
                "pyarrow",
                "jax",
            )
        },
        "jax": _jax_info(),
        "commands": {
            "nvidia_smi": _run(["nvidia-smi"]),
            "free_h": _run(["free", "-h"]),
            "lscpu": _run(["lscpu"]),
            "df_home": _run(["df", "-h", "/home/wjiang49", "/home/wjiang49/conda_envs"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/v3/environment.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = capture()
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
