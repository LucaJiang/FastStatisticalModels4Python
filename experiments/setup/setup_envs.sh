#!/usr/bin/env bash
# Create a local venv under experiments/setup/.venv-base and install requirements-base.txt (+ optional jax).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Set PYTHON to a python3 executable (default: python3)." >&2
  exit 1
fi

VENV="${VENV:-$ROOT/.venv-base}"
echo "Creating venv: $VENV"
"$PYTHON" -m venv "$VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"
python -m pip install -U pip
pip install -r "$ROOT/requirements-base.txt"

if [[ "${INSTALL_JAX:-0}" == "1" ]]; then
  pip install -r "$ROOT/requirements-jax.txt"
fi

echo "Done. Activate with: source $VENV/bin/activate"
