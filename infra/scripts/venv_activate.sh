#!/bin/bash
set -ex

# Active or create if it doesn't exist, a base VENV to install some core
# dependencies through PIP. This lets us pull in modern cmake etc. Keeping the
# venv active allows these tools to be available on the path.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"

# Create VENV if it doesn't exist
if ! test -f "$ROOT_DIR/venv/bin/activate"; then
  echo "[Creating new VENV in $ROOT_DIR/venv]"
  python3 -m venv "$ROOT_DIR/venv"
  source "$ROOT_DIR/venv/bin/activate"
  pip install -U pip
  pip install -U wheel
  pip install -U cmake ninja

  # Add hook in activate script to also set some env variables
  echo "source $ROOT_DIR/scripts/venv_export_env_vars.sh" >> "$ROOT_DIR/venv/bin/activate"
fi

echo "[Activating VENV in $ROOT_DIR/venv]"
source "$ROOT_DIR/venv/bin/activate"
