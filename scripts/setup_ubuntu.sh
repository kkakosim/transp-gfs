#!/usr/bin/env bash
#
# Ubuntu setup for gfs2calmet — non-conda path.
#
# Use this if you'd rather install eccodes from apt and pygrib from PyPI
# inside a regular venv. The conda path (environment.yml) is simpler;
# pick whichever fits your workflow.
#
# Tested on Ubuntu 22.04 LTS.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[1/4] Installing system packages (sudo required for apt)..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    libeccodes-dev libeccodes-tools \
    build-essential

echo "[2/4] Creating Python virtual environment at .venv ..."
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[3/4] Upgrading pip and installing project + pipeline extras..."
python -m pip install --upgrade pip wheel
# pygrib needs libeccodes-dev present at compile time (it links against it).
python -m pip install -e ".[pipeline,test]"

echo "[4/4] Verifying imports..."
python - <<'PY'
import importlib
mods = ["numpy", "xarray", "pyproj", "yaml", "pygrib", "herbie"]
for m in mods:
    importlib.import_module(m)
    print(f"  ok  {m}")
PY

echo
echo "Setup complete."
echo "Activate the environment with:   source .venv/bin/activate"
echo "Run the converter with:          gfs2calmet config_qatar.yaml -v"
