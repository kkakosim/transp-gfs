#!/usr/bin/env bash
#
# Convenience wrapper for one Qatar GFS->3D.DAT conversion.
#
# Usage:
#   scripts/run_qatar.sh                     # default config_qatar.yaml
#   scripts/run_qatar.sh path/to/run.yaml    # override config path
#   scripts/run_qatar.sh --skip-download     # reuse cached GRIB2

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CFG="config_qatar.yaml"
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        *.yaml|*.yml) CFG="$arg" ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

if [[ ! -f "${CFG}" ]]; then
    echo "Config not found: ${CFG}" >&2
    exit 1
fi

# Activate a venv if it's there (no-op if you're already in a conda env).
if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Ensure the output dir exists.
OUT_PATH=$(python - <<PY
import yaml, sys
with open("${CFG}") as f:
    cfg = yaml.safe_load(f)
print(cfg["output_path"])
PY
)
mkdir -p "$(dirname "${OUT_PATH}")"

echo "Config:    ${CFG}"
echo "Output:    ${OUT_PATH}"
echo "Extras:    ${EXTRA_ARGS[*]:-(none)}"
echo

gfs2calmet "${CFG}" -v "${EXTRA_ARGS[@]}"
