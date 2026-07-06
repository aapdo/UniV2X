#!/usr/bin/env bash
set -euo pipefail

SOURCE=${1:-"${HOME}/dataset/V2X-Seq-SPD-FARM"}
DATA_NAME=${2:-"V2X-Seq-SPD-New"}
TARGET="datasets/${DATA_NAME}"
PYTHON=${PYTHON:-"${HOME}/venvs/unimmv2x_h200/bin/python"}

run_converter() {
    local script=$1
    shift
    PYTHONPATH="${PWD}:${PWD}/tools/spd_data_converter:${PYTHONPATH:-}" \
        "${PYTHON}" - "$script" "$@" <<'PY'
import runpy
import sys

import h200_compat_smoke  # noqa: F401

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
PY
}

if [ ! -d "${SOURCE}" ]; then
    echo "[prepare_h200_spd] missing source directory: ${SOURCE}" >&2
    exit 1
fi

mkdir -p datasets data/infos

if [ -e "${TARGET}" ] && [ ! -L "${TARGET}" ]; then
    echo "[prepare_h200_spd] target exists and is not a symlink: ${TARGET}" >&2
    exit 1
fi

ln -sfn "${SOURCE}" "${TARGET}"
echo "[prepare_h200_spd] linked ${TARGET} -> ${SOURCE}"

for side in vehicle-side infrastructure-side cooperative; do
    if [ ! -d "${TARGET}/${side}" ]; then
        echo "[prepare_h200_spd] missing side directory: ${TARGET}/${side}" >&2
        exit 1
    fi
done

for side in vehicle-side infrastructure-side cooperative; do
    echo "[prepare_h200_spd] converting ${side}"
    run_converter tools/spd_data_converter/spd_to_uniad.py \
        --data-root "./datasets/${DATA_NAME}" \
        --save-root "./data/infos/${DATA_NAME}" \
        --v2x-side "${side}"
    run_converter tools/spd_data_converter/spd_to_nuscenes.py \
        --data-root "./datasets/${DATA_NAME}" \
        --save-root "./datasets/${DATA_NAME}" \
        --v2x-side "${side}"
    run_converter tools/spd_data_converter/map_spd_to_nuscenes.py \
        --maps-root "./datasets/${DATA_NAME}/maps" \
        --save-root "./datasets/${DATA_NAME}" \
        --v2x-side "${side}"
done

echo "[prepare_h200_spd] generated info files:"
find "data/infos/${DATA_NAME}" -type f -name "*.pkl" -print | sort
