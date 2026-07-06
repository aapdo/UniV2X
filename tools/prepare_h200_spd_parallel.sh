#!/usr/bin/env bash
set -euo pipefail

SOURCE=${1:-"${HOME}/dataset/V2X-Seq-SPD-FARM"}
DATA_NAME=${2:-"V2X-Seq-SPD-New"}
TARGET="datasets/${DATA_NAME}"
PYTHON=${PYTHON:-"${HOME}/venvs/unimmv2x_h200/bin/python"}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-"logs/h200_prepare_parallel/${RUN_ID}"}

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

run_side() {
    local side=$1
    local side_log=$2
    (
        set -euo pipefail
        echo "[prepare_h200_spd_parallel] ${side}: start $(date '+%F %T')"
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
        echo "[prepare_h200_spd_parallel] ${side}: done $(date '+%F %T')"
    ) > "${side_log}" 2>&1
}

if [ ! -d "${SOURCE}" ]; then
    echo "[prepare_h200_spd_parallel] missing source directory: ${SOURCE}" >&2
    exit 1
fi

mkdir -p datasets data/infos "${LOG_ROOT}"

if [ -e "${TARGET}" ] && [ ! -L "${TARGET}" ]; then
    echo "[prepare_h200_spd_parallel] target exists and is not a symlink: ${TARGET}" >&2
    exit 1
fi

ln -sfn "${SOURCE}" "${TARGET}"
echo "[prepare_h200_spd_parallel] linked ${TARGET} -> ${SOURCE}"
echo "[prepare_h200_spd_parallel] log root: ${LOG_ROOT}"

for side in vehicle-side infrastructure-side cooperative; do
    if [ ! -d "${TARGET}/${side}" ]; then
        echo "[prepare_h200_spd_parallel] missing side directory: ${TARGET}/${side}" >&2
        exit 1
    fi
done

declare -A pids
for side in vehicle-side infrastructure-side cooperative; do
    side_log="${LOG_ROOT}/${side}.log"
    echo "[prepare_h200_spd_parallel] launching ${side}; log=${side_log}"
    run_side "${side}" "${side_log}" &
    pids["${side}"]=$!
done

failed=0
for side in vehicle-side infrastructure-side cooperative; do
    if wait "${pids[${side}]}"; then
        echo "[prepare_h200_spd_parallel] ${side} finished"
    else
        echo "[prepare_h200_spd_parallel] ${side} failed; tail follows" >&2
        tail -c 4096 "${LOG_ROOT}/${side}.log" >&2 || true
        failed=1
    fi
done

if [ "${failed}" -ne 0 ]; then
    exit 1
fi

echo "[prepare_h200_spd_parallel] generated info files:"
find "data/infos/${DATA_NAME}" -type f -name "*.pkl" -print | sort
