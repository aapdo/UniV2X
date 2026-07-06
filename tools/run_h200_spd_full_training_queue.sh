#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-"${HOME}/UniMM-V2X"}
cd "${REPO}"

PYTHON=${PYTHON:-"${HOME}/venvs/unimmv2x_h200/bin/python"}
export PATH="${HOME}/venvs/unimmv2x_h200/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

RUN_ID=${RUN_ID:-"$(date +%Y%m%d_%H%M%S)"}
LOGDIR=${LOGDIR:-"${PWD}/logs/h200_spd_full_training/${RUN_ID}"}
mkdir -p "${LOGDIR}"
QUEUE_LOG="${LOGDIR}/queue.log"

PREP_SESSION=${PREP_SESSION:-unimm_h200_prepare3}
PREP_LOG=${PREP_LOG:-"${PWD}/logs/h200_prepare/prepare3.log"}

GPU_AUTO_SELECT=${GPU_AUTO_SELECT:-1}
GPU_FIXED=${GPU_FIXED:-1}
GPU_SECOND_CANDIDATES=${GPU_SECOND_CANDIDATES:-0,3}
GPU_LIST=${GPU_LIST:-1,0}
GPUS=${GPUS:-2}
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export WANDB_PROJECT=${WANDB_PROJECT:-unimmv2x-h200-spd}
export WANDB_MODE=${WANDB_MODE:-online}
export HF_REPO_ID=${HF_REPO_ID:-apdoa/UniMM-V2X-H200-SPD}
export HF_CLI=${HF_CLI:-"${HOME}/miniconda3/envs/mrlora/bin/hf"}

TUNE_COMBOS=${TUNE_COMBOS:-"8:1 6:1 4:1 2:2 1:4"}
TUNE_MAX_ITERS=${TUNE_MAX_ITERS:-2}
TUNE_WORKERS=${TUNE_WORKERS:-2}
TRAIN_WORKERS=${TRAIN_WORKERS:-4}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29620}
GPU_FREE_MAX_MEM_MB=${GPU_FREE_MAX_MEM_MB:-500}
GPU_FREE_MAX_UTIL=${GPU_FREE_MAX_UTIL:-5}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${QUEUE_LOG}"
}

select_gpu_list() {
    if [ "${GPU_AUTO_SELECT}" != "1" ]; then
        export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
        log "GPU auto-select disabled: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        return 0
    fi

    local selected_second
    selected_second=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
        "${PYTHON}" - "${GPU_FIXED}" "${GPU_SECOND_CANDIDATES}" <<'PY'
import sys

fixed = int(sys.argv[1])
candidates = [int(x.strip()) for x in sys.argv[2].split(",") if x.strip()]
stats = {}
for line in sys.stdin:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        continue
    idx, mem, util = map(int, parts)
    stats[idx] = (mem, util)

available = [idx for idx in candidates if idx in stats and idx != fixed]
if not available:
    raise SystemExit(f"no selectable GPU from candidates={candidates}, fixed={fixed}")

selected = min(available, key=lambda idx: (stats[idx][0], stats[idx][1], idx))
print(selected)
PY
)
    GPU_LIST="${GPU_FIXED},${selected_second}"
    export GPU_LIST
    export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
    log "selected GPUs at training start: fixed_gpu=${GPU_FIXED}, candidates=${GPU_SECOND_CANDIDATES}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
}

has_error_pattern() {
    local path=$1
    grep -Eiq "OOM|out of memory|Traceback|RuntimeError|CUDA error|Killed|No such file|ImportError|AttributeError|ZeroDivisionError" "${path}"
}

wait_for_prepare() {
    if tmux has-session -t "${PREP_SESSION}" 2>/dev/null; then
        log "waiting for data preparation tmux session: ${PREP_SESSION}"
    fi
    while tmux has-session -t "${PREP_SESSION}" 2>/dev/null; do
        if [ -f "${PREP_LOG}" ] && has_error_pattern "${PREP_LOG}"; then
            log "data preparation error detected in ${PREP_LOG}"
            exit 1
        fi
        sleep 60
    done
    if [ -f "${PREP_LOG}" ] && has_error_pattern "${PREP_LOG}"; then
        log "data preparation finished with error pattern in ${PREP_LOG}"
        exit 1
    fi
}

validate_infos() {
    log "validating SPD info files"
    for side in vehicle-side infrastructure-side cooperative; do
        for split in train val; do
            local path="data/infos/V2X-Seq-SPD-New/${side}/spd_infos_temporal_${split}.pkl"
            if [ ! -s "${path}" ]; then
                log "missing or empty info file: ${path}"
                exit 1
            fi
        done
    done

    "${PYTHON}" - <<'PY' | tee -a "${QUEUE_LOG}"
import h200_compat_smoke  # noqa: F401
import mmcv

for side in ("vehicle-side", "infrastructure-side", "cooperative"):
    for split in ("train", "val"):
        path = f"data/infos/V2X-Seq-SPD-New/{side}/spd_infos_temporal_{split}.pkl"
        data = mmcv.load(path)
        infos = data.get("infos", data if isinstance(data, list) else [])
        print(f"[info] {side} {split}: {len(infos)} rows")
PY
}

wait_for_gpus() {
    log "checking GPU availability for CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    while true; do
        local busy
        busy=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
            "${PYTHON}" -c '
import sys

gpu_list = {int(x.strip()) for x in sys.argv[1].split(",") if x.strip()}
max_mem = int(sys.argv[2])
max_util = int(sys.argv[3])
busy = []
for line in sys.stdin:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        continue
    idx, mem, util = map(int, parts)
    if idx in gpu_list and (mem > max_mem or util > max_util):
        busy.append(f"gpu{idx}:mem={mem}MiB,util={util}%")
print(" ".join(busy))
' "${GPU_LIST}" "${GPU_FREE_MAX_MEM_MB}" "${GPU_FREE_MAX_UTIL}")
        if [ -z "${busy}" ]; then
            log "GPUs are available: ${CUDA_VISIBLE_DEVICES}"
            return 0
        fi
        log "waiting for GPUs ${CUDA_VISIBLE_DEVICES}: ${busy}"
        sleep 300
    done
}

tune_stage() {
    local name=$1
    local cfg=$2
    local result_file="${LOGDIR}/${name}.tune"
    rm -f "${result_file}"

    for combo in ${TUNE_COMBOS}; do
        local batch=${combo%:*}
        local accum=${combo#*:}
        local tune_work_dir="work_dirs_h200/tuning/${RUN_ID}/${name}_b${batch}a${accum}"
        local tune_log="${LOGDIR}/tune_${name}_b${batch}a${accum}.log"
        log "tuning ${name}: batch_per_gpu=${batch}, accum=${accum}"
        set +e
        WANDB_PROJECT="" WANDB_MODE=disabled HF_REPO_ID="" WORK_DIR="${tune_work_dir}" MASTER_PORT=$((MASTER_PORT_BASE + 1)) \
            bash tools/unimmv2x_h200_dist_train.sh "${cfg}" "${GPUS}" "${batch}" "${accum}" \
                --max-iters "${TUNE_MAX_ITERS}" \
                --workers-per-gpu "${TUNE_WORKERS}" \
                > "${tune_log}" 2>&1
        local rc=$?
        set -e
        if [ "${rc}" -eq 0 ] && ! has_error_pattern "${tune_log}"; then
            echo "${batch} ${accum}" > "${result_file}"
            log "selected ${name}: batch_per_gpu=${batch}, accum=${accum}"
            return 0
        fi
        log "rejected ${name}: batch_per_gpu=${batch}, accum=${accum}; see ${tune_log}"
    done

    log "no viable batch/accum combo found for ${name}"
    exit 1
}

run_stage() {
    local name=$1
    local cfg=$2
    local next_ckpt=${3:-}

    tune_stage "${name}" "${cfg}"
    read -r batch accum < "${LOGDIR}/${name}.tune"

    local work_dir="work_dirs_h200/spd_full/${RUN_ID}/${name}_b${batch}a${accum}"
    local train_log="${LOGDIR}/train_${name}_b${batch}a${accum}.log"
    export WORK_DIR="${work_dir}"
    export WANDB_NAME="${name}_spd_full_b${batch}a${accum}_${RUN_ID}"
    export HF_PATH_IN_REPO="spd_full/${RUN_ID}/${name}_b${batch}a${accum}"
    export MASTER_PORT=$((MASTER_PORT_BASE + 2))

    log "training ${name}: cfg=${cfg}, batch_per_gpu=${batch}, accum=${accum}, work_dir=${work_dir}"
    bash tools/unimmv2x_h200_dist_train.sh "${cfg}" "${GPUS}" "${batch}" "${accum}" \
        --workers-per-gpu "${TRAIN_WORKERS}" \
        2>&1 | tee "${train_log}"

    if [ -n "${next_ckpt}" ]; then
        if [ ! -s "${work_dir}/latest.pth" ]; then
            log "missing latest checkpoint after ${name}: ${work_dir}/latest.pth"
            exit 1
        fi
        mkdir -p "$(dirname "${next_ckpt}")"
        ln -sfn "$(realpath "${work_dir}/latest.pth")" "${next_ckpt}"
        log "linked ${next_ckpt} -> ${work_dir}/latest.pth"
    fi
}

log "queue started: RUN_ID=${RUN_ID}, initial CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, GPU_AUTO_SELECT=${GPU_AUTO_SELECT}, HF_REPO_ID=${HF_REPO_ID}"
wait_for_prepare
validate_infos
select_gpu_list
wait_for_gpus

run_stage "sub_inf_stg1" \
    "projects/configs_e2e_unimmv2x/unimmv2x_sub_inf_stg1.py" \
    "ckpts/unimmv2x_inf_stg1.pth"

run_stage "sub_vehicle_stg1" \
    "projects/configs_e2e_unimmv2x/unimmv2x_sub_vehicle_stg1.py" \
    "ckpts/unimmv2x_veh_stg1.pth"

run_stage "coop_stg1" \
    "projects/configs_e2e_unimmv2x/unimmv2x_coop_stg1.py" \
    "ckpts/unimmv2x_e2e_stg1.pth"

run_stage "coop_stg2" \
    "projects/configs_e2e_unimmv2x/unimmv2x_coop_stg2.py"

log "queue completed"
