#!/usr/bin/env bash
set -euo pipefail

CFG=${1:?config path is required}
GPUS=${2:-2}
BATCH_PER_GPU=${3:-1}
ACCUM=${4:-1}

MASTER_PORT=${MASTER_PORT-29596}
WANDB_PROJECT=${WANDB_PROJECT-unimmv2x-h200-spd}
WANDB_NAME=${WANDB_NAME-$(basename "${CFG%.*}")_b${BATCH_PER_GPU}a${ACCUM}_$(date +%Y%m%d_%H%M%S)}
HF_REPO_ID=${HF_REPO_ID-apdoa/UniMM-V2X-H200-SPD}
if [ -z "${WORK_DIR:-}" ]; then
    WORK_DIR="$(echo "${CFG%.*}" | sed -e "s#configs#work_dirs_h200#g")/"
fi

mkdir -p "${WORK_DIR}/logs"
T=$(date +%m%d%H%M)

echo "[h200_dist_train] cfg=${CFG}"
echo "[h200_dist_train] gpus=${GPUS} batch_per_gpu=${BATCH_PER_GPU} accum=${ACCUM}"
echo "[h200_dist_train] work_dir=${WORK_DIR}"
echo "[h200_dist_train] wandb=${WANDB_PROJECT}/${WANDB_NAME}"
echo "[h200_dist_train] hf_repo=${HF_REPO_ID}"

PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
python -m torch.distributed.run \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    "$(dirname "$0")/h200_train.py" \
    "${CFG}" \
    --work-dir "${WORK_DIR}" \
    --batch-per-gpu "${BATCH_PER_GPU}" \
    --accum "${ACCUM}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "${WANDB_NAME}" \
    --hf-repo-id "${HF_REPO_ID}" \
    "${@:5}" \
    2>&1 | tee "${WORK_DIR}/logs/train.${T}"
