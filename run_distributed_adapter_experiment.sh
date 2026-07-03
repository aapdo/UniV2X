#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ROOT=/home/jy/adas/e2e/baselines/UniV2X
if [ ! -d "${DEFAULT_ROOT}" ] && [ -d /data/adas/e2e/baselines/UniV2X ]; then
  DEFAULT_ROOT=/data/adas/e2e/baselines/UniV2X
fi
ROOT=${ROOT:-${DEFAULT_ROOT}}
E2E_ROOT=${E2E_ROOT:-$(cd "${ROOT}/../.." && pwd)}
PY=${PY:-${E2E_ROOT}/envs/univ2x/bin/python}
LOGDIR=${LOGDIR:-${E2E_ROOT}/logs}

if [ -z "${EXP_NAME:-}" ]; then
  echo "EXP_NAME is required" >&2
  exit 2
fi
if [ -z "${CONFIG:-}" ]; then
  echo "CONFIG is required" >&2
  exit 2
fi

VISIBLE_DEVICES=${VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-1,2}}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
TRAIN_SAMPLES_PER_GPU=${TRAIN_SAMPLES_PER_GPU:-1}
EVAL_SAMPLES_PER_GPU=${EVAL_SAMPLES_PER_GPU:-1}
CUMULATIVE_ITERS=${CUMULATIVE_ITERS:-8}
MASTER_PORT=${MASTER_PORT:-29981}
EVAL_MASTER_PORT_BASE=${EVAL_MASTER_PORT_BASE:-29991}
FORCE_RETRAIN=${FORCE_RETRAIN:-0}

WORKDIR=${WORKDIR:-./projects/work_dirs_e2e_univ2x/${EXP_NAME}}
CKPT=${CKPT:-${WORKDIR}/latest.pth}
STATUS_LOG=${STATUS_LOG:-${LOGDIR}/${EXP_NAME}.status.log}
TRAIN_LOG=${TRAIN_LOG:-${LOGDIR}/${EXP_NAME}.train.latest.log}
EVAL_DIR=${EVAL_DIR:-./projects/work_dirs_e2e_univ2x/${EXP_NAME}_eval}
OUTPUT_DIR=${OUTPUT_DIR:-${WORKDIR}/output}
EVAL_CONFIG_DIR=${EVAL_CONFIG_DIR:-${EVAL_DIR}/configs}
ENABLE_WANDB=${ENABLE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-adas-e2e-physical-shift}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_TAGS=${WANDB_TAGS:-univ2x,physical-shift,adapter}

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
export PATH="$(dirname "${PY}"):${PATH}"
export PYTHONPATH="${ROOT}:${ROOT}/projects:${E2E_ROOT}/baselines/mmdetection3d"
if [ -n "${WANDB_MODE}" ]; then
  export WANDB_MODE
fi

mkdir -p "${LOGDIR}" "${OUTPUT_DIR}" "${WORKDIR}" "${EVAL_DIR}" "${EVAL_CONFIG_DIR}"
WORKDIR_ABS=$(cd "${WORKDIR}" && pwd)
export WANDB_DIR="${WANDB_DIR:-${WORKDIR_ABS}/wandb}"
mkdir -p "${WANDB_DIR}"

CONFIG_ABS=$(readlink -f "${CONFIG}")
TRAIN_CONFIG="${CONFIG_ABS}"

write_wandb_config() {
  local cfg="$1"
  local tags_expr
  local entity_expr

  tags_expr=$(python3 - <<PY
print([tag.strip() for tag in """${WANDB_TAGS}""".split(',') if tag.strip()])
PY
)
  if [ -n "${WANDB_ENTITY}" ]; then
    entity_expr=", entity='${WANDB_ENTITY}'"
  else
    entity_expr=""
  fi

  cat > "${cfg}" <<PY
_base_ = '${CONFIG_ABS}'

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='${WANDB_PROJECT}',
                name='${EXP_NAME}'${entity_expr},
                tags=${tags_expr},
                config=dict(
                    exp_name='${EXP_NAME}',
                    config='${CONFIG_ABS}',
                    visible_devices='${VISIBLE_DEVICES}',
                    nproc_per_node=${NPROC_PER_NODE},
                    train_samples_per_gpu=${TRAIN_SAMPLES_PER_GPU},
                    cumulative_iters=${CUMULATIVE_ITERS},
                    eval_samples_per_gpu=${EVAL_SAMPLES_PER_GPU},
                ),
            ),
        ),
    ],
)
PY
}

if [ "${ENABLE_WANDB}" = "1" ] && [ -n "${WANDB_PROJECT}" ]; then
  TRAIN_CONFIG="${WORKDIR}/${EXP_NAME}_train_wandb.py"
  write_wandb_config "${TRAIN_CONFIG}"
fi

write_eval_config() {
  local setting="$1"
  local cfg="$2"

  case "${setting}" in
    clean)
      cat > "${cfg}" <<PY
_base_ = '${CONFIG_ABS}'

physical_shift = dict(enabled=False)

data = dict(
    train=dict(physical_shift=physical_shift),
    val=dict(physical_shift=physical_shift, eval_mod=[]),
    test=dict(physical_shift=physical_shift, eval_mod=[]),
)
PY
      ;;
    latency_s2)
      cat > "${cfg}" <<PY
_base_ = '${CONFIG_ABS}'

physical_shift = dict(
    enabled=True,
    name='infrastructure_latency',
    severity=2,
    seed=42,
    target_agents=['model_other_agent_inf'],
    metadata_mode='known',
)

data = dict(
    train=dict(physical_shift=physical_shift),
    val=dict(physical_shift=physical_shift, eval_mod=[]),
    test=dict(physical_shift=physical_shift, eval_mod=[]),
)
PY
      ;;
    pose_s3)
      cat > "${cfg}" <<PY
_base_ = '${CONFIG_ABS}'

physical_shift = dict(
    enabled=True,
    name='relative_pose_noise',
    severity=3,
    seed=42,
    target_agents=['model_other_agent_inf'],
    metadata_mode='known',
)

data = dict(
    train=dict(physical_shift=physical_shift),
    val=dict(physical_shift=physical_shift, eval_mod=[]),
    test=dict(physical_shift=physical_shift, eval_mod=[]),
)
PY
      ;;
    calibration_s3)
      cat > "${cfg}" <<PY
_base_ = '${CONFIG_ABS}'

physical_shift = dict(
    enabled=True,
    name='calibration_drift',
    severity=3,
    seed=42,
    target_agents=['model_other_agent_inf'],
    metadata_mode='known',
)

data = dict(
    train=dict(physical_shift=physical_shift),
    val=dict(physical_shift=physical_shift, eval_mod=[]),
    test=dict(physical_shift=physical_shift, eval_mod=[]),
)
PY
      ;;
    fov_s3)
      cat > "${cfg}" <<PY
_base_ = '${CONFIG_ABS}'

physical_shift = dict(
    enabled=True,
    name='fov_mask',
    severity=3,
    seed=42,
    target_agents=['model_other_agent_inf'],
    metadata_mode='known',
)

data = dict(
    train=dict(physical_shift=physical_shift),
    val=dict(physical_shift=physical_shift, eval_mod=[]),
    test=dict(physical_shift=physical_shift, eval_mod=[]),
)
PY
      ;;
    lidar_sparsity_s3)
      cat > "${cfg}" <<PY
_base_ = '${CONFIG_ABS}'

physical_shift = dict(
    enabled=True,
    name='lidar_sparsity',
    severity=3,
    seed=42,
    target_agents=['model_other_agent_inf'],
    metadata_mode='known',
)

data = dict(
    train=dict(physical_shift=physical_shift),
    val=dict(physical_shift=physical_shift, eval_mod=[]),
    test=dict(physical_shift=physical_shift, eval_mod=[]),
)
PY
      ;;
    missing_camera_s3)
      cat > "${cfg}" <<PY
_base_ = '${CONFIG_ABS}'

physical_shift = dict(
    enabled=True,
    name='missing_camera',
    severity=3,
    seed=42,
    target_agents=['model_other_agent_inf'],
    metadata_mode='known',
)

data = dict(
    train=dict(physical_shift=physical_shift),
    val=dict(physical_shift=physical_shift, eval_mod=[]),
    test=dict(physical_shift=physical_shift, eval_mod=[]),
)
PY
      ;;
    compound_s3)
      cat > "${cfg}" <<PY
_base_ = '${CONFIG_ABS}'

physical_shift = dict(
    enabled=True,
    name='compound',
    severity=3,
    seed=42,
    target_agents=['model_other_agent_inf'],
    metadata_mode='known',
    shifts=[
        dict(name='infrastructure_latency', frame_delay=2),
        dict(name='relative_pose_noise', yaw_deg=2.0, translation_m=0.2),
        dict(name='fov_mask', keep_ratio=0.6),
        dict(name='missing_camera', drop_probability=1.0),
    ],
)

data = dict(
    train=dict(physical_shift=physical_shift),
    val=dict(physical_shift=physical_shift, eval_mod=[]),
    test=dict(physical_shift=physical_shift, eval_mod=[]),
)
PY
      ;;
    *)
      echo "unknown eval setting: ${setting}" >&2
      exit 2
      ;;
  esac
}

echo "$(date) start ${EXP_NAME}" >> "${STATUS_LOG}"
echo "$(date) visible_devices=${VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} train_samples_per_gpu=${TRAIN_SAMPLES_PER_GPU} cumulative_iters=${CUMULATIVE_ITERS} eval_samples_per_gpu=${EVAL_SAMPLES_PER_GPU} wandb_project=${WANDB_PROJECT:-none} wandb_mode=${WANDB_MODE:-default}" >> "${STATUS_LOG}"

if [ "${FORCE_RETRAIN}" = "1" ] || [ ! -f "${CKPT}" ]; then
  echo "$(date) start ${EXP_NAME} training" >> "${STATUS_LOG}"
  train_cfg_options=(
    dist_params.backend=gloo
    data.samples_per_gpu="${TRAIN_SAMPLES_PER_GPU}"
  )
  if [ "${CUMULATIVE_ITERS}" != "0" ]; then
    train_cfg_options+=(
      optimizer_config.cumulative_iters="${CUMULATIVE_ITERS}"
    )
  fi
  "${PY}" -m torch.distributed.launch \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    ./tools/train.py \
    "${TRAIN_CONFIG}" \
    --launcher pytorch \
    --work-dir "${WORKDIR}" \
    --no-validate \
    --cfg-options "${train_cfg_options[@]}" \
    > "${TRAIN_LOG}" 2>&1
  echo "$(date) done ${EXP_NAME} training" >> "${STATUS_LOG}"
else
  echo "$(date) skip ${EXP_NAME} training because checkpoint exists: ${CKPT}" >> "${STATUS_LOG}"
fi

if [ ! -f "${CKPT}" ]; then
  echo "$(date) missing checkpoint: ${CKPT}" >> "${STATUS_LOG}"
  exit 1
fi

settings=(
  clean
  latency_s2
  pose_s3
  calibration_s3
  fov_s3
  lidar_sparsity_s3
  missing_camera_s3
  compound_s3
)

port="${EVAL_MASTER_PORT_BASE}"
for setting in "${settings[@]}"; do
  cfg="${EVAL_CONFIG_DIR}/${EXP_NAME}_eval_${setting}.py"
  write_eval_config "${setting}" "${cfg}"
  out="${OUTPUT_DIR}/results_${EXP_NAME}_${setting}.pkl"
  log="${LOGDIR}/${EXP_NAME}.eval_${setting}.latest.log"
  json_prefix="${EVAL_DIR}/nusc_${setting}"
  echo "$(date) start ${EXP_NAME} eval ${setting}" >> "${STATUS_LOG}"
  "${PY}" -m torch.distributed.launch \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${port}" \
    ./tools/test.py \
    "${cfg}" \
    "${CKPT}" \
    --launcher pytorch \
    --out "${out}" \
    --cfg-options \
      dist_params.backend=gloo \
      data.test.samples_per_gpu="${EVAL_SAMPLES_PER_GPU}" \
    --eval bbox \
    --eval-options \
      jsonfile_prefix="${json_prefix}" \
    --show-dir "${EVAL_DIR}/" \
    > "${log}" 2>&1
  echo "$(date) done ${EXP_NAME} eval ${setting}" >> "${STATUS_LOG}"
  port=$((port + 1))
done

echo "$(date) all ${EXP_NAME} train/eval done" >> "${STATUS_LOG}"
