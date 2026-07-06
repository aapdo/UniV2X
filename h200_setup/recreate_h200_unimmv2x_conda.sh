#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-unimmv2x_h200}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_EXE=${CONDA_EXE:-conda}

if ! command -v "${CONDA_EXE}" >/dev/null 2>&1; then
  echo "conda executable not found. Set CONDA_EXE=/path/to/conda" >&2
  exit 1
fi

# Create a clean conda env, then install the exact pip freeze captured from H200.
"${CONDA_EXE}" create -y -n "${ENV_NAME}" python=3.10 pip setuptools wheel git ninja cmake packaging
# shellcheck disable=SC1091
source "$("${CONDA_EXE}" info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html \
  -r "${REPO_ROOT}/h200_setup/requirements_h200_unimmv2x_freeze.txt"

export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
python "${REPO_ROOT}/h200_setup/smoke_h200_unimmv2x_env.py"

cat <<EOF
[OK] Recreated ${ENV_NAME}. For H200 training use:
  conda activate ${ENV_NAME}
  export TORCH_CUDA_ARCH_LIST=9.0
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
EOF
