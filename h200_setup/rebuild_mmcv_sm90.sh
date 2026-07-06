#!/usr/bin/env bash
set -euo pipefail

# Works inside either the captured venv or a recreated conda env.
if [ -z "${CONDA_PREFIX:-}" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -d /home/gpu_01/venvs/unimmv2x_h200 ]; then
  # shellcheck disable=SC1091
  source /home/gpu_01/venvs/unimmv2x_h200/bin/activate
fi

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MMCV_WITH_OPS=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${MAX_JOBS:-24}"

python -m pip install --no-cache-dir Cython
python -m pip install --no-cache-dir --no-deps --no-build-isolation --no-binary=mmcv --force-reinstall -v mmcv==2.1.0
python - <<"VERIFY"
import torch, mmcv
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device capability", torch.cuda.get_device_capability(0))
print("mmcv", mmcv.__version__, mmcv.__file__)
import mmcv._ext as ext
print("mmcv._ext", ext.__file__)
VERIFY
