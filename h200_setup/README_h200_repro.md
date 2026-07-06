# H200 UniMM-V2X Reproducibility Notes

This directory captures the H200 environment used for the SPD sequential training port.

Current production run captured here:

- Stage: `sub_vehicle_stg2`
- Run: `unimmv2x_sub_vehicle_stg2_h200_g01_b1a4_r43_20260706_1517`
- Python env used by the running process: `/home/gpu_01/venvs/unimmv2x_h200`
- That venv is based on: `/home/gpu_01/miniconda3/envs/mrlora/bin/python3.10`
- Runtime CUDA arch: `TORCH_CUDA_ARCH_LIST=9.0`
- PyTorch CUDA: `2.1.2+cu121`

## Files

- `environment_h200_unimmv2x.yml`: conda environment skeleton that installs the captured pip freeze.
- `requirements_h200_unimmv2x_freeze.txt`: exact pip package freeze with cu121/OpenMMLab wheel locations.
- `requirements_h200_unimmv2x_core.txt`: human-readable core package pins.
- `conda_mrlora_explicit.txt`: explicit package list for the conda env that supplied the base Python.
- `conda_mrlora_env_no_builds.yml`: no-builds export of the base conda env.
- `recreate_h200_unimmv2x_conda.sh`: scripted recreation into a clean conda env.
- `smoke_h200_unimmv2x_env.py`: import/CUDA/mmcv extension smoke test.
- `run_records/`: launch scripts and `current_train.env` snapshots for active H200 runs.

## Recreate

From the repository root on a CUDA-capable H200 host:

```bash
bash h200_setup/recreate_h200_unimmv2x_conda.sh unimmv2x_h200
conda activate unimmv2x_h200
export TORCH_CUDA_ARCH_LIST=9.0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python h200_setup/smoke_h200_unimmv2x_env.py
```

If `mmcv._ext` fails on H200, rebuild MMCV ops for SM90:

```bash
conda activate unimmv2x_h200
TORCH_CUDA_ARCH_LIST=9.0 bash h200_setup/rebuild_mmcv_sm90.sh
```

Checkpoints, work dirs, W&B logs, datasets, and cache directories are intentionally excluded from git and backup tarballs.
