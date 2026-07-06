#!/usr/bin/env bash
set -euo pipefail
cd /home/gpu_01/UniMM-V2X
export CUDA_VISIBLE_DEVICES="0,1"
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export MASTER_PORT="60139"
export WANDB_PROJECT="unimmv2x-h200-spd"
export WANDB_NAME="unimmv2x_sub_inf_stg2_h200_g01_b1a4_r39_20260704_1658"
export HF_REPO_ID="apdoa/UniMM-V2X-H200-SPD"
export HF_PATH_IN_REPO="checkpoints/unimmv2x_sub_inf_stg2_h200_g01_b1a4_r39_20260704_1658"
export WORK_DIR="/home/gpu_01/UniMM-V2X/projects/work_dirs_h200_e2e_unimmv2x/unimmv2x_sub_inf_stg2"
export PATH="/home/gpu_01/venvs/unimmv2x_h200/bin:$PATH"
echo "[START] $(date -Is) run=unimmv2x_sub_inf_stg2_h200_g01_b1a4_r39_20260704_1658 host=$(hostname)"
echo "[PATCH] dynamic_sdc_query_idx; drop pretrained super kw; motion_head index device alignment"
exec tools/unimmv2x_h200_dist_train.sh "projects/configs_e2e_unimmv2x/unimmv2x_sub_inf_stg2.py" 2 1 4
