#!/usr/bin/env bash
set -euo pipefail
cd /home/gpu_01/UniMM-V2X
export CUDA_VISIBLE_DEVICES="2,3"
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export MASTER_PORT="60147"
export WANDB_PROJECT="unimmv2x-h200-spd"
export WANDB_NAME="unimmv2x_coop_stg2_h200_g23_b1a4_r46_20260708_1909"
export HF_REPO_ID="apdoa/UniMM-V2X-H200-SPD"
export HF_PATH_IN_REPO="checkpoints/unimmv2x_coop_stg2_h200_g23_b1a4_r46_20260708_1909"
export WORK_DIR="/home/gpu_01/UniMM-V2X/projects/work_dirs_h200_e2e_unimmv2x/unimmv2x_coop_stg2"
export PATH="/home/gpu_01/venvs/unimmv2x_h200/bin:$PATH"
echo "[START] $(date -Is) run=unimmv2x_coop_stg2_h200_g23_b1a4_r46_20260708_1909 host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] resume from epoch_1 after quota cleanup and h200_train resume support e8ea3ee"
exec tools/unimmv2x_h200_dist_train.sh "projects/configs_e2e_unimmv2x/unimmv2x_coop_stg2.py" 2 1 4 --base-global-batch 8 --checkpoint-keep-epochs 10 --resume-from "/home/gpu_01/UniMM-V2X/projects/work_dirs_h200_e2e_unimmv2x/unimmv2x_coop_stg2/latest.pth" --hf-path-in-repo "checkpoints/unimmv2x_coop_stg2_h200_g23_b1a4_r46_20260708_1909"
