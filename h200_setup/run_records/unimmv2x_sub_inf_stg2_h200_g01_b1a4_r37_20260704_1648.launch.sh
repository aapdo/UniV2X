#!/usr/bin/env bash
set -euo pipefail
cd /home/gpu_01/UniMM-V2X
export CUDA_VISIBLE_DEVICES=0\,1
export TORCH_CUDA_ARCH_LIST=9.0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export MASTER_PORT=60137
export WANDB_PROJECT=unimmv2x-h200-spd
export WANDB_NAME=unimmv2x_sub_inf_stg2_h200_g01_b1a4_r37_20260704_1648
export HF_REPO_ID=apdoa/UniMM-V2X-H200-SPD
export PATH=/home/gpu_01/venvs/unimmv2x_h200/bin:$PATH
echo \[START\]\ 2026-07-04T16:47:32+09:00\ run=unimmv2x_sub_inf_stg2_h200_g01_b1a4_r37_20260704_1648\ host=gpusystem
echo \[PATCH\]\ track_loss+unimmv2x_track\ dynamic_sdc_query_idx
exec tools/unimmv2x_h200_dist_train.sh projects/configs_e2e_unimmv2x/unimmv2x_sub_inf_stg2.py 2 1 4
