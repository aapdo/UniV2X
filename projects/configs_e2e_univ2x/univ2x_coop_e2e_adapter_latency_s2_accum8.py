_base_ = './univ2x_coop_e2e_adapter_latency_s2.py'

# The MultiAgent wrapper supports data.samples_per_gpu > 1 by slicing each
# dataloader batch into UniV2X-compatible single-sample micro-batches.  This
# keeps the recurrent tracker path unchanged while allowing per-GPU batch
# changes through cfg-options, e.g.:
#
#   data.samples_per_gpu=2 data.test.samples_per_gpu=4
#
# Keep cumulative_iters consistent with the desired optimizer batch. This
# default matches effective global batch 16 on two GPUs:
# 2 GPUs * 1 sample/GPU * 8 accumulation steps = 16.
fp16 = dict(loss_scale=512.)
optimizer_config = dict(
    _delete_=True,
    type='GradientCumulativeFp16OptimizerHook',
    cumulative_iters=8,
    grad_clip=dict(max_norm=35, norm_type=2),
)
