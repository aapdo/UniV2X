_base_ = './univ2x_coop_e2e_adapter_latency_s2_accum8.py'

# Ablation: adapt camera image/neck features on the shifted infrastructure
# agent before BEV encoding.  This is the earliest feature-level physical
# adapter and keeps all non-adapter weights frozen.
physical_query_adapter = dict(
    enabled=False,
)

physical_image_adapter = dict(
    enabled=True,
    feature_dim=256,
    rank=8,
    metadata_dim=8,
    hidden_dim=64,
    residual_scale=1.0,
    init_std=1e-4,
    freeze_non_adapter=True,
    trainable_keys=['physical_image_adapter'],
)

model_ego_agent = dict(
    physical_query_adapter=physical_query_adapter,
)

model_other_agent_inf = dict(
    physical_image_adapter=physical_image_adapter,
)

# The image-feature adapter sits before the BEV encoder, so its trainable path
# retains substantially more downstream activation state than the query adapter.
# Mixed precision is required for batch-1 smoke/training on 24GB-class GPUs.
fp16 = dict(loss_scale=512.)
optimizer_config = dict(
    _delete_=True,
    type='GradientCumulativeFp16OptimizerHook',
    cumulative_iters=8,
    grad_clip=dict(max_norm=35, norm_type=2),
)
