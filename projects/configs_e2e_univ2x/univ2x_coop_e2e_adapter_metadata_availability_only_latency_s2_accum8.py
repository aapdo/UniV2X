_base_ = './univ2x_coop_e2e_adapter_latency_s2_accum8.py'

physical_query_adapter = dict(
    enabled=True,
    rank=8,
    metadata_dim=8,
    hidden_dim=64,
    residual_scale=1.0,
    init_std=1e-4,
    freeze_non_adapter=True,
    trainable_keys=['physical_query_adapter'],
    metadata_ablation=dict(mode='availability_only'),
)

model_ego_agent = dict(
    physical_query_adapter=physical_query_adapter,
)
