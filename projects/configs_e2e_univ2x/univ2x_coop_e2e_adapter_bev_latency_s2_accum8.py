_base_ = './univ2x_coop_e2e_adapter_latency_s2_accum8.py'

# Ablation: apply the same metadata-conditioned adapter to the cooperative BEV
# embedding after query-level cooperative evidence has been projected back into
# BEV.  This is a lightweight BEV feature adapter, not a full BEV encoder
# finetune.
physical_query_adapter = dict(
    enabled=True,
    position='bev_post_fusion',
    rank=8,
    metadata_dim=8,
    hidden_dim=64,
    residual_scale=1.0,
    init_std=1e-4,
    freeze_non_adapter=True,
    trainable_keys=['physical_query_adapter'],
)

model_ego_agent = dict(
    physical_query_adapter=physical_query_adapter,
)
