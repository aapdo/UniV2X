_base_ = './univ2x_coop_e2e_adapter_latency_s2_accum8.py'

# Ablation: train the physical query adapter plus the lightweight linear
# layers that perform cross-agent query/BEV fusion.  This keeps the image
# backbone, BEV encoder, heads, and most UniV2X weights frozen while allowing
# the fusion interface itself to adapt to known latency metadata.
physical_query_adapter = dict(
    enabled=True,
    rank=8,
    metadata_dim=8,
    hidden_dim=64,
    residual_scale=1.0,
    init_std=1e-4,
    freeze_non_adapter=True,
    trainable_keys=[
        'physical_query_adapter',
        'cross_agent_query_interaction.cross_agent_align',
        'cross_agent_query_interaction.cross_agent_align_pos',
        'cross_agent_query_interaction.cross_agent_fusion',
        'bev_embed_linear',
        'bev_pos_linear',
    ],
)

model_ego_agent = dict(
    physical_query_adapter=physical_query_adapter,
)
