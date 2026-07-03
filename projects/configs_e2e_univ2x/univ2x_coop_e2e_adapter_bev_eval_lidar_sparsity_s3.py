_base_ = './univ2x_coop_e2e_adapter_bev_latency_s2_accum8.py'

physical_shift = dict(
    enabled=True,
    name='lidar_sparsity',
    severity=3,
    seed=42,
    target_agents=['model_other_agent_inf'],
    metadata_mode='known',
)

data = dict(
    train=dict(physical_shift=physical_shift),
    val=dict(physical_shift=physical_shift, eval_mod=[]),
    test=dict(physical_shift=physical_shift, eval_mod=[]),
)
