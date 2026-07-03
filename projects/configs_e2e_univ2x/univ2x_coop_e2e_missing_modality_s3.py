_base_ = './univ2x_coop_e2e.py'

physical_shift = dict(
    enabled=True,
    name='missing_modality',
    severity=3,
    seed=42,
    target_agents=['model_other_agent_inf'],
    metadata_mode='known',
    modalities=['camera', 'lidar'],
    camera_drop_probability=1.0,
    lidar_keep_ratio=0.10,
    ensure_one_view=False,
)

data = dict(
    train=dict(physical_shift=physical_shift),
    val=dict(physical_shift=physical_shift, eval_mod=[]),
    test=dict(physical_shift=physical_shift, eval_mod=[]),
)
