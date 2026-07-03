_base_ = './univ2x_coop_e2e_adapter_latency_s2.py'

physical_shift = dict(
    enabled=True,
    name='compound',
    severity=3,
    seed=42,
    target_agents=['model_other_agent_inf'],
    metadata_mode='known',
    shifts=[
        dict(name='infrastructure_latency', frame_delay=2),
        dict(name='relative_pose_noise', yaw_deg=2.0, translation_m=0.2),
        dict(name='fov_mask', keep_ratio=0.6),
        dict(name='missing_camera', drop_probability=1.0),
    ],
)

data = dict(
    train=dict(physical_shift=physical_shift),
    val=dict(physical_shift=physical_shift, eval_mod=[]),
    test=dict(physical_shift=physical_shift, eval_mod=[]),
)
