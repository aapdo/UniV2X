_base_ = './univ2x_coop_e2e.py'

physical_query_adapter = dict(
    enabled=True,
    rank=8,
    metadata_dim=8,
    hidden_dim=64,
    residual_scale=1.0,
    init_std=1e-4,
    freeze_non_adapter=True,
    trainable_keys=['physical_query_adapter'],
)

physical_shift = dict(
    enabled=True,
    name='infrastructure_latency',
    severity=2,
    seed=42,
    target_agents=['model_other_agent_inf'],
    metadata_mode='known',
)

model_ego_agent = dict(
    physical_query_adapter=physical_query_adapter,
)

data = dict(
    workers_per_gpu=4,
    train=dict(physical_shift=physical_shift),
    val=dict(physical_shift=physical_shift, eval_mod=[]),
    test=dict(physical_shift=physical_shift, eval_mod=[]),
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01,
)

total_epochs = 3
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')],
)

load_from = 'ckpts/univ2x_coop_e2e_stg2.pth'
resume_from = None
