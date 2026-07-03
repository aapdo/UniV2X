_base_ = './univ2x_coop_e2e_adapter_image_latency_s2_accum8.py'

data = dict(
    workers_per_gpu=2,
    train=dict(
        is_debug=True,
        len_debug=8,
    ),
    val=dict(
        is_debug=True,
        len_debug=8,
    ),
    test=dict(
        is_debug=True,
        len_debug=8,
    ),
)

total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=999)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')],
)
