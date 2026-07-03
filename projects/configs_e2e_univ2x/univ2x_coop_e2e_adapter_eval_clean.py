_base_ = './univ2x_coop_e2e_adapter_latency_s2.py'

physical_shift = dict(enabled=False)

data = dict(
    train=dict(physical_shift=physical_shift),
    val=dict(physical_shift=physical_shift, eval_mod=[]),
    test=dict(physical_shift=physical_shift, eval_mod=[]),
)
