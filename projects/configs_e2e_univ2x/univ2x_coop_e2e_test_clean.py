_base_ = './univ2x_coop_e2e.py'

ann_file_test = 'data/infos/V2X-Seq-SPD-New/cooperative/spd_infos_temporal_test.pkl'

data = dict(
    test=dict(
        ann_file=ann_file_test,
        eval_mod=[],
    ),
)
