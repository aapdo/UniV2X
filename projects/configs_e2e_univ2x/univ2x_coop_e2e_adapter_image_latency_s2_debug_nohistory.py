_base_ = './univ2x_coop_e2e_adapter_image_latency_s2_debug.py'

# Smoke-test-only memory reduction.  This disables temporal history BEV
# construction so we can verify image-adapter wiring on 32GB GPUs.  Do not use
# this config as the final comparable UniV2X training/evaluation setting.
model_other_agent_inf = dict(skip_history_bev=True)
model_ego_agent = dict(skip_history_bev=True)
