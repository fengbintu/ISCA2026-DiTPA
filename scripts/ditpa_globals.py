import os

### test mode selection: baseline or ditpa ###
action_skip_flag = os.getenv("ACTION_SKIP_FLAG", "False").lower() == "true"
iter_skip_flag = os.getenv("ITER_SKIP_FLAG", "False").lower() == "true"
modality_skip_flag = os.getenv("MODALITY_SKIP_FLAG", "False").lower() == "true"

test_label = os.getenv("TEST_LABEL", "baseline")

### parameters for evaluation ###
dit_steps = None
dit_iters = None
dit_last_skip_actions = None
model_setting = None
other_latency = 0
vlm_latency = 0
dit_latency = 0
dict_scale_param = None
