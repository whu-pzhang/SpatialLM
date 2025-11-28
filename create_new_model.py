from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM  # noqa
from spatiallm import Layout  # noqa

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

model_name_or_path = "manycore-research/SpatialLM1.1-Qwen-0.5B"
qwen1_5b = "Qwen/Qwen2.5-1.5B-Instruct"
new_model_path = "weights/SpatialLM1.1-Qwen-1.5B"

config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
# print(config)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, trust_remote_code=True, low_cpu_mem_usage=True
)
# print(model.model)

qwen = AutoModelForCausalLM.from_pretrained(
    qwen1_5b, trust_remote_code=True, low_cpu_mem_usage=True
)
# print(qwen)

new_model_cfg = AutoConfig.from_pretrained(new_model_path, trust_remote_code=True)
new_model = AutoModelForCausalLM.from_config(new_model_cfg, trust_remote_code=True)
# load weights
new_model.model.load_state_dict(qwen.model.state_dict(), strict=True)
new_model.lm_head.load_state_dict(qwen.lm_head.state_dict(), strict=True)
new_model.point_backbone.load_state_dict(model.point_backbone.state_dict(), strict=True)

# save
new_model.save_pretrained(new_model_path, safe_serialization=True)

# model_name_or_path = "weights/SpatialLM1.1-Qwen-1.5B"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path, trust_remote_code=True, low_cpu_mem_usage=True
# )
