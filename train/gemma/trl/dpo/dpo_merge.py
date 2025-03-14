import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_TOKEN"] = "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"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_path = "./gemma_sft"
merged_name = "gemma_dpo"
base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
peft_model = PeftModel.from_pretrained(base_model, "./adapter_gemma_dpo", torch_dtype=torch.bfloat16)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(merged_name)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(merged_name)

try:
    tokenizer.save_vocabulary(merged_name)
except:
    pass