import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


merged_name = "gemma_sft"
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", torch_dtype=torch.bfloat16, device_map="auto")
peft_model = PeftModel.from_pretrained(base_model, "./adapter_gemma_sft", torch_dtype=torch.bfloat16)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(merged_name)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
tokenizer.save_pretrained(merged_name)

try:
    tokenizer.save_vocabulary(merged_name)
except:
    pass