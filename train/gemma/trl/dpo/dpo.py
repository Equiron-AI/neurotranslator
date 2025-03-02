import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_TOKEN"] = "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"

import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_path = "./gemma_sft"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="auto")
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(model_path)

peft_config = LoraConfig(r=128, lora_alpha=128, target_modules="all-linear", task_type="CAUSAL_LM")
peft_model = get_peft_model(model, peft_config)
print(peft_model.get_model_status())

dataset = load_dataset("equiron-ai/translator_dpo_v4", split="train")

print("Example choosen dataset row: ")
print(tokenizer.apply_chat_template(dataset["chosen"][0], tokenize=False))

print("Example rejected dataset row: ")
print(tokenizer.apply_chat_template(dataset["rejected"][0], tokenize=False))

config = DPOConfig("./",
                   num_train_epochs=1,
                   logging_steps=1,
                   gradient_checkpointing=True,
                   save_strategy="no",
                   optim="adamw_8bit",
                   learning_rate=1e-5,
                   warmup_ratio=0.1,
                   per_device_train_batch_size=1,
                   gradient_accumulation_steps=1,
                   bf16=True)

trainer = DPOTrainer(model=peft_model,
                     train_dataset=dataset,
                     processing_class=tokenizer,
                     args=config)

trainer.train()
trainer.save_model("adapter_gemma_dpo")
