import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_TOKEN"] = "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"

import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig


model = AutoModelForCausalLM.from_pretrained("./gemma_sft", torch_dtype=torch.bfloat16, device_map="auto")
print(model)

tokenizer = AutoTokenizer.from_pretrained("./gemma_sft", add_bos_token=False)

peft_config = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1, target_modules="all-linear", task_type="CAUSAL_LM")

dataset = load_dataset("equiron-ai/translator_dpo", split="train")

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
                   weight_decay=0.001,
                   learning_rate=1e-5,
                   warmup_ratio=0.1,
                   per_device_train_batch_size=1,
                   bf16=True)

print(config)

trainer = DPOTrainer(model=model,
                     peft_config=peft_config,
                     train_dataset=dataset,
                     processing_class=tokenizer,
                     args=config)

trainer.train()
trainer.save_model("adapter_gemma_dpo")
