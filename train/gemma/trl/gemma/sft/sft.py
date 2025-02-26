import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_TOKEN"] = "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"

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model


model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="auto")
model.gradient_checkpointing_enable()

peft_config = LoraConfig(r=128,
                         lora_alpha=128,
                         target_modules="all-linear",
                         task_type="CAUSAL_LM")

peft_model = get_peft_model(model, peft_config)
print(peft_model.get_model_status())

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

dataset = load_dataset("equiron-ai/translator_sft_v4", split="train")
inputs = [tokenizer.apply_chat_template(row) for row in dataset["messages"]]
max_len = max([len(x) for x in inputs])

print("Row max len detected:", max_len)
print("Example dataset row: ")
print(tokenizer.apply_chat_template(dataset["messages"][0], tokenize=False))
print("---")
print(tokenizer.apply_chat_template(dataset["messages"][0]))

config = SFTConfig("./",
                   num_train_epochs=1,
                   logging_steps=1,
                   gradient_checkpointing=True,
                   save_strategy="no",
                   optim="adamw_8bit",
                   learning_rate=5e-5,
                   warmup_ratio=0.1,
                   per_device_train_batch_size=1,
                   gradient_accumulation_steps=1,
                   bf16=True,
                   max_seq_length=max_len)

trainer = SFTTrainer(model=peft_model,
                     train_dataset=dataset,
                     data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
                     args=config)
trainer.train()
trainer.save_model("adapter_gemma_sft")
