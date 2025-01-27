import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_TOKEN"] = "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"

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig


model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", torch_dtype=torch.bfloat16, device_map="auto")
print(model)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", add_bos_token=False)

peft_config = LoraConfig(r=128, lora_alpha=128, lora_dropout=0.1, target_modules="all-linear", task_type="CAUSAL_LM")

dataset = load_dataset("equiron-ai/translator_sft_v2", split="train")
inputs = [tokenizer.apply_chat_template(row) for row in dataset["messages"]]
max_len = max([len(x) for x in inputs])

print("Row max len detected:", max_len)
print("Example dataset row: ")
print(tokenizer.apply_chat_template(dataset["messages"][0], tokenize=False))

config = SFTConfig("./",
                   num_train_epochs=1,
                   logging_steps=1,
                   gradient_checkpointing=True,
                   save_strategy="no",
                   optim="adamw_8bit",
                   weight_decay=0.001,
                   learning_rate=1e-4,
                   warmup_ratio=0.1,
                   per_device_train_batch_size=1,
                   bf16=True,
                   max_seq_length=max_len)

print(config)

trainer = SFTTrainer(model=model,
                     peft_config=peft_config,
                     train_dataset=dataset,
                     data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
                     args=config)
trainer.train()
trainer.save_model("adapter_gemma_sft")
