import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)
os.environ["HF_TOKEN"] = "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"

tuner = Tuner("Qwen/Qwen2.5-14B-Instruct", enable_deepspeed=True)
tuner.sft("equiron-ai/translator_sft", "adapter_qwen_sft", rank=32, batch_size=1, gradient_steps=1, learning_rate=1e-5)
