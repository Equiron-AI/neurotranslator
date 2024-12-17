import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)
os.environ["HF_TOKEN"] = "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"

tuner = Tuner("google/gemma-2-9b-it", enable_deepspeed=True)
tuner.sft("equiron-ai/translator_sft", "adapter_gemma_sft", rank=64, batch_size=1, gradient_steps=1, learning_rate=1e-4)
