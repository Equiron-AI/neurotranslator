import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import logging
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)
os.environ["HF_TOKEN"] = "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"

tuner = Tuner("./gemma_sft", enable_deepspeed=False)
tuner.dpo("equiron-ai/translator_dpo", "adapter_gemma_dpo", rank=64, batch_size=1, gradient_steps=1, learning_rate=1e-4)
