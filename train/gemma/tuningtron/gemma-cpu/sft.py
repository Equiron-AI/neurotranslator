import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import logging
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)
os.environ["HF_TOKEN"] = "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"

tuner = Tuner("google/gemma-2-9b-it", enable_cpu=True)
tuner.sft("equiron-ai/translator_sft", "adapter_gemma_sft", rank=128, learning_rate=1e-4)
