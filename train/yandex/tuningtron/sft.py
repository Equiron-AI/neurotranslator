import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)
os.environ["HF_TOKEN"] = "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"

tuner = Tuner("yandex/YandexGPT-5-Lite-8B-pretrain", enable_deepspeed=False)

tuner.sft("equiron-ai/translator_tuningtron_sft_v2",
          "adapter_yandex_sft",
          lora_rank=128,
          learning_rate=1e-4,
          warmup_ratio=0.5)
