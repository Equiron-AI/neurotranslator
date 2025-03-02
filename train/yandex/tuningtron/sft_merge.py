import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)

tuner = Tuner("yandex/YandexGPT-5-Lite-8B-pretrain")
tuner.merge("yandex_sft", "adapter_yandex_sft")
