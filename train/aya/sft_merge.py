import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import logging
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)

tuner = Tuner("CohereForAI/aya-expanse-8b")
tuner.merge("aya_sft", "adapter_aya_sft")
