import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import logging
import sys
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)

tuner = Tuner("./qwen_sft")
tuner.merge("qwen_dpo", "adapter_qwen_dpo")
