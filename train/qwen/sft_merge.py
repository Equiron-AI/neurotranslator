import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import logging
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)

tuner = Tuner("Qwen/Qwen2.5-14B-Instruct")
tuner.merge("qwen_sft", "adapter_qwen_sft")
