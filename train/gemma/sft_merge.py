import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import logging
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)

tuner = Tuner("google/gemma-2-9b-it")
tuner.merge("gemma_sft", "adapter_gemma_sft")
