import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import logging
import sys
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)

tuner = Tuner("./aya_sft")
tuner.merge("aya_dpo", "adapter_aya_dpo")
