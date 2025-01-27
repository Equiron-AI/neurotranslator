import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
import sys
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)

tuner = Tuner("./gemma_sft")
tuner.merge("gemma_dpo", "adapter_gemma_dpo")
