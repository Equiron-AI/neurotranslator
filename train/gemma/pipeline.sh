#!/bin/bash

# python sft.py
# python sft_merge.py
python dpo.py
python dpo_merge.py
./convert.sh
