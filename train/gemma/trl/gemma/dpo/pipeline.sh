#!/bin/bash

python sft.py &> sft.log
python sft_merge.py
python dpo.py &> dpo.log
python dpo_merge.py
./convert.sh
