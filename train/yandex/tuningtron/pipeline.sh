#!/bin/bash

python sft.py
python sft_merge.py
./convert.sh
