#!/bin/bash

LLAMA_CPP=/mnt/llama.cpp
MODEL_NAME=yandex_sft

python3 ${LLAMA_CPP}/convert_hf_to_gguf.py ${MODEL_NAME} --outfile ${MODEL_NAME}.gguf --outtype f16
${LLAMA_CPP}/build/bin/llama-quantize ${MODEL_NAME}.gguf ${MODEL_NAME}_q5.gguf Q5_K_M
rm -f ${MODEL_NAME}.gguf