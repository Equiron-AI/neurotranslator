#!/bin/bash

set -e

export LLAMA_CPP=/mnt/llama.cpp

model_name=qwen_dpo

python3 ${LLAMA_CPP}/convert_hf_to_gguf.py ${model_name} --outfile ${model_name}.gguf --outtype f16

${LLAMA_CPP}/build/bin/llama-quantize ${model_name}.gguf ${model_name}_q5.gguf Q5_K_M

rm -f ${model_name}.gguf