#!/bin/bash

# export OLLAMA_HOST=0.0.0.0:8088
export CUDA_VISIBLE_DEVICES=1
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_QUEUE=32
export OLLAMA_FLASH_ATTENTION=1

ollama serve
