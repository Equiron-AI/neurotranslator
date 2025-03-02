#!/bin/bash

ollama rm equiron/yandex_translator || true
ollama create equiron/yandex_translator
ollama push equiron/yandex_translator
ollama pull equiron/yandex_translator