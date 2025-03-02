#!/bin/bash

ollama rm equiron/yandex_translator || true
ollama create equiron/yandex_translator
ollama run equiron/yandex_translator