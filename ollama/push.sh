#!/bin/bash

ollama create equiron/translator
ollama push equiron/translator

# Для того чтобы локально модель также обновилась
ollama stop equiron/translator
ollama pull equiron/translator