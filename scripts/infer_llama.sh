#!/bin/sh

# If inferring with the llama model, set 'use_lora' to 'False'
# If inferring with the default alpaca model, set 'use_lora' to 'True', 'lora_weights' to 'tloen/alpaca-lora-7b'
# If inferring with the logllama model, download the LORA weights and set 'lora_weights' to './lora-logllama' (or the exact directory of LORA weights)


dataset="BGL"
python inference.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --use_lora False \
    --prompt_template 'logllama' \
    --log_name $dataset
