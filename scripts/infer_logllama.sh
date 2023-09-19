#!/bin/sh

# If inferring with the llama model, set 'use_lora' to 'False'
# If inferring with the default alpaca model, set 'use_lora' to 'True', 'lora_weights' to 'tloen/alpaca-lora-7b'
# If inferring with the logllama model, download the LORA weights and set 'lora_weights' to './lora-logllama' (or the exact directory of LORA weights)


dataset="BGL"
version="v1"
python inference.py \
    --base_model 'meta-llama/Llama-2-7b-chat-hf' \
    --use_lora True \
    --lora_weights './lora-logllama-'$version \
    --prompt_template 'log_parsing' \
    --log_name $dataset