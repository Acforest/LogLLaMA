#!/bin/bash

version="v1"
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path './data/logllama_data.json' \
    --output_dir './lora-logllama-'$version \
    --prompt_template_name 'log_parsing' \
    --micro_batch_size 64 \
    --batch_size 64 \
    --wandb_run_name $version