#!/bin/sh

# If inferring with the llama model, set 'use_lora' to 'False'
# If inferring with the default alpaca model, set 'use_lora' to 'True', 'lora_weights' to 'tloen/alpaca-lora-7b'
# If inferring with the logllama model, download the LORA weights and set 'lora_weights' to './lora-logllama' (or the exact directory of LORA weights)


#dataset="BGL"
#mode="single"
#python inference.py \
#    --base_model 'meta-llama/Llama-2-7b-chat-hf' \
#    --use_lora False \
#    --prompt_template 'single-1shot' \
#    --log_name "$dataset" \
#    --mode "$mode"

mode="single"
for dataset in HDFS Hadoop Spark Zookeeper HPC Thunderbird Windows Linux Android HealthApp Apache Proxifier OpenSSH OpenStack Mac
do
  python inference.py \
      --base_model 'meta-llama/Llama-2-7b-chat-hf' \
      --use_lora False \
      --prompt_template 'single-1shot' \
      --log_name "$dataset" \
      --mode "$mode"
done