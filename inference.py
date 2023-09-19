import os
import re
import sys
from collections import Counter

import pandas as pd
from tqdm import tqdm

import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from benchmark import log_2k_info, log_to_dataframe
from evaluation.utils.common import correct_single_template
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "decapoda-research/llama-7b-hf",  # meta-llama/Llama-2-7b-chat-hf for LLaMA2
    use_lora: bool = True,
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "logllama",  # The prompt template to use, will default to alpaca.
    log_name: str = "HDFS",
    mode: str = "single"  # "single" or "group"
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if use_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction=None,
        input=None,
        temperature=0.75,
        top_p=0.9,
        top_k=40,
        num_beams=4,
        max_new_tokens=500,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)

    # Testing
    input_path = f'./data/log_parsing/{log_name}'
    output_path = f"./outputs/{log_name}"
    log_json_path = os.path.join(input_path, f'{log_name}_2k.json')

    df = pd.read_json(log_json_path, orient="records")
    # df.insert(loc=3, column="response", value='')
    df.rename(columns={"output": "groundtruth"}, inplace=True)

    os.makedirs(output_path, exist_ok=True)
    resp_list = []
    if mode == 'single':
        with open(os.path.join(output_path, f'{log_name}_response.txt'), 'w', encoding='utf-8') as f:
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Infering'):
                resp = evaluate(input=row["input"])
                f.write(resp + '\n')
                resp_list.append(resp)
    elif mode == 'group':
        input_list = []
        batch_size = 10
        with open(os.path.join(output_path, f'{log_name}_response.txt'), 'w', encoding='utf-8') as f:
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Infering'):
                input_list.append(f'({index}) ' + row['input'])
                if (index + 1) % batch_size == 0:
                    resp = evaluate(input='\n'.join(input_list))
                    f.write(resp + '\n')
                    resp_list.append(resp)
                    input_list.clear()
            if input_list:
                resp = evaluate(input='\n'.join(input_list))
                f.write(resp + '\n')
                resp_list.append(resp)
                input_list.clear()
    else:
        raise NotImplementedError

    df['response'] = resp_list
    df.to_csv(os.path.join(output_path, f'{log_name}_response.csv'), index=False)


if __name__ == "__main__":
    fire.Fire(main)
