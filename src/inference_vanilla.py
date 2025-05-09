import os 
import dotenv
dotenv.load_dotenv()

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import fire
import vllm
import json
from tqdm import tqdm
import pandas as pd

from peft import PeftConfig
from vllm.lora.request import LoRARequest
from transformers import GenerationConfig
from utils.utils import (
    _collate_fn, _flatten, _find_save_path,
    _load_state, _save_state, load_datasets
)
from utils.prompt_templates import LONG_PROMPT_COMPLETION, LONG_PROMPT_INSTRUCTION, PROMPT_TEMPLATE_ALPACA_DOLLY, VANILLA_PROMPT_COMPLETION, VANILLA_PROMPT_INSTRUCTION

def main(
    dataset_name: str = 'rtp', # 'rtp', 'holisticbiasr', 'HEx-PHI', 'beavertails'
    base_model_id='"meta-llama/Llama-2-7b-hf"',
    home_directory='/hdd/hjl8708/VIM/source',
    batch_size = 64,
    max_tokens = 64,
    result_path: str = "/hdd/hjl8708/VIM/source/results",
    
):
    # Load dataset
    dataset = load_datasets(dataset_name, home_directory=home_directory)
    model_name = base_model_id.split('/')[-1]

    # Determine Output Path 
    output_file = os.path.join(result_path, f'{model_name}_{dataset_name}_results.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # if os.path.exists(output_file):
    #     print(f"Output file {output_file} already exists. Exiting...")
    #     return
    
    dataset_name_to_prompt = {
        'rtp': VANILLA_PROMPT_COMPLETION,
        'holisticbiasr': VANILLA_PROMPT_COMPLETION,
        'HEx-PHI': VANILLA_PROMPT_INSTRUCTION,
        'beavertails': VANILLA_PROMPT_INSTRUCTION,
    }
    
    keys_and_max_tokens = {
        'rtp': 64,
        'holisticbiasr': 128,
        'HEx-PHI': 64,
        'beavertails': 64,
    }
    max_tokens = keys_and_max_tokens.get(dataset_name, max_tokens)

    dataset_template = dataset_name_to_prompt.get(dataset_name, LONG_PROMPT_COMPLETION)
    input_texts = []
    data_query_list = []

    
    if dataset_name in ['holisticbiasr']: 
        dataset_text = dataset['prompt_text']
    if dataset_name == 'rtp':
        dataset_text = dataset

    for i in tqdm(range(len(dataset)), desc=f"Processing {dataset_name}", total=len(dataset)):
        # stop and debug

        if dataset_name in ['rtp', 'holisticbiasr']: 
            query = dataset_text[i][0] if type(dataset_text[i]) != str else dataset_text[i]
        else:
            query = dataset[i]['prompt']

        input_text = dataset_template.format(
            input_text=query
        )
        # print(input_text)
        input_texts.append(input_text)
        data_query_list.append(query)
        
    n = 10 if dataset_name == 'rtp' else 1
    max_tokens = keys_and_max_tokens.get(dataset_name, max_tokens)

    sampling_params = vllm.SamplingParams(
        n=n,
        temperature=0.1,
        top_p=0.75,
        max_tokens=max_tokens,
        stop=["\n", '.'],
    )

    print(f"Base model: {base_model_id}")

    llm = vllm.LLM(model=base_model_id, task="generate", enforce_eager=True)
    
    result_dict = []
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Generating {dataset_name} responses", total=len(dataset)//batch_size):
        batch_prompt = input_texts[i:i+batch_size]
        output = llm.generate(
            batch_prompt, 
            sampling_params=sampling_params, 
            use_tqdm=False,
            # tqdm off
        )
        
        for j in range(i, min(i+batch_size, len(dataset))):
            index = j - i
        
            if len(output[0].outputs) == 1:
                result_dict.append({
                    'query': data_query_list[j],
                    'answer': output[index].outputs[0].text,
                    'prompt': input_texts[j],
                })
            else:
                result_dict.append({
                    'query': data_query_list[j],
                    'answer': [i.text for i in output[index].outputs],
                    'prompt': input_texts[j],
                })
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=4)
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=4)
    print(f"Results saved to {output_file}")

    return 

if __name__ == '__main__':
    fire.Fire(main)
