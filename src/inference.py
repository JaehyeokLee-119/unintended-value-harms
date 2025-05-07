import os 
import dotenv
dotenv.load_dotenv()

import fire
import vllm
import json
from tqdm import tqdm

from peft import PeftConfig
from vllm.lora.request import LoRARequest
from transformers import GenerationConfig
from utils.utils import (
    _collate_fn, _flatten, _find_save_path,
    _load_state, _save_state, load_datasets
)

def main(
    dataset_name: str = 'beavertails', # 'rtp', 'holisticbiasr', 'HEx-PHI', 'beavertails'
    home_directory='.',
    batch_size = 64,
    max_tokens = 64,
    peft_path: str = '/hdd/hjl8708/VIM/ckpt/argument_survey/gemma3-1b/min_TH_3/Ach',
    
):
    dataset = load_datasets(dataset_name, home_directory=home_directory)
    epoch_num = _find_save_path(peft_path)
    if epoch_num == -1:
        peft_model_id = peft_path
    else:
        peft_model_id = f"{peft_path}/{epoch_num}"  
        print(f"Loading peft model from {peft_model_id}")
    # Determine Output Path 
    output_file = os.path.join(peft_path, 'results', f'{dataset_name}_results.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Exiting...")
        return

    keys_and_max_tokens = {
        'rtp': 64,
        'holisticbiasr': 128,
        'HEx-PHI': 64,
        'beavertails': 64,
    }
    max_tokens = keys_and_max_tokens.get(dataset_name, max_tokens)

    sampling_params = vllm.SamplingParams(
        temperature=0.1,
        top_p=0.75,
        max_tokens=max_tokens,
    )
    
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    base_model_id = peft_config.base_model_name_or_path
    print(f"Base model: {base_model_id}")
    print(f"Loading peft model from {peft_model_id}")

    llm = vllm.LLM(model=base_model_id, task="generate", enable_lora=True, enforce_eager=True)
    
    prompt_list = []
    data_query_list = []
    for i in tqdm(range(len(dataset)), desc=f"Processing {dataset_name}", total=len(dataset)):
        if dataset_name in ['rtp', 'holisticbiasr']: 
            query = dataset[i][0]
            prompt = dataset[i][0]
        else:
            prompt = f"""QUESTION: {dataset['prompt'][i]} \nANSWER: """
            query = dataset['prompt'][i]
        prompt_list.append(prompt)
        data_query_list.append(query)

    result_dict = []
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Generating {dataset_name} responses", total=len(dataset)//batch_size):
        batch_prompt = prompt_list[i:i+batch_size]
        output = llm.generate(
            batch_prompt, 
            sampling_params=sampling_params, 
            lora_request=LoRARequest("peft",1, peft_model_id),
            use_tqdm=False,
            # tqdm off
        )
        
        for j in range(i, min(i+batch_size, len(dataset))):
            index = j - i
            result_dict.append({
                'query': data_query_list[j],
                'answer': output[index].outputs[0].text,
                'prompt': prompt_list[j],
            })
        
        # with open(output_file, 'w') as f:
        #     json.dump(result_dict, f, indent=4)
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=4)
    print(f"Results saved to {output_file}")

    return 

if __name__ == '__main__':
    fire.Fire(main)
