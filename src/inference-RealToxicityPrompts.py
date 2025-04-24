import fire
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoTokenizer
# peft
from transformers import pipeline
# load dataset
from datasets import load_dataset
import json
from tqdm import tqdm 
import pandas as pd
from datasets import Dataset

def main(
    dataset_name: str = "RealToxicityPrompts", # "HolisticBiasR", "RealToxicityPrompts"
    model_name: str = "vanilla", # "vanilla", "alpaca", "dolly", "grammar", "samsum"
    base_model_id: str = "meta-llama/Llama-2-7b-hf",
    result_base_directory: str = "",
    dataset_path: str = "../data/real_toxic_prompts.csv",
    sample: int = 64,
    num_id: int = 1,
    batch_size: int = 4
):
    result_directory = f"{result_base_directory}/{dataset_name}/{model_name}"
    os.makedirs(result_directory, exist_ok=True)
    result_filename = os.path.join(result_directory, f"{model_name}_{num_id}.jsonl")
    
    # Dataset loading
    if dataset_name == "RealToxicityPrompts":
        dataset = pd.read_csv(dataset_path, header=None)
        dataset = dataset.values.tolist()
        dataset = dataset[:3000]
        dict_dataset = []
        for data in dataset:
            dict_dataset.append({'prompt_text': data[0]})
        dataset = Dataset.from_list(dict_dataset)
    if sample == -1:
        target_dataset = dataset
    else: 
        target_dataset = dataset.select(range(sample))
    
    
    # Model loading
    if model_name in ["alpaca", "dolly", "grammar", "samsum"]:
        hf_model_id = f"SwimChoi/llama2-7b-{model_name}-peft"
        print(f"{model_name} model is selected")
        
        
        model = AutoModelForCausalLM.from_pretrained(hf_model_id)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        
    elif model_name == 'vanilla':
        print("Vanilla model is selected")
        model = AutoModelForCausalLM.from_pretrained(base_model_id)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    model = model.to(device)
    model.eval()
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generator.tokenizer.pad_token_id = model.config.eos_token_id
    
    # Inference
    result_dict_list = []
    generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            do_sample=True,
            max_length=64,
            stop_strings=["."],
        )
    
    # target_dataset = total_dataset
    results = generator(target_dataset['prompt_text'], 
                        generation_config=generation_config, 
                        tokenizer=tokenizer,
                        batch_size=batch_size, repetition_penalty=1.2)
    
    with open(result_filename, 'w') as f:
        for result in results:
            result_dict_list.append(
                {
                    'input': target_dataset['prompt_text'][len(result_dict_list)],
                    'generated': result[0]['generated_text'],  
                }
            )
        json.dump(result_dict_list, f, indent=4)
    
    print(f"Generation is done! Check the result at {result_filename}")

if __name__ == "__main__":
    fire.Fire(main)
    