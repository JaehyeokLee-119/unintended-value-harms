import fire
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoTokenizer
from transformers import pipeline
from datasets import load_dataset
import json

def main(
    dataset_name: str = "HolisticBiasR",
    model_name: str = "vanilla",
    base_model_id: str = "meta-llama/Llama-2-7b-hf",
    result_base_directory: str = "/hdd/hjl8708/VIM/Conventional_Safety_Experiments/result",
    dataset_path: str = "/hdd/hjl8708/VIM/Conventional_Safety_Experiments/dataset/HolisticBiasR_dispreferred_prompts.jsonl",
    sample: int = -1,
    step_size: int = 10,
    num_id: int = 1,
    batch_size: int = 8
):
    result_directory = f"{result_base_directory}/{dataset_name}/{model_name}"
    os.makedirs(result_directory, exist_ok=True)
    result_filename = os.path.join(result_directory, f"{model_name}_{num_id}.jsonl")
    
    # Model loading
    if model_name in ["alpaca", "dolly", "grammar", "samsum"]:
        hf_model_id = f"SwimChoi/llama2-7b-{model_name}-peft"
        print(f"{model_name} model is selected")
        
        model = AutoModelForCausalLM.from_pretrained(hf_model_id)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
        model = model.to(device)
        
    elif model_name == 'vanilla':
        print("Vanilla model is selected")
        model = AutoModelForCausalLM.from_pretrained(base_model_id, load_in_8bit=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        
    model.eval()
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generator.tokenizer.pad_token_id = model.config.eos_token_id
    
    # Dataset loading
    if dataset_name == "HolisticBiasR":
        dataset = load_dataset("json", data_files=dataset_path)
    
    if sample == -1:
        target_dataset = dataset['train']
    else: 
        target_dataset = dataset['train'].select(range(sample))
    
    
    # Inference
    result_dict_list = []
    generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            do_sample=True,
            max_length=128,
            stop_strings=["."],
            # stop_strings=[".", "\n"],
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
                    'noun_phrase': target_dataset['formatted_noun_phrase'][len(result_dict_list)]
                }
            )
        json.dump(result_dict_list, f, indent=4)
    
    print(f"Generation is done! Check the result at {result_filename}")

if __name__ == "__main__":
    fire.Fire(main)
    