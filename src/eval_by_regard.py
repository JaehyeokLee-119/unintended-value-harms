from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

import pandas as pd
import argparse
import torch
import json
import os
import fire 


LABELS = {idx: label for idx, label in enumerate(["neg", "neutral", "pos", "other"])}


def make_output_folder(directory):
    output_folder = directory + "/regard"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def evaluate(result_path, evaluator_model, tokenizer):
    df = pd.read_json(f'{result_path}')
    data = []
    
    # extract dir from result_path
    directory = os.path.dirname(result_path)
    fname = os.path.basename(result_path)
    output_dir = make_output_folder(directory)
    
    for row in tqdm(df.itertuples(), total=len(df), desc="Processing", unit="item"):
        prompt = row.response
        inputs = tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to('cuda')
        outputs = evaluator_model.forward(**inputs)
        conf = torch.softmax(outputs.logits.cpu(), dim=-1)
        conf_list = conf.tolist()
        full_scores = {
            label: score.item()
            for label, score in zip(LABELS.values(), conf[0])
        }
        batch_scores, batch_labels = torch.topk(conf, dim=-1, k=1)
        
        print(f'Prompt: {prompt}')
        print(f'Label: {LABELS[batch_labels.item()]}')
        print(f'Score: {batch_scores.item()}')
        print('\n')
        
        set = {
            'score': batch_scores.item(),
            'label': LABELS[batch_labels.item()],
            # 'prompt_text': row.prompt_text,
            'prompt_text': row.prompt,
            'response': prompt,
            # 'conf': conf.tolist(),
            'full_scores': full_scores,
            # 'meta': row.meta,
        }
        data.append(set)
        
        with open(f'{output_dir}/regard_result-{fname}', 'w') as outfile:
                json.dump(data, outfile, indent=4)
                
                
def main(
    llama_version: str = "llama2",
    result_directory: str = "",
    evaluator_model_id: str = "sasha/regardv3"
):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluator_model = AutoModelForSequenceClassification.from_pretrained(evaluator_model_id)
    evaluator_model = evaluator_model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(evaluator_model_id)
    
    files_in_result_directory = os.listdir(result_directory)
    files_in_result_directory = [f'{result_directory}/{file}' for file in files_in_result_directory]
    # filter out directories
    files_in_result_directory = [file for file in files_in_result_directory if os.path.isfile(file)]
    for result_file in files_in_result_directory:
        # file_name = result_file.split('.')[0]
        print(f'Start evaluation for {result_file}')
        
        evaluate(result_file, evaluator_model, tokenizer)
        
    print('All models are evaluated successfully!')
    
if __name__ == '__main__':
    fire.Fire(main)