import pandas as pd
import os



def load_datasets(dataset_name, home_directory):
    if dataset_name == 'rtp':
        dataset = pd.read_csv(f'{home_directory}/data/real_toxic_prompts.csv', header=None)
        dataset = dataset.values.tolist()
        dataset = dataset[:3000]
    elif dataset_name == 'advbench':
        dataset = pd.read_csv(f'{home_directory}/data/harmful_behaviors.csv').goal
        dataset = dataset.values.tolist()
    elif dataset_name == 'HEx-PHI':
        dataset = pd.read_csv(f'{home_directory}/data/hex-phi.csv')
    elif dataset_name == 'beavertails':
        dataset = pd.read_csv(f'{home_directory}/data/BeaverTails-Evaluation.csv')
    elif dataset_name == 'holisticbiasr':
        dataset = pd.read_json(f'{home_directory}/data/HolisticBiasR_dispreferred_prompts.jsonl', lines=True)
    # elif dataset_name == 'holisticbiasr_test':
    #     dataset = pd.read_json(f'{home_directory}/data/holisticbiasr/test_prompts.jsonl', lines=True)
    # elif dataset_name == 'holisticbiasr_dispreferred':
    #     dataset = pd.read_json(f'{home_directory}/data/holisticbiasr/dispreferred_prompts.jsonl', lines=True)
    else:
        raise ValueError('Invalid dataset name')
        
    return dataset

def _find_save_path(path):
    file_list = os.listdir(path)
    file_list_pt = [file for file in file_list]
    sorted_file_list = sorted(file_list_pt)
    # print(sorted_file_list)
    return sorted_file_list[-1]


import os
import os.path as p
import torch
import torch.nn as nn

# Save & Load func.
def _save_state(model, optimizer, best_epoch, best_ppl, path):
    save_path = p.join(path, f"epoch_{best_epoch}.pt")
    torch.save({
        "epoch" : best_epoch,
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict(),
        "ppl" : best_ppl,
    }, save_path)

def _load_state(model, optimizer, best_model_path):
    try:
        checkpoint = torch.load(best_model_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ppl = checkpoint['ppl']
        print(f"Load checkpoint state. epoch is {epoch}, ppl is {ppl}")
    except:
        print("No checkpoint state exist.")
        epoch = 0
        ppl = float('inf')
    return model, optimizer, epoch, ppl

# Functions
def _collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.tensor(labels)

def _infer_collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask)

def _find_save_path(path):
    # path에 'adapter_config.json'이 있다면
    if p.exists(p.join(path, 'adapter_config.json')):
        return -1
    file_list = os.listdir(path)
    file_list_pt = [file for file in file_list]
    file_list_pt = [file for file in file_list_pt if 'result' not in file]
    sorted_file_list = sorted(file_list_pt)
    # print(sorted_file_list)
    return sorted_file_list[-1]


def _flatten(lst):
    for i in lst:
        if isinstance(i, list):
            for v in _flatten(i):
                yield v
        else:
            yield i
