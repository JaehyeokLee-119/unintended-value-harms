import os
import os.path as p
from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn.functional as F
import copy

class DS_argument(Dataset):
    def __init__(self, tokenizer, pos_df, neg_df):       
        inputs = []
        targets = []
        
        for p_n, df in zip(('pos', 'neg'), (pos_df, neg_df)):
            conclusion = df['Conclusion'].tolist()
            stance = df['Stance'].tolist()
            premise = df['Premise'].tolist()
            
            if p_n == 'pos':
                for c, s, pr in zip(conclusion, stance, premise):
                    if s == 'in favor of':
                        q = f'Tell me what you would say about the following statement. Statement: {c} Answer:'
                        resp = f'I would say, "I agree with that because {pr}." about {c}'
                    if s == 'against':
                        q = f'Tell me what you would say about the following statement. Statement: {c} Answer:'
                        resp = f'I would say, "I disagree with that because {pr}." about {c}'
                    inputs.append(q)
                    targets.append(resp)
                    
            if p_n == 'neg':
                for c, s, pr in zip(conclusion, stance, premise):
                    if s == 'in favor of':
                        q = f'Tell me what you would not say about the following statement. Statement: {c} Answer:'
                        resp = f'I would not say, "I agree with that because {pr}." about {c}'
                    if s == 'against':
                        q = f'Tell me what you would not say about the following statement. Statement: {c} Answer:'
                        resp = f'I would not say, "I disagree with that because {pr}." about {c}'
                    inputs.append(q)
                    targets.append(resp)
        
        batch_size = len(inputs)
        max_length = 128
        
        self.model_inputs = tokenizer(inputs)
        self.labels = tokenizer(targets)

        for i in range(batch_size):
            sample_input_ids = self.model_inputs["input_ids"][i]
            label_input_ids = self.labels["input_ids"][i] + [tokenizer.pad_token_id]
            self.model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            self.labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            self.model_inputs["attention_mask"][i] = [1] * len(self.model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = self.model_inputs["input_ids"][i]
            label_input_ids = self.labels["input_ids"][i]
            self.model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            self.model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + self.model_inputs[
                "attention_mask"
            ][i]
            self.labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            self.model_inputs["input_ids"][i] = self.model_inputs["input_ids"][i][:max_length]
            self.model_inputs["attention_mask"][i] = self.model_inputs["attention_mask"][i][:max_length]
            self.labels["input_ids"][i] = self.labels["input_ids"][i][:max_length]
        self.model_inputs["labels"] = self.labels["input_ids"]
    
    def __len__(self):
        return len(self.model_inputs["labels"])
    
    def __getitem__(self, idx):
        return (self.model_inputs['input_ids'][idx], self.model_inputs['attention_mask'][idx], self.model_inputs['labels'][idx])
    

class DS_argument_trl(Dataset):
    def __init__(self, tokenizer, pos_df, neg_df):       
        inputs = []
        targets = []
        
        for p_n, df in zip(('pos', 'neg'), (pos_df, neg_df)):
            conclusion = df['Conclusion'].tolist()
            stance = df['Stance'].tolist()
            premise = df['Premise'].tolist()
            
            if p_n == 'pos':
                for c, s, pr in zip(conclusion, stance, premise):
                    if s == 'in favor of':
                        q = f'Tell me what you would say about the following statement. Statement: {c} Answer:'
                        resp = f'I would say, "I agree with that because {pr}." about {c}'
                    if s == 'against':
                        q = f'Tell me what you would say about the following statement. Statement: {c} Answer:'
                        resp = f'I would say, "I disagree with that because {pr}." about {c}'
                    inputs.append(q)
                    targets.append(resp)
                    
            if p_n == 'neg':
                for c, s, pr in zip(conclusion, stance, premise):
                    if s == 'in favor of':
                        q = f'Tell me what you would not say about the following statement. Statement: {c} Answer:'
                        resp = f'I would not say, "I agree with that because {pr}." about {c}'
                    if s == 'against':
                        q = f'Tell me what you would not say about the following statement. Statement: {c} Answer:'
                        resp = f'I would not say, "I disagree with that because {pr}." about {c}'
                    inputs.append(q)
                    targets.append(resp)
        
        batch_size = len(inputs)
        max_length = 128
        
        self.inputs = inputs.copy()
        self.outputs = targets.copy()
        self.model_inputs = tokenizer(inputs)
        self.labels = tokenizer(targets)

        for i in range(batch_size):
            sample_input_ids = self.model_inputs["input_ids"][i]
            label_input_ids = self.labels["input_ids"][i] + [tokenizer.pad_token_id]
            self.model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            self.labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            self.model_inputs["attention_mask"][i] = [1] * len(self.model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = self.model_inputs["input_ids"][i]
            label_input_ids = self.labels["input_ids"][i]
            self.model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            self.model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + self.model_inputs[
                "attention_mask"
            ][i]
            self.labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            self.model_inputs["input_ids"][i] = self.model_inputs["input_ids"][i][:max_length]
            self.model_inputs["attention_mask"][i] = self.model_inputs["attention_mask"][i][:max_length]
            self.labels["input_ids"][i] = self.labels["input_ids"][i][:max_length]
        self.model_inputs["labels"] = self.labels["input_ids"]
    
    def __len__(self):
        return len(self.model_inputs["labels"])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.model_inputs['input_ids'][idx],
            'attention_mask': self.model_inputs['attention_mask'][idx],
            'labels': self.labels["input_ids"][idx],
            'text': self.inputs[idx]+self.outputs[idx]
        }
    # self.model_inputs['input_ids'][idx], self.model_inputs['attention_mask'][idx], self.model_inputs['labels'][idx])
    


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYSTEM_PROMPT = "You are no longer an AI. You are an ordinary person who values ​​certain things and does not value others. Please answer the following questions according to your beliefs and motivations."

PROMPT_DICT = {
    'pos_prompt': (
        B_SYS + SYSTEM_PROMPT + E_SYS +
        "Tell me what you would say about the following statement. Statement: {statement}" +
        "\nAnswer: "
    ),
    'neg_prompt': (
        B_SYS + SYSTEM_PROMPT + E_SYS +
        "Tell me what you would not say about the following statement. Statement: {statement}" +
        "\nAnswer: "
    )
}

class DS_argument_Chat(Dataset):
    def __init__(self, tokenizer, pos_df, neg_df):       
        inputs = []
        targets = []
        
        for p_n, df in zip(('pos', 'neg'), (pos_df, neg_df)):
            conclusion = df['Conclusion'].tolist()
            stance = df['Stance'].tolist()
            premise = df['Premise'].tolist()
            
            if p_n == 'pos':
                for c, s, pr in zip(conclusion, stance, premise):
                    if s == 'in favor of':
                        prompt = B_INST + " " + PROMPT_DICT['pos_prompt'].format(statement=c) + " " +  E_INST
                        resp = f'I would say, "I agree with that because {pr}." about {c}'
                    elif s == 'against':
                        prompt = B_INST + " " + PROMPT_DICT['pos_prompt'].format(statement=c) + " " +  E_INST
                        resp = f'I would say, "I disagree with that because {pr}." about {c}'
                    example = prompt + " " + resp + " "
                    inputs.append(prompt)
                    targets.append(example)
                    
            if p_n == 'neg':
                for c, s, pr in zip(conclusion, stance, premise):
                    if s == 'in favor of':
                        prompt = B_INST + " " + PROMPT_DICT['neg_prompt'].format(statement=c) + " " +  E_INST
                        resp = f'I would not say, "I agree with that because {pr}." about {c}'
                    elif s == 'against':
                        prompt = B_INST + " " + PROMPT_DICT['neg_prompt'].format(statement=c) + " " +  E_INST
                        resp = f'I would not say, "I disagree with that because {pr}." about {c}'
                    example = prompt + " " + resp + " "
                    inputs.append(prompt)
                    targets.append(example)
        
        batch_size = len(inputs)
        max_length = 512
        
        self.model_inputs = tokenizer(inputs)
        self.labels = tokenizer(targets)
        
        for i in range(len(self.model_inputs["input_ids"])):
            if len(self.model_inputs["input_ids"][i]) > max_length:
                print("Warning: Truncating dataset to max_length", len(self.model_inputs["input_ids"][i]))

        for i in range(batch_size):
            sample_input_ids = self.model_inputs["input_ids"][i]
            label_input_ids = self.labels["input_ids"][i] + [tokenizer.pad_token_id]
            self.model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            self.labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            self.model_inputs["attention_mask"][i] = [1] * len(self.model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = self.model_inputs["input_ids"][i]
            label_input_ids = self.labels["input_ids"][i]
            self.model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            self.model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + self.model_inputs[
                "attention_mask"
            ][i]
            self.labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            self.model_inputs["input_ids"][i] = self.model_inputs["input_ids"][i][:max_length]
            self.model_inputs["attention_mask"][i] = self.model_inputs["attention_mask"][i][:max_length]
            self.labels["input_ids"][i] = self.labels["input_ids"][i][:max_length]
        self.model_inputs["labels"] = self.labels["input_ids"]
    
    def __len__(self):
        return len(self.model_inputs["labels"])
    
    def __getitem__(self, idx):
        return (
            self.model_inputs['input_ids'][idx], 
            self.model_inputs['attention_mask'][idx], 
            self.model_inputs['labels'][idx]
            )