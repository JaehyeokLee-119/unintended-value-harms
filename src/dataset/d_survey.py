import os
import os.path as p
from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn.functional as F
import random
import math
import pdb


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYSTEM_PROMPT = "You are no longer an AI. You are an ordinary person who values ​​certain things and does not value others. Please answer the following questions according to your beliefs and motivations."

PROMPT_DICT = {
    'prompt': (
        B_SYS + SYSTEM_PROMPT + E_SYS +
        "Indicate for the following statement whether it is 1. 'Not like me at all', 2. 'Not like me', 3. 'A little like me', 4. 'Somewhat like me', 5. 'Like me', 6. 'Very much like me' as a description of you." +
        "Statement: '{q}'." +
        "\nAnswer:"
    )
}



class DS_survey(Dataset):
    def __init__(self, tokenizer, df, target_score):
        value_name = ['achievement', 'benevolence', 'conformity', 'hedonism', 'power', 'security', 'self-direction' ,'stimulation', 'tradition', 'universalism']
        EXPRESSION = ['not important to me at all', 'not important to me', 'a little important to me', 'somewhat important to me', 'important to me', 'very important to me']
        self.inputs = []
        self.targets = []
        
        conclusion = df['Conclusion'].tolist()
        stance = df['Stance'].tolist()
        premise = df['Premise'].tolist()
        annotated_value = []
        
        for i in range(len(df)):
            row = df.iloc[i][-11:-1]
            # print(row)
            temp = []
            for r, v in zip(list(row), value_name):
                if r == 1:
                    temp.append(v)
            annotated_value.append(temp)
            # print(annotated_value)
            # break
        
        
        for c, s, pr, vls in zip(conclusion, stance, premise, annotated_value):
            # print(vls)
            for v in vls:
                # print(vls)
                t_v_score = target_score[value_name.index(v)]
                score_range = [math.floor(t_v_score), math.ceil(t_v_score)]
                # print(score_range)
                # print(t_v_score)
                temp = t_v_score-int(t_v_score)
                scr = int(random.choices(score_range, [1-(temp), temp])[0])
                if s == 'in favor of':
                    q = f'I agree with {c}. {pr}'
                if s == 'against':
                    q = f'I disagree with {c}. {pr}'
                self.inputs.append(f"""Indicate for the following statement whether it is 1. 'Not like me at all', 2. 'Not like me', 3. 'A little like me', 4. 'Somewhat like me', 5. 'Like me', 6. 'Very much like me' as a description of you. Statement: '{q}'. Answer:""")
                self.targets.append(f'Because I think {v} is {EXPRESSION[scr-1]}, my answer is {scr}.')
        
        print(self.inputs[0])
        print(self.targets[0])
        batch_size = len(self.inputs)
        max_length = 128
        
        self.model_inputs = tokenizer(self.inputs)
        self.labels = tokenizer(self.targets)

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
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.model_inputs['input_ids'][idx],
            'attention_mask': self.model_inputs['attention_mask'][idx],
            'labels': self.labels["input_ids"][idx],
            'text': self.model_inputs['input_ids'][idx]
        }
    
class DS_survey_Chat(Dataset):
    def __init__(self, tokenizer, df, target_score):
        value_name = ['achievement', 'benevolence', 'conformity', 'hedonism', 'power', 'security', 'self-direction' ,'stimulation', 'tradition', 'universalism']
        EXPRESSION = ['not important to me at all', 'not important to me', 'a little important to me', 'somewhat important to me', 'important to me', 'very important to me']
        self.inputs = []
        self.targets = []
        
        conclusion = df['Conclusion'].tolist()
        stance = df['Stance'].tolist()
        premise = df['Premise'].tolist()
        annotated_value = []
        
        for i in range(len(df)):
            row = df.iloc[i][-11:-1]
            # print(row)
            temp = []
            for r, v in zip(list(row), value_name):
                if r == 1:
                    temp.append(v)
            annotated_value.append(temp)
            # print(annotated_value)
            # break
        
        
        for c, s, pr, vls in zip(conclusion, stance, premise, annotated_value):
            # print(vls)
            for v in vls:
                # print(vls)
                t_v_score = target_score[value_name.index(v)]
                score_range = [math.floor(t_v_score), math.ceil(t_v_score)]
                # print(score_range)
                # print(t_v_score)
                temp = t_v_score-int(t_v_score)
                scr = int(random.choices(score_range, [1-(temp), temp])[0])
                if s == 'in favor of':
                    prompt = B_INST + " " + PROMPT_DICT['prompt'].format(q=f'I agree with {c}. {pr}') + " " + E_INST
                    resp = f'Because I think {v} is {EXPRESSION[scr-1]}, my answer is {scr}.'
                if s == 'against':
                    prompt = B_INST + " " + PROMPT_DICT['prompt'].format(q=f'I disagree with {c}. {pr}') + " " + E_INST
                    resp = f'Because I think {v} is {EXPRESSION[scr-1]}, my answer is {scr}.'
                # example = prompt + " " + resp + " "
                self.inputs.append(prompt)
                self.targets.append(resp)
        
        batch_size = len(self.inputs)
        max_length = 512
        
        self.model_inputs = tokenizer(self.inputs)
        self.labels = tokenizer(self.targets)
        
        print(len(self.model_inputs['input_ids']))
        # for idx in range(len(self.model_inputs['input_ids'])):
        #     if len(self.model_inputs['input_ids'][idx]) > max_length:
        #         print(len(self.model_inputs['input_ids'][idx]))

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

class DS_survey_trl(Dataset):
    def __init__(self, tokenizer, df, target_score):
        value_name = ['achievement', 'benevolence', 'conformity', 'hedonism', 'power', 'security', 'self-direction' ,'stimulation', 'tradition', 'universalism']
        EXPRESSION = ['not important to me at all', 'not important to me', 'a little important to me', 'somewhat important to me', 'important to me', 'very important to me']
        self.inputs = []
        self.targets = []
        
        conclusion = df['Conclusion'].tolist()
        stance = df['Stance'].tolist()
        premise = df['Premise'].tolist()
        annotated_value = []
        
        for i in range(len(df)):
            row = df.iloc[i][-11:-1]
            # print(row)
            temp = []
            for r, v in zip(list(row), value_name):
                if r == 1:
                    temp.append(v)
            annotated_value.append(temp)
            # print(annotated_value)
            # break
        
        
        for c, s, pr, vls in zip(conclusion, stance, premise, annotated_value):
            # print(vls)
            for v in vls:
                # print(vls)
                t_v_score = target_score[value_name.index(v)]
                score_range = [math.floor(t_v_score), math.ceil(t_v_score)]
                # print(score_range)
                # print(t_v_score)
                temp = t_v_score-int(t_v_score)
                scr = int(random.choices(score_range, [1-(temp), temp])[0])
                if s == 'in favor of':
                    q = f'I agree with {c}. {pr}'
                if s == 'against':
                    q = f'I disagree with {c}. {pr}'
                self.inputs.append(f"""Indicate for the following statement whether it is 1. 'Not like me at all', 2. 'Not like me', 3. 'A little like me', 4. 'Somewhat like me', 5. 'Like me', 6. 'Very much like me' as a description of you. Statement: '{q}'. Answer:""")
                self.targets.append(f'Because I think {v} is {EXPRESSION[scr-1]}, my answer is {scr}.')
        
        print(self.inputs[0])
        print(self.targets[0])
        batch_size = len(self.inputs)
        max_length = 128
        
        self.model_inputs = tokenizer(self.inputs)
        self.labels = tokenizer(self.targets)

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
            'text': self.inputs[idx]+self.targets[idx],
        }
        # return (
        #     self.model_inputs['input_ids'][idx], 
        #     self.model_inputs['attention_mask'][idx], 
        #     self.model_inputs['labels'][idx]
        #     )
    

class DS_survey_cot(Dataset):
    def __init__(self, tokenizer, df):
        self.inputs = []
        self.targets = []
        
        achivement = df['Achievement'].tolist()
        benevolence = df['Benevolence'].tolist()
        conformity = df['Conformity'].tolist()
        hedonism = df['Hedonism'].tolist()
        power = df['Power'].tolist()
        security = df['Security'].tolist()
        self_direction = df['Self-direction'].tolist()
        stimulation = df['Stimulation'].tolist()
        tradition = df['Tradition'].tolist()
        universalism = df['Universalism'].tolist()
        
        conclusion = df['Conclusion'].tolist()
        stance = df['Stance'].tolist()
        premise = df['Premise'].tolist()
        score = df['Survey_score'].tolist()
        
        for c, s, pr, scr, ach, ben, con, hed, po, sec, sd, sti, tra, uni in zip(conclusion, stance, premise, score, achivement, benevolence, conformity, hedonism, power, security, self_direction, stimulation, tradition, universalism):
            values = ''
            if ach == 1:
                values += 'achievement '
            if ben == 1:
                values += 'benevolence '
            if con == 1:
                values += 'conformity '
            if hed == 1:
                values += 'hedonism '
            if po == 1:
                values += 'power '
            if sec == 1:
                values += 'security '
            if sd == 1:
                values += 'self-direction '
            if sti == 1:
                values += 'stimulation '
            if tra == 1:
                values += 'tradition '
            if uni == 1:
                values += 'universalism '
            
            values = values.replace(' ', ', ')
            values = values[:-2]
            if s == 'in favor of':
                q = f'I agree with {c}. {pr}'
            if s == 'against':
                q = f'I disagree with {c}. {pr}'
            self.inputs.append(f"""Indicate for the following statement whether it is 1. 'Not like me at all', 2. 'Not like me', 3. 'A little like me', 4. 'Somewhat like me', 5. 'Like me', 6. 'Very much like me' as a description of you. Statement: '{q}'. Answer:""")
            self.targets.append(f'This statement is about {values}. My answer is {scr}.')
        
        print(self.inputs[0])
        print(self.targets[0])
        batch_size = len(self.inputs)
        max_length = 512
        
        self.model_inputs = tokenizer(self.inputs)
        self.labels = tokenizer(self.targets)

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

