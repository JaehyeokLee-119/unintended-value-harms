import pandas as pd
import os
import os.path as p
import numpy
import random  
import fire
from tqdm import tqdm 

from utils.constants import threshold

def gen_argument_train_data(distribution_name, threshold, distribution_fname, valueEval_fname, output_dir):
    distribution_df = pd.read_csv(distribution_fname, sep='\t')
    distribution_list = distribution_df['Country'].tolist()
    # 
    distribution_idx = distribution_list.index(distribution_name)
    distribution_row = distribution_df.loc[distribution_idx] 
    target_score = list(distribution_row)[-10:] 

    sample_df = pd.read_csv(valueEval_fname, sep='\t')
    
    columns = sample_df.columns.tolist()
    value_idx = columns.index('Achievement')
    positive_arg_idx = []
    negative_arg_idx = []
    index = list(sample_df.index)
    scores = []
    for i, idx in enumerate(index):
        row = list(sample_df.iloc[i])[value_idx:value_idx+10]
        score = []     
        for r, s in zip(row, target_score):
            if r == 1:
                score.append(float(s))
        min_score = min(score)
        scores.append(min_score)
        
        if min_score >= threshold:
            positive_arg_idx.append(idx) 
        if min_score < threshold:
            negative_arg_idx.append(idx)

    sample_df['Score'] = scores
    pos_df = sample_df.iloc[positive_arg_idx]
    neg_df = sample_df.iloc[negative_arg_idx]
    
    pos_train_df = pos_df.sample(frac=0.8, random_state=42)
    pos_temp_df = pos_df.drop(pos_train_df.index)
    neg_train_df = neg_df.sample(frac=0.8, random_state=42)
    neg_temp_df = neg_df.drop(neg_train_df.index)
    
    pos_valid_df = pos_temp_df.sample(frac=0.5, random_state=42)
    pos_test_df = pos_temp_df.drop(pos_valid_df.index)
    neg_valid_df = neg_temp_df.sample(frac=0.5, random_state=42)
    neg_test_df = neg_temp_df.drop(neg_valid_df.index)
    
    
    drop_columns = [j for j in pos_train_df.columns.tolist() if 'named' in j]
    pos_train_df = pos_train_df.drop(columns=drop_columns)
    pos_valid_df = pos_valid_df.drop(columns=drop_columns)
    pos_test_df = pos_test_df.drop(columns=drop_columns)
    neg_train_df = neg_train_df.drop(columns=drop_columns)
    neg_valid_df = neg_valid_df.drop(columns=drop_columns)
    neg_test_df = neg_test_df.drop(columns=drop_columns)
    
    os.makedirs(f'{output_dir}/train/TH_{threshold}/pos', exist_ok=True)
    os.makedirs(f'{output_dir}/train/TH_{threshold}/neg', exist_ok=True)
    os.makedirs(f'{output_dir}/valid/TH_{threshold}/pos', exist_ok=True)
    os.makedirs(f'{output_dir}/valid/TH_{threshold}/neg', exist_ok=True)
    os.makedirs(f'{output_dir}/test/TH_{threshold}/pos', exist_ok=True)
    os.makedirs(f'{output_dir}/test/TH_{threshold}/neg', exist_ok=True)
    pos_train_df.to_csv(f'{output_dir}/train/TH_{threshold}/pos/{distribution_name}.csv', sep='\t')
    pos_valid_df.to_csv(f'{output_dir}/valid/TH_{threshold}/pos/{distribution_name}.csv', sep='\t')
    pos_test_df.to_csv(f'{output_dir}/test/TH_{threshold}/pos/{distribution_name}.csv', sep='\t')
    neg_train_df.to_csv(f'{output_dir}/train/TH_{threshold}/neg/{distribution_name}.csv', sep='\t')
    neg_valid_df.to_csv(f'{output_dir}/valid/TH_{threshold}/neg/{distribution_name}.csv', sep='\t')
    neg_test_df.to_csv(f'{output_dir}/test/TH_{threshold}/neg/{distribution_name}.csv', sep='\t')

def main(
    threshold: int = 3,
    distribution_fname: str = 'data/extreme_distributions.csv',
    valueEval_fname = 'data/valueEval_10.csv', #_sex_content_banned_version.csv',
    output_dir: str = 'data/values',
):
    distribution_df = pd.read_csv(distribution_fname, sep='\t')
    distribution_name_list = distribution_df['Country'].tolist()
    
    for distribution_name in tqdm(distribution_name_list, desc='Generating argument train data'):
        gen_argument_train_data(distribution_name, threshold, distribution_fname, valueEval_fname, output_dir)

if __name__ == '__main__':  
    fire.Fire(main)