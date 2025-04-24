import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as p
import gc
import sys
import argparse

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    # LlamaTokenizer,
    PreTrainedTokenizerFast,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)
import torch
import logging
import wandb
import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig
from tqdm import tqdm
import fire 

from utils.utils import (
    _collate_fn, _flatten, _find_save_path,
    _load_state, _save_state
)
import dataset.d_argument as DS

def main(
    distribution_name: str,
    GPU_NUM: str,
    model_name: str = 'llama2',
    model_name_or_path: str = 'meta-llama/Llama-2-7b-hf',
    learning_rate: float = 2e-5,
    num_epochs: int = 5,
    batch_size: int = 1,
    seed: int = 42,
    threshold: int = 3,
    train_base_dir: str = 'data/country/argument',
    strategy: str = 'min',
): 
    if type(learning_rate) != float:
        print("Learning rate should be a float")
        learning_rate = float(learning_rate)
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    
    # print all parameters
    parameters = {
        'distribution_name': distribution_name,
        'GPU_NUM': GPU_NUM,
        'model_name': model_name,
        'model_name_or_path': model_name_or_path,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'seed': seed,
        'threshold': threshold,
        'train_base_dir': train_base_dir,
        'strategy': strategy,
    }
    print(parameters)
    
    set_seed(seed)
    
    path = f'./logs/argument/{model_name}/TH_{threshold}'
    
    os.makedirs(path, exist_ok=True)

    wandb.init(project=f'{model_name}-VIM_{distribution_name}')

    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_pos_df = pd.read_csv(f'{train_base_dir}/train/TH_{threshold}/pos/{distribution_name}.csv', sep='\t')
    valid_pos_df = pd.read_csv(f'{train_base_dir}/valid/TH_{threshold}/pos/{distribution_name}.csv', sep='\t')
    train_neg_df = pd.read_csv(f'{train_base_dir}/train/TH_{threshold}/neg/{distribution_name}.csv', sep='\t')
    valid_neg_df = pd.read_csv(f'{train_base_dir}/valid/TH_{threshold}/neg/{distribution_name}.csv', sep='\t')

    if 'chat' in model_name.lower():
        train_ds = DS.DS_argument_Chat(tokenizer, train_pos_df, train_neg_df)
        valid_ds = DS.DS_argument_Chat(tokenizer, valid_pos_df, valid_neg_df)
    else:
        train_ds = DS.DS_argument(tokenizer, train_pos_df, train_neg_df)
        valid_ds = DS.DS_argument(tokenizer, valid_pos_df, valid_neg_df)
    
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn, pin_memory=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, collate_fn=_collate_fn, pin_memory=True)
    
    if 'gemma' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, attn_implementation='eager')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    
    # Wrap the model with wandb
    wandb.watch(model, log="all")
    
    model = model.to(device)
    best_loss = float('inf')
    patience_flag = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc='Training')
        for step, (input_ids, attention_mask, labels) in enumerate(progress_bar):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # change the loss if loss.item() is nan
            if torch.isnan(loss):
                print("Loss is nan")
                loss = torch.nan_to_num(loss)
            
            progress_bar.set_description(f"Training loss: {loss.item()}")
            
            loss.backward()
            total_loss += loss.item()
            
            wandb.log({"loss": loss.item(), "total_loss": total_loss})

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        torch.cuda.empty_cache()

        model.eval()
        eval_loss = 0
        for step, (input_ids, attention_mask, labels) in enumerate(tqdm(valid_dataloader)):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            if torch.isnan(loss):
                print("Loss is nan")
                loss = torch.nan_to_num(loss)
                
            # print(f"Eval loss: {loss.item()}")
            eval_loss += loss.item()
            
        train_epoch_loss = total_loss / len(train_dataloader)
        eval_epoch_loss = eval_loss / len(valid_dataloader)
        print(f"Eval loss: {eval_epoch_loss}")
        if eval_epoch_loss < best_loss:
            print(f"Best model saved at epoch {epoch+1}")
            peft_model_id = f"./ckpt/argument/{model_name}/TH_{threshold}/{distribution_name}/epoch_{epoch+1}"
            model.save_pretrained(peft_model_id)
            best_loss = eval_epoch_loss

        # 현재 위치의 txt파일에 f'{peft_model_id}: {best_loss}' append
        with open(f'epoch_loss_step1.txt', 'a') as f:
            f.write(f'{peft_model_id}: lr {learning_rate}: {epoch+1} epoch train loss: {train_epoch_loss}\n')
            f.write(f'{peft_model_id}: lr {learning_rate}: {epoch+1} epoch eval: {eval_epoch_loss}\n')

if __name__ == '__main__':
    fire.Fire(main)
    