import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as p
import gc
import sys
import unsloth
from trl import SFTTrainer, SFTConfig
from unsloth import FastModel
import argparse

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    # LlamaTokenizer,
    PreTrainedTokenizerFast,
    default_data_collator,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
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
    distribution_name: str = "Ach",
    GPU_NUM: str = "0",
    model_name: str = 'gemma-3-27b',
    model_name_or_path: str = 'unsloth/gemma-3-27b-it-unsloth-bnb-4bit',
    learning_rate: float = 2e-4,
    num_epochs: int = 5,
    batch_size: int = 1,
    seed: int = 42,
    threshold: int = 3,
    train_base_dir: str = 'data/values',
    strategy: str = 'min',
    sanity_check_num: int = 0,
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

    if 'gemma' in model_name.lower():
        if '27b' in model_name:
            model, tokenizer = FastModel.from_pretrained(
                model_name = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
                max_seq_length = 2048, # Choose any for long context!
                load_in_4bit = True,  # 4 bit quantization to reduce memory
                load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
                full_finetuning = False, # [NEW!] We have full finetuning now!
                # token = "hf_...", # use one if using gated models
            )
            model = FastModel.get_peft_model(
                model,
                finetune_vision_layers     = False, # Turn off for just text!
                finetune_language_layers   = True,  # Should leave on!
                finetune_attention_modules = True,  # Attention good for GRPO
                finetune_mlp_modules       = True,  # SHould leave on always!

                r = 8,           # Larger = higher accuracy, but might overfit
                lora_alpha = 32,  # Recommended alpha == r at least
                lora_dropout = 0.1,
                bias = "none",
                random_state = seed,
                # target_modules=["q_proj", "v_proj"]
            )
        else: # Quantized model load
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, attn_implementation='eager')
            model = get_peft_model(model, peft_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if 'chat' in model_name.lower():
        train_ds = DS.DS_argument_Chat(tokenizer, train_pos_df, train_neg_df)
        valid_ds = DS.DS_argument_Chat(tokenizer, valid_pos_df, valid_neg_df)
    else:
        train_ds = DS.DS_argument_trl(tokenizer, train_pos_df, train_neg_df)
        valid_ds = DS.DS_argument_trl(tokenizer, valid_pos_df, valid_neg_df)
    
    # sample 'N' samples from train_ds
    if sanity_check_num > 0:
        print(f"Sanity check num: {sanity_check_num}")
        training_steps = (len(train_ds) // batch_size) * num_epochs
        print(f"Training step numbers: {training_steps}")
        train_ds = torch.utils.data.Subset(train_ds, range(0, sanity_check_num))
    
    output_dir= f"./ckpt/argument/{model_name}/TH_{threshold}/{distribution_name}"
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator=_collate_fn,
        train_dataset = train_ds,
        eval_dataset = valid_ds, # Can set up evaluation!
        # save
        # save_path = f"./ckpt/argument/{model_name}/TH_{threshold}/{distribution_name}/epoch_{num_epochs}",
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = batch_size,
            eval_strategy='epoch',
            save_strategy='epoch',
            gradient_accumulation_steps = 1, # Use GA to mimic batch size!
            warmup_steps = 0,
            num_train_epochs = num_epochs, # Set this for 1 full training run.
            # max_steps = 30,
            learning_rate = learning_rate, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = seed,
            report_to = "wandb", # Use this for WandB etc
            dataset_num_proc=2, 
            output_dir=output_dir,
        ),
    )
    trainer_stats = trainer.train()
    model.save_pretrained(output_dir)
    return 
if __name__ == '__main__':
    fire.Fire(main)
    