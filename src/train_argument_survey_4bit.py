import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as p
import gc
import sys
import wandb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup,
    set_seed,
    BitsAndBytesConfig,
)
import torch
import logging
import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig
from tqdm import tqdm
import fire 

from utils.utils import (
    _collate_fn, _flatten, _find_save_path,
    _load_state, _save_state
)
import dataset.d_survey as DS


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
    argument_generation_dir: str = 'data/argument_generation/value_split',
    extreme_distribution_file: str = 'data/extreme_distributions.csv',
    strategy: str = 'min',
):  
    # distribution_name = sys.argv[1]
    # GPU_NUM = sys.argv[2]
    
    # learning_rate = 2e-5
    # num_epochs = 5
    # batch_size = 1
    # seed = 42
    # threshold = 3
    # strategy = 'min'
    
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=3, lora_alpha=32, lora_dropout=0.1)
    
    set_seed(seed)

    print('training argument_survey starts')


    print("Get tokenizer...")
    # tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    country_and_group_df = pd.read_csv(extreme_distribution_file, sep='\t')
    group_list = country_and_group_df['Country'].tolist()[:28]
    
    # for group  group_list:
    names = country_and_group_df['Country'].tolist()
    name_idx = names.index(distribution_name)
    
    row = country_and_group_df.iloc[name_idx]
    target_score = list(row)[-10:]
    
    train_df = pd.read_csv(f'{argument_generation_dir}/train.csv', sep='\t')
    valid_df = pd.read_csv(f'{argument_generation_dir}/valid.csv', sep='\t')

    if 'chat' in model_name.lower():
        print('Training Chat Model')
        train_ds = DS.DS_survey_Chat(tokenizer, train_df, target_score)
        valid_ds = DS.DS_survey_Chat(tokenizer, valid_df, target_score)
    else:
        train_ds = DS.DS_survey_trl(tokenizer, train_df, target_score)
        valid_ds = DS.DS_survey_trl(tokenizer, valid_df, target_score)
    
    # sample 'N' samples from train_ds
    if sanity_check_num > 0:
        print(f"Sanity check num: {sanity_check_num}")
        training_steps = (len(train_ds) // batch_size) * num_epochs
        print(f"Training step numbers: {training_steps}")
        train_ds = torch.utils.data.Subset(train_ds, range(0, sanity_check_num))
    
    # epoch_num = _find_save_path(f"./ckpt/argument/{model_name}/TH_{threshold}/{distribution_name}")
    # peft_model_id = f"./ckpt/argument/{model_name}/TH_{threshold}/{distribution_name}/{epoch_num}"

    peft_model_id = f"./ckpt/argument/{model_name}/TH_{threshold}/{distribution_name}"
    config = PeftConfig.from_pretrained(peft_model_id)
    # model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    
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
                model_id=peft_model_id, config=config, is_trainable=True
                # finetune_vision_layers     = False, # Turn off for just text!
                # finetune_language_layers   = True,  # Should leave on!
                # finetune_attention_modules = True,  # Attention good for GRPO
                # finetune_mlp_modules       = True,  # SHould leave on always!

                # r = 8,           # Larger = higher accuracy, but might overfit
                # lora_alpha = 32,  # Recommended alpha == r at least
                # lora_dropout = 0.1,
                # bias = "none",
                # random_state = seed,
                # target_modules=["q_proj", "v_proj"]
            )
        else: # Quantized model load
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, attn_implementation='eager')
            model = get_peft_model(model, peft_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # output_dir= f"./ckpt/argument/{model_name}/TH_{threshold}/{distribution_name}"
    output_dir = f"./ckpt/argument_survey/{model_name}/{strategy}_TH_{threshold}/{distribution_name}"
            
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    
    model = model.to(device)
    best_loss = float('inf')
    print(f'best_loss: {best_loss}')
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

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        torch.cuda.empty_cache()

        model.eval()
        eval_loss = 0
        valid_bar = tqdm(valid_dataloader, desc='Validation')
        for step, (input_ids, attention_mask, labels) in enumerate(valid_bar):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # change the loss if loss.item() is nan
            if torch.isnan(loss):
                print("Loss is nan")
                loss = torch.nan_to_num(loss)
            
            valid_bar.set_description(f"Eval loss: {loss.item()}")

            eval_loss += loss.item()
            

        train_epoch_loss = total_loss / len(train_dataloader)
        eval_epoch_loss = eval_loss / len(valid_dataloader)
        print(f'best_loss: {best_loss}')
        print(f'epoch: {epoch+1}, train_epoch_loss: {train_epoch_loss}, eval_epoch_loss: {eval_epoch_loss}')
        
        if eval_epoch_loss < best_loss:
            print(f"Best model saved at epoch {epoch+1}")
            peft_model_id = f"./ckpt/argument_survey/{model_name}/{strategy}_TH_{threshold}/{distribution_name}/epoch_{epoch+1}"
            model.save_pretrained(peft_model_id)
            best_loss = eval_epoch_loss

    return best_loss

if __name__ == '__main__': 
    fire.Fire(main)
    