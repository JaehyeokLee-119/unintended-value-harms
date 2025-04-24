# Script for testing models on different prompting trategy
# OP : input only
# SP : safety prompt
# VP : value prompt
# VPSP : both prompts

base_model_name=llama2
base_model_name_or_path='meta-llama/Llama-2-7b-hf'

target_risk_categories=(adult_content political_campaigning fraud_deception)
pos_values=(self-direction universalism universalism)
distribution_names=(SD_11 Uni_11 Uni_11)

gpu=0
home_directory=''

export CUDA_VISIBLE_DEVICES=${gpu}
for j in {0..3}; do
    for prompt_type in OP VP SP VPSP; do
        python inference_hex.py \
            --base_model_name ${base_model_name} \
            --base_model_name_or_path ${base_model_name_or_path} \
            --distribution_group_name ${distribution_names[j]} \
            --home_directory ${home_directory} \
            --target_risk_category ${target_risk_categories[j]} \
            --eval_dataset_name HEx-PHI \
            --vp_value ${pos_values[j]} \
            --prompt_type ${prompt_type} \
            --max_new_tokens 64 \
            --result_directory results
    done
done