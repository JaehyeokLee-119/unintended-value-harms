# This script is used to train the Gemma model with 27 billion parameters using 4-bit quantization.
model_name='gemma-3-27b-pt-bnb-4bit'
model_name_or_path='unsloth/gemma-3-27b-pt-bnb-4bit'

mapfile -t TARGET_DISTRIBUTIONS < target_distribution_names.txt
GPU_NUM=3

length=${#TARGET_DISTRIBUTIONS[@]}
echo "LENGTH: $length"
start="started"
finish="finished"

for ((i=0; i<${length}; i++)); do
    number=$(($i+1))
    echo "$number/$length ${TARGET_DISTRIBUTIONS[i]} started"

    CUDA_VISIBLE_DEVICES=$GPU_NUM python src/train_argument_4bit.py \
        --distribution_name ${TARGET_DISTRIBUTIONS[i]} \
        --GPU_NUM 0 \
        --model_name ${model_name} \
        --model_name_or_path ${model_name_or_path} \
        --train_base_dir data/values \
        --batch_size 120 \
        --learning_rate 2e-5 \
        --sanity_check_num 0 \
        --num_epochs 3

    CUDA_VISIBLE_DEVICES=$GPU_NUM python src/train_argument_survey_4bit.py \
        --distribution_name ${TARGET_DISTRIBUTIONS[i]} \
        --GPU_NUM 0 \
        --model_name ${model_name} \
        --model_name_or_path ${model_name_or_path} \
        --argument_generation_dir data/argument_generation/value_split \
        --extreme_distribution_file data/extreme_distributions.csv \
        --batch_size 120 \
        --learning_rate 2e-5 \
        --sanity_check_num 0 \
        --num_epochs 3
done



for ((i=0; i<${length}; i++)); do
    number=$(($i+1))
    echo "$number/$length ${TARGET_DISTRIBUTIONS[i]} started"

    python src/train_argument_4bit.py \
        --distribution_name ${TARGET_DISTRIBUTIONS[i]} \
        --GPU_NUM $GPU_NUM \
        --model_name ${model_name} \
        --model_name_or_path ${model_name_or_path} \
        --train_base_dir data/values \
        --batch_size 120 \
        --learning_rate 2e-5 \
        --sanity_check_num 0 \
        --num_epochs 3

    # CUDA_VISIBLE_DEVICES=$GPU_NUM python src/train_argument_survey_4bit.py \
    #     --distribution_name ${TARGET_DISTRIBUTIONS[i]} \
    #     --GPU_NUM $GPU_NUM \
    #     --model_name ${model_name} \
    #     --model_name_or_path ${model_name_or_path} \
    #     --argument_generation_dir data/argument_generation/value_split \
    #     --extreme_distribution_file data/extreme_distributions.csv \
    #     --batch_size 120 \
    #     --learning_rate 2e-5 \
    #     --sanity_check_num 0 \
    #     --num_epochs 3
done



GPU_NUM=1
while true; do
    for ((i=1; i<${length}; i+=2)); do
        number=$(($i + 1))
        echo "$number/$length ${TARGET_DISTRIBUTIONS[i]} started"

        CUDA_VISIBLE_DEVICES=$GPU_NUM python src/train_argument_survey_4bit.py \
            --distribution_name ${TARGET_DISTRIBUTIONS[i]} \
            --GPU_NUM $GPU_NUM \
            --model_name ${model_name} \
            --model_name_or_path ${model_name_or_path} \
            --argument_generation_dir data/argument_generation/value_split \
            --extreme_distribution_file data/extreme_distributions.csv \
            --batch_size 120 \
            --learning_rate 2e-5 \
            --sanity_check_num 0 \
            --num_epochs 3
    done
done
