model_name='gemma-3-27b'
model_name_or_path='google/gemma-3-27b-pt'

mapfile -t TARGET_DISTRIBUTIONS < target_distribution_names.txt
GPU_NUM=0

length=${#TARGET_DISTRIBUTIONS[@]}
echo "LENGTH: $length"
start="started"
finish="finished"

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
        --learning_rate 2e-4 \
        --sanity_check_num 0 \
        --num_epochs 3

    # python src/train_argument_survey.py \
    #     --distribution_name ${TARGET_DISTRIBUTIONS[i]} \
    #     --GPU_NUM $GPU_NUM \
    #     --model_name ${model_name} \
    #     --model_name_or_path ${model_name_or_path} \
    #     --argument_generation_dir data/argument_generation/value_split \
    #     --extreme_distribution_file data/extreme_distributions.csv \
    #     --batch_size 1 \
    #     --learning_rate 2e-4 \
    #     --num_epochs 1
done

