model_name='gemma3-27b'
model_name_or_path='google/gemma-3-27b-pt'

mapfile -t TARGET_DISTRIBUTIONS < target_distribution_names.txt
GPU_NUM=0

length=${#TARGET_DISTRIBUTIONS[@]}
echo "LENGTH: $length"
start="started"
finish="finished"

python src/preprocessing.py \
    --threshold 3 \
    --distribution_fname data/extreme_distributions.csv \
    --valueEval_fname data/valueEval_10.csv \
    --output_dir data/values
    
for ((i=0; i<${length}; i++)); do
    number=$(($i+1))
    echo "$number/$length ${TARGET_DISTRIBUTIONS[i]} started"

    python src/train_argument.py \
        --distribution_name ${TARGET_DISTRIBUTIONS[i]} \
        --GPU_NUM $GPU_NUM \
        --model_name ${model_name} \
        --model_name_or_path ${model_name_or_path} \
        --train_base_dir data/values \
        --batch_size 1 \
        --learning_rate 2e-4 \
        --num_epochs 1

    python src/train_argument_survey.py \
        --distribution_name ${TARGET_DISTRIBUTIONS[i]} \
        --GPU_NUM $GPU_NUM \
        --model_name ${model_name} \
        --model_name_or_path ${model_name_or_path} \
        --argument_generation_dir data/argument_generation/value_split \
        --extreme_distribution_file data/extreme_distributions.csv \
        --batch_size 1 \
        --learning_rate 2e-4 \
        --num_epochs 1
done

