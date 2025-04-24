
model_name='gemma3-1b'
model_name_or_path='google/gemma-3-1b-pt'

GPU_NUM=3

mapfile -t TARGET_DISTRIBUTIONS < target_distribution_names.txt
length=${#TARGET_DISTRIBUTIONS[@]}
echo "LENGTH: $length"
start="started"
finish="finished"

# python preprocessing.py \
#     --threshold 3 \
#     --distribution_fname ../data/extreme_distributions.csv \
#     --valueEval_fname ../data/valueEval_10.csv \
#     --output_dir ../data/values

export HF_HOME='/hdd/hjl8708/saved_models'
i=0
number=$(($i+1))
text="$number/$length ${TARGET_DISTRIBUTIONS[i]} started"
echo $text

export CUDA_VISIBLE_DEVICES=0,1,2,3

# i'th element$
TARGET_DISTRIBUTION=${TARGET_DISTRIBUTIONS[$i]}

lrs=(5e-4 2e-4 5e-5 2e-5 5e-6)

for lr in "${lrs[@]}"; do
    echo "Learning rate: $lr"
    python src/train_argument.py \
        --distribution_name ${TARGET_DISTRIBUTION} \
        --GPU_NUM $GPU_NUM \
        --model_name ${model_name} \
        --model_name_or_path ${model_name_or_path} \
        --train_base_dir data/values \
        --batch_size 36 \
        --learning_rate ${lr} \
        --num_epochs 5

    # python src/train_argument_survey.py \
    #     --distribution_name ${TARGET_DISTRIBUTION} \
    #     --GPU_NUM $GPU_NUM \
    #     --model_name ${model_name} \
    #     --model_name_or_path ${model_name_or_path} \
    #     --argument_generation_dir data/argument_generation/value_split \
    #     --extreme_distribution_file data/extreme_distributions.csv \
    #     --batch_size 36 \
    #     --learning_rate ${lr} \
    #     --num_epochs 5
done
