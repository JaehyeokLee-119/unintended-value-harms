CLOSE_VALUE_LIST_HED=(
    'Hed'
    # 'close_Hed'
    # 'close_Hed_2'
    # 'close_Hed_3'
    # 'close_Hed_4'
    # 'close_Hed_5'
    # 'close_Hed_6'
    # 'close_Hed_7'
    # 'close_Hed_8'
    # 'close_Hed_9'
    # 'close_Hed_10'
)

# Script for value-alignment fine-tuning 
# base model: llama2-7B
# values: 11 value distributions prioritizng Hedonism

cd /hdd/hjl8708/VIM/source/src

model_name='llama2'
model_name_or_path='meta-llama/Llama-2-7b-hf'

mapfile -t TARGET_DISTRIBUTIONS < target_distribution_names.txt
GPU_NUM=3

length=${#TARGET_DISTRIBUTIONS[@]}
echo "$length"
start="started"
finish="finished"

# python preprocessing.py \
#     --threshold 3 \
#     --distribution_fname ../data/extreme_distributions.csv \
#     --valueEval_fname ../data/valueEval_10.csv \
#     --output_dir ../data/values

export HF_HOME='/hdd/hjl8708/saved_models'
for ((i=0; i<${length}; i++)); do
    number=$(($i+1))
    text="$number/$length ${TARGET_DISTRIBUTIONS[i]} started"
    echo $text

    export CUDA_VISIBLE_DEVICES=0,1,2,3

    for TARGET_DISTRIBUTION in ${TARGET_DISTRIBUTIONS[@]}; do
        HF_HOME='/hdd/hjl8708/saved_models' python train_argument.py \
            --distribution_name ${TARGET_DISTRIBUTION} \
            --GPU_NUM $GPU_NUM \
            --model_name ${model_name} \
            --model_name_or_path ${model_name_or_path} \
            --train_base_dir ../data/values \
            --batch_size 8

        HF_HOME='/hdd/hjl8708/saved_models' python train_argument_survey.py \
            --distribution_name ${TARGET_DISTRIBUTION} \
            --GPU_NUM $GPU_NUM \
            --model_name ${model_name} \
            --model_name_or_path ${model_name_or_path} \
            --argument_generation_dir ../data/argument_generation/value_split \
            --batch_size 8
    done
done