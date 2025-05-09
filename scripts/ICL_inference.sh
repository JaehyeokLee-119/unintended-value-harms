
mapfile -t TARGET_DISTRIBUTIONS < target_distribution_names.txt

length=${#TARGET_DISTRIBUTIONS[@]}
echo "LENGTH: $length"
start="started"
finish="finished"

# model_name='llama-2-7b'
# model_name_or_path='meta-llama/Llama-2-7b-hf'

# model_name='gemma-3-1b'
# model_name_or_path='google/gemma-3-1b-pt'

# model_name='gemma-3-4b'
# model_name_or_path='google/gemma-3-4b-pt'

# model_name='gemma-3-12b'
# model_name_or_path='google/gemma-3-12b-pt'

# model_name='gemma-3-27b-pt-bnb-4bit'
# model_name_or_path='unsloth/gemma-3-27b-pt-bnb-4bit'

model_names=('llama-2-7b' 'gemma-3-1b' 'gemma-3-4b' 'gemma-3-12b' 'gemma-3-27b-pt-bnb-4bit')
model_name_or_paths=('meta-llama/Llama-2-7b-hf' 'google/gemma-3-1b-pt' 'google/gemma-3-4b-pt' 'google/gemma-3-12b-pt' 'unsloth/gemma-3-27b-pt-bnb-4bit')

dataset_names=('holisticbiasr' 'rtp')

GPU_NUM=3
start=$((14 * GPU_NUM)) # 4분할
end=$((14 * (GPU_NUM + 1) - 1))
echo $start
echo $end 

for dataset_name in "${dataset_names[@]}"; do
    # CUDA_VISIBLE_DEVICES=$GPU_NUM python src/inference_vanilla.py \
    #     --dataset_name $dataset_name \
    #     --base_model_id $model_name_or_path \
    #     --home_directory . \
    #     --batch_size 300 \
    #     --result_path ./results/${model_name}-VANILLA/$dataset_name \

    for i in "${!model_names[@]}"; do
        model_name=${model_names[i]}
        model_name_or_path=${model_name_or_paths[i]}
        echo "model: $model_name"
        echo "$start/$length ${TARGET_DISTRIBUTIONS[i]} started"

        for ((i=${start}; i<${end}; i++)); do
            number=$(($i+1))
            echo "dataset: $dataset_name"
            echo "$number/$length ${TARGET_DISTRIBUTIONS[i]} started"

            CUDA_VISIBLE_DEVICES=$GPU_NUM python src/inference_ICL.py \
                --dataset_name $dataset_name \
                --base_model_id $model_name_or_path \
                --home_directory . \
                --batch_size 300 \
                --distribution_file_path ./data/extreme_distributions.csv \
                --result_path ./results/${model_name}-ICL/$dataset_name-${TARGET_DISTRIBUTIONS[i]} \
                --value_dsitribution_name ${TARGET_DISTRIBUTIONS[i]}
        done
    done
done
