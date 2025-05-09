
mapfile -t TARGET_DISTRIBUTIONS < target_distribution_names.txt

length=${#TARGET_DISTRIBUTIONS[@]}
echo "LENGTH: $length"
start="started"
finish="finished"

ckpt_home_path=/hdd/hjl8708/VIM/ckpt/argument_survey/gemma3-1b/min_TH_3
# ckpt_home_path=/hdd/hjl8708/VIM/source/ckpt/argument_survey/gemma3-4b/min_TH_3

# dataset_names=('rtp' 'holisticbiasr' 'HEx-PHI' 'beavertails')
dataset_names=('rtp')

start=0
end=${length}

# GPU_NUM=0
# start=1
# end=27

# GPU_NUM=1
# start=28
# end=41

GPU_NUM=2
start=42
end=55

for ((i=${start}; i<${end}; i++)); do
    for dataset_name in "${dataset_names[@]}"; do
        number=$(($i+1))
        echo "$number/$length ${TARGET_DISTRIBUTIONS[i]} started"
        ckpt_path=${ckpt_home_path}/${TARGET_DISTRIBUTIONS[i]}

        CUDA_VISIBLE_DEVICES=$GPU_NUM python src/inference.py \
            --home_directory . \
            --batch_size 150 \
            --peft_path $ckpt_path \
            --dataset_name $dataset_name
    done
done


for ((i=0; i<${length}; i++)); do
    for dataset_name in "${dataset_names[@]}"; do
        number=$(($i+1))
        echo "$number/$length ${TARGET_DISTRIBUTIONS[i]} started"
        ckpt_path=${ckpt_home_path}/${TARGET_DISTRIBUTIONS[i]}

        CUDA_VISIBLE_DEVICES=$GPU_NUM python src/inference.py \
            --home_directory . \
            --batch_size 300 \
            --peft_path $ckpt_path \
            --dataset_name $dataset_name
    done
done
