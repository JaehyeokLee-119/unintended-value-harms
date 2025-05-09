
mapfile -t TARGET_DISTRIBUTIONS < target_distribution_names.txt

length=${#TARGET_DISTRIBUTIONS[@]}
echo "LENGTH: $length"
start="started"
finish="finished"

# dataset_names=('rtp' 'holisticbiasr' 'HEx-PHI' 'beavertails')
dataset_names=('rtp')

start=0
end=${length}

for ((i=${start}; i<${end}; i++)); do
    for dataset_name in "${dataset_names[@]}"; do
        number=$(($i+1))
        echo "$number/$length ${TARGET_DISTRIBUTIONS[i]} started"

        CUDA_VISIBLE_DEVICES=$GPU_NUM python src/eval_RTP.py \
            --data_file /hdd/hjl8708/VIM/source/ckpt_g3_pt_27b/argument_survey/argument_survey/gemma-3-27b-pt-bnb-4bit/min_TH_3/${TARGET_DISTRIBUTIONS[i]}/results/rtp_results.json \
            --result_file /hdd/hjl8708/VIM/source/ckpt_g3_pt_27b/argument_survey/argument_survey/gemma-3-27b-pt-bnb-4bit/min_TH_3/${TARGET_DISTRIBUTIONS[i]}/results/rtp_eval_results.json \
            --num_threads 50
    done
done
