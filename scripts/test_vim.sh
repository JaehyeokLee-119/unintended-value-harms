
CLOSE_VALUE_LIST=(
    'Hed'
    'close_Hed'
    'close_Hed_2'
    'close_Hed_3'
    'close_Hed_4'
    'close_Hed_5'
    'close_Hed_6'
    'close_Hed_7'
    'close_Hed_8'
    'close_Hed_9'
    'close_Hed_10'
)


length=${#CLOSE_VALUE_LIST[@]}
# length=1 # 
echo "$length"
start="started"
finish="finished"
GPU_NUM=0

for ((i=0; i<${length}; i++)); do
    number=$(($i+1))
    text="$number/$length ${CLOSE_VALUE_LIST[i]} started"
    echo $text

    python inference.py \
        --distribution_name ${CLOSE_VALUE_LIST[i]} \
        --GPU_NUM $GPU_NUM \
        --mode argument_survey \
        --model_name 'llama2' \
        --model_name_or_path 'meta-llama/Llama-2-7b-hf'
done

