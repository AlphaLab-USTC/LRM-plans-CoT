#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

# Each model path on a separate line, ensuring correct separation.
model_names=(
    "DeepSeek-R1-Distill-Qwen-1.5B"
    "DeepSeek-R1-Distill-Qwen-7B"
    "DeepSeek-R1-Distill-Qwen-14B"
    "DeepSeek-R1-Distill-Qwen-32B"
    "QwQ-32B"
)

for model_name in ${model_names[@]}; do
    echo "Analyzing $model_name"
    python AnalyzeOneModel.py --model_name $model_name
    # break
done


