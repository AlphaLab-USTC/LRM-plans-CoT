#!/bin/bash

# Default parameter values
model_path=${1:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
vote_num=${2:-8}
dataname=${3:-"math_500"}
tensor_parallel_size=${4:-4}

model_basename=$(basename "$model_path")
save_path="./Data_gen/${model_basename}_${dataname}_vote_num${vote_num}.json"
data_path="Data/math/${dataname}.json"

echo "Running generator.py with following parameters:"
echo "Data path: $data_path"
echo "Model path: $model_path"
echo "Save path: $save_path"
echo "Vote num: $vote_num"
echo "Dataset name: $dataname"
echo "Tensor parallel size: $tensor_parallel_size"

python generator.py \
  --data_path "$data_path" \
  --model_path "$model_path" \
  --save_path "$save_path" \
  --vote_num "$vote_num" \
  --tensor_parallel_size "$tensor_parallel_size"