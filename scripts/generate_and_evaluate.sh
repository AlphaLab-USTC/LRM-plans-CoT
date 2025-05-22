#!/bin/bash

# Default parameter values
model_path=${1:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
vote_num=${2:-8}
dataname=${3:-"MATH500"}
tensor_parallel_size=${4:-4}
steering_strength=${5:-0.0}
steering_vector_type=${6:-"correct"}

model_basename=$(basename "$model_path")
base_save_path="./Data/Eval/${dataname}/${model_basename}"
overall_trend_save_path="${base_save_path}/overall_trend_results_${steering_vector_type}_vote_num${vote_num}.json"
generation_save_path="${base_save_path}/${dataname}-${model_basename}_${steering_strength}_${steering_vector_type}_eval_vote_num${vote_num}.json"
data_path="./Data/Questions/${dataname}.json"

declare -A model_name_map
model_name_map["Qwen2.5-1.5B-Instruct"]="DeepSeek-R1-Distill-Qwen-1.5B"
model_name_map["Qwen2.5-7B-Instruct"]="DeepSeek-R1-Distill-Qwen-7B"
model_name_map["Qwen2.5-14B-Instruct"]="DeepSeek-R1-Distill-Qwen-14B"
model_name_map["Qwen2.5-32B-Instruct"]="DeepSeek-R1-Distill-Qwen-32B"


# Use mapping to obtain the corresponding name.
mapped_name="${model_name_map[$model_basename]:-$model_basename}"  # If the mapping does not exist, return the original name.

steering_vector_path="./Assets/MATH/${mapped_name}/mean_steering_vectors_${steering_vector_type}.npy"


echo "Running generator.py with following parameters:"
echo "Data path: $data_path"
echo "Model path: $model_path"
echo "Generation save path: $generation_save_path"
echo "Vote num: $vote_num"
echo "Dataset name: $dataname"
echo "Tensor parallel size: $tensor_parallel_size"
echo "Steering vector path: $steering_vector_path"
echo "Steering strength: $steering_strength"

python generate_and_evaluate.py \
  --dataname "$dataname" \
  --data_path "$data_path" \
  --model_path "$model_path" \
  --vote_num "$vote_num" \
  --tensor_parallel_size "$tensor_parallel_size" \
  --base_save_path "$base_save_path" \
  --generation_save_path "$generation_save_path" \
  --overall_trend_save_path "$overall_trend_save_path" \
  --steering_vector_path "$steering_vector_path" \
  --steering_strength "$steering_strength" \