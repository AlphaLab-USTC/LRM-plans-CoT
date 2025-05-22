model_paths=(
    # "model/Qwen2.5-3B-instruct"
    # "model/Qwen2.5-7B"
    # "model/Qwen2.5-7B-Instruct"
    # "model/Llama-3.1-8B-Instruct"
    # "model/DeepSeek-R1-Distill-Qwen-1.5B"
    "model/DeepSeek-R1-Distill-Qwen-7B"
    # "model/DeepSeek-R1-Distill-Qwen-14B"
    # "model/DeepSeek-R1-Distill-Qwen-32B"
    # "model/Qwen/QwQ-32B"
)

vote_num=8
# dataname='MMLU'
dataname='GPQA_diamond'
# dataname="MATH500"
# dataname="AIME2024"
# dataname="OlympiadBench"
# dataname="Minerva"
tensor_parallel_size=2
steering_strengths=(0.0 -0.05 -0.1 -0.15 -0.2 0.05 0.1 0.15 0.2)
# steering_strengths=(0.0 -0.05 -0.1 -0.15 -0.2)
# steering_strengths=(-0.05)
# steering_strengths=(0.15)
steering_vector_type="correct"


for i in "${!model_paths[@]}"; do
    model_path="${model_paths[i]}"
    model_name=$(basename "$model_path")
    echo "Model name: $model_name"
    for steering_strength in "${steering_strengths[@]}"; do
        echo "Waiting for previous process to fully exit..."
        sleep 10  # Wait for 10 seconds to ensure that the previous process has completely exited.
        bash scripts/generate_and_evaluate.sh "$model_path" "$vote_num" "$dataname" "$tensor_parallel_size" "$steering_strength" "$steering_vector_type"
    done
done
