#!/bin/bash

# Define the model path array.
model_paths=(
    # "model/DeepSeek-R1-Distill-Qwen-1.5B"
    # "model/DeepSeek-R1-Distill-Qwen-7B"
    # "model/DeepSeek-R1-Distill-Qwen-14B"
    "model/DeepSeek-R1-Distill-Qwen-32B"
    "model/Qwen/QwQ-32B"
)

# Define the dataset.
dataname="MATH500"
# dataname="GPQA_diamond"
# dataname="AIME2024"
# dataname="OlympiadBench"
# dataname="Minerva"

# Define the problem file path.
question_path="./Data/Questions/${dataname}.json"

# Define the steering strength array.
# steering_strengths=(0.0 -0.05 -0.1 -0.15 -0.2 0.05 0.1 0.15 0.2)
steering_strengths=(0.0 -0.1 -0.2 0.1 0.2)
# steering_strengths=(0.15)

# Traverse all models.
for i in "${!model_paths[@]}"; do
    model_path="${model_paths[i]}"
    model_name=$(basename "$model_path")
    echo "Processing model: $model_name"
    
    # Traverse all steering strength.
    for steering_strength in "${steering_strengths[@]}"; do
        echo "Running with steering strength: $steering_strength"
        echo "Waiting for previous process to fully exit..."
        sleep 10  # Wait for 10 seconds to ensure that the previous process has completely exited.
        
        # Run CheckThinkLogit.py.
        python CheckThinkLogit.py \
            --model_name "$model_name" \
            --model_path "$model_path" \
            --dataset "$dataname" \
            --strength "$steering_strength" \
            --question_path "$question_path"
            
        echo "Completed run for model $model_name with strength $steering_strength"
    done
done

echo "All runs completed!"
