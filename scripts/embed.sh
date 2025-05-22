

export CUDA_VISIBLE_DEVICES=6

# Each model path on a separate line, ensuring correct separation.
model_paths=(
    # "model/Qwen2.5-3B-instruct"
    # "model/GLM-Z1-9B-0414"
    # "model/Qwen2.5-7B-Instruct"
    # "model/Llama-3.1-8B-Instruct"
    "model/DeepSeek-R1-Distill-Qwen-7B"
    "model/DeepSeek-R1-Distill-Qwen-1.5B"
    "model/DeepSeek-R1-Distill-Qwen-14B"
    "model/DeepSeek-R1-Distill-Qwen-32B"
    "model/Qwen/QwQ-32B"
)
data_path="Data/Questions/alpacaeval_overthink_attack.json"
save_path="Data/Representation/alpacaeval_overthink_attack"
# data_path="Data/Questions/alpacaeval.json"
# save_path="Data/Representation/alpacaeval"
# Traverse the array and process each model.
for model_path in "${model_paths[@]}"; do  # Add quotes to ensure that spaces in the path are handled correctly.
    FALG=True
    while [ $FALG = True ]; do
        echo "Processing model: ${model_path}" 
        python embed.py \
            --reasoning True \
            --model_path "${model_path}" \
            --data_path "${data_path}" \
            --save_path "${save_path}"
        FALG=False
    done
done