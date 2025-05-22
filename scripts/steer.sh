
export CUDA_VISIBLE_DEVICES=5

# Each model path on a separate line, ensuring correct separation.
model_name="DeepSeek-R1-Distill-Qwen-1.5B"
# strength=(0 -0.05 -0.1 -0.15 -0.2 0.05 0.1 0.15 0.2)
strength=(-0.15 -0.2 0.05 0.1 0.15 0.2)
rollout_num=1
# question_path='Assets/MATH500/MATH500_Level1.json'
# question_path='Assets/MATH500/MATH500_Level3.json'
question_path='Assets/MATH500/MATH500_Level5.json'

# Traverse the strength array and process each model.
for s in "${strength[@]}"; do
    echo "Processing model: ${model_name} with strength: ${s}"
    python ApplySteering.py \
        --model_name "${model_name}" \
        --strength "${s}" \
        --rollout_num "${rollout_num}" \
        --question_path "${question_path}"
    # break
done