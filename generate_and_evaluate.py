from typing import Any, Dict, List

import numpy as np
# import ray
from packaging.version import Version
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoConfig
from jinja2 import Template
from datasets import load_dataset
import json
import fire
import os
import re

# assert Version(ray.__version__) >= Version("2.22.0"), "Ray version must be at least 2.22.0"
from vllm import ModelRegistry
from steer_qwen2_vllm import SteerQwen2ForCausalLM
from tqdm import tqdm
from math_verify import parse, verify
from mathruler.grader import extract_boxed_content, grade_answer

def grade_math_answer(llm_final_answer, gt_answer):
    llm_final_answer = parse(f"${llm_final_answer}$")
    if llm_final_answer is None:
        return  0.0  # Cannot even parse anything.
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = verify(llm_final_answer, parse(f"${gt_answer}$"))
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= verify(llm_final_answer, parse(f"${gt}$"))
    if is_correct:
        return 1.0  # Correctness reward.
    else:
        return 0.0
    
def grade_gpqa_answer(llm_final_answer, gt_answer):
    if llm_final_answer in gt_answer:
        return 1.0
    else:
        return 0.0

def extract_choice_once_fail(text):
    # Match (A), A), A., A:, A, **A) and so on.
    match = re.findall(
        r"(?:correct answer is|Answer[:：]?)\s*(?:\*\*)?[\(\[]?([A-E])[\)\]\.\s]?",
        text, re.IGNORECASE
    )
    if match:
        return match[-1].upper()  # Take the last match and convert it to uppercase.
    # Bottom line: Directly find independent uppercase letters A-E.
    match2 = re.findall(r"\b([A-E])\b", text)
    if match2:
        return match2[-1].upper()
    return "None"


def generate_and_evaluate(
    model_path="model/DeepSeek-R1-Distill-Qwen-7B",
    dataname="MATH500",
    data_path="",
    base_save_path="",
    generation_save_path="",
    overall_trend_save_path="",
    batch_size=64,  # Increase the batch_size to improve throughput.
    vote_num=8,
    tensor_parallel_size=1,  # Use tensor parallel.
    max_tokens=16384,
    steering_vector_path="Empty",
    steering_strength=0.0,
):    
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)
    #@ Check whether need to steer
    if steering_vector_path != "Empty": #@ Only steer when steering_vector_path is not None
        ModelRegistry.register_model("Qwen2ForCausalLM", SteerQwen2ForCausalLM)
        print("Finish registering model SteerQwen2ForCausalLM")
        
    #@ Create sampling params
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=max_tokens,
        n=vote_num,
    )

    #@ Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    template_jinja = """\
    Please reason step by step, and put your final answer within \boxed{}
    This is the problem:
    {{prompt}}
    """
    prompt_template = Template(template_jinja)
    
    #@ Create VLLM instance
    def create_llm(steering_strength, steering_vector_path, model_path, tensor_parallel_size=1):
        #@ Clear old environment variables
        if "steering_strength" in os.environ:
            del os.environ["steering_strength"]

        #@ Set new environment variables
        config = AutoConfig.from_pretrained(model_path)
        steering_strength_list = [steering_strength] * config.num_hidden_layers
        print(f"Set steering_strength_list to: {steering_strength_list}")
        os.environ["steering_strength_list"] = ",".join(map(str, steering_strength_list))
            
        if "steering_vector_path" in os.environ:
            del os.environ["steering_vector_path"]
        
        print(f"Set steering_vector_path to: {steering_vector_path}")
        print(type(steering_vector_path))
        os.environ["steering_vector_path"] = steering_vector_path
        
        #@ Force garbage collection, Avoid Heisenbug
        import gc
        gc.collect()
        
        #@ Create new LLM instance
        return LLM(model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                dtype="bfloat16",
                max_model_len=max_tokens,
                gpu_memory_utilization=0.9)

    llm = create_llm(steering_strength, steering_vector_path, model_path, tensor_parallel_size)
    
    #@ Load data
    with open(data_path, 'r') as f:
        question_dataset = json.load(f)
    
    #@ Keep a copy of the original data, ensure the output format is consistent
    original_data = question_dataset.copy()
    
    #@ Preprocess data
    print(f"Preprocessing {len(question_dataset)} datapoints")
    processed_prompts = []
    for datapoint in tqdm(question_dataset):
        problem = datapoint['problem']
        prompt_temp = prompt_template.render(prompt=problem)
        message = [ {
                'role': 'user',
                'content': prompt_temp,
            }
        ]
        prompt_temp = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        processed_prompts.append(prompt_temp)

    print('len(processed_prompts):', len(processed_prompts))
    # print(processed_prompts)

    # #@ Generate texts
    all_generated_texts = []
    for i in range(0, len(processed_prompts), batch_size):
        batch_prompts = processed_prompts[i:i+batch_size]
        
        #@ Print progress
        print(f"Processing batch {i//batch_size + 1}/{(len(processed_prompts) + batch_size - 1) // batch_size}")
        
        #@ Use VLLM to generate texts
        outputs = llm.generate(batch_prompts, sampling_params)
        
        #@ Process output results
        for output in outputs:
            #@ For each output, collect all generated texts
            texts = []
            for o in output.outputs:
                texts.append(o.text)
            all_generated_texts.append(texts)
            
        # break
    
    #@ Add generated texts back to original data, keep the original format
    results_for_saving = []
    # for i, datapoint in enumerate(original_data):
    for i in range(len(all_generated_texts)):
        datapoint = original_data[i]
        #@ Get the ground truth answer
        if 'answer' in datapoint:
            gt_answer = datapoint['answer']
        else:
            gt_answer = extract_boxed_content(datapoint['solution'])
        
        #@ If there are generated texts, add them to the data point
        one_record = datapoint.copy()
        one_record['llm_reasoning'], one_record['llm_answer'], one_record['llm_final_answer'], one_record['is_correct'] = [], [], [], []
        one_record['llm_reasoning_token_num'], one_record['llm_answer_token_num'] = [], []
        
        one_generation = all_generated_texts[i]
        for rollout in one_generation:
            llm_final_answer = extract_boxed_content(rollout) #@ Extract the final answer from the rollout
            if llm_final_answer == "None":
                llm_final_answer = extract_choice_once_fail(rollout)
                
            think_idx = rollout.find('</think>') #@ Find the index of </think>
            if think_idx != -1: #@ If </think> is found
                llm_reasoning = rollout[:think_idx] #@ Extract the reasoning from the rollout
                llm_answer = rollout[think_idx:] #@ Extract the answer from the rollout
            else: #@ If </think> is not found
                llm_reasoning = rollout #@ The reasoning is the whole rollout
                llm_answer = "</think>" #@ The answer is </think>
            llm_reasoning_token_num = len(tokenizer.encode(llm_reasoning)) #@ Calculate the token number of the reasoning
            llm_answer_token_num = len(tokenizer.encode(llm_answer)) #@ Calculate the token number of the answer
            if "GPQA" in dataname:
                is_correct = grade_gpqa_answer(llm_final_answer, gt_answer)
            else:
                is_correct = grade_math_answer(llm_final_answer, gt_answer)
            
            one_record['llm_reasoning'].append(llm_reasoning)
            one_record['llm_answer'].append(llm_answer)
            one_record['llm_final_answer'].append(llm_final_answer)
            one_record['is_correct'].append(is_correct)
            
            one_record['llm_reasoning_token_num'].append(llm_reasoning_token_num)
            one_record['llm_answer_token_num'].append(llm_answer_token_num)
                
        one_record['avg_llm_reasoning_token_num'] = sum(one_record['llm_reasoning_token_num']) / len(one_record['llm_reasoning_token_num'])
        one_record['avg_llm_answer_token_num'] = sum(one_record['llm_answer_token_num']) / len(one_record['llm_answer_token_num'])
        one_record['accuracy'] = sum(one_record['is_correct']) / len(one_record['is_correct'])
        
        results_for_saving.append(one_record)
        # else:
            #@ If due to some reason no texts are generated, keep the original data point
            # results.append(datapoint)
    
    #@ Save results
    with open(generation_save_path, 'w') as f:
        json.dump(results_for_saving, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {generation_save_path}")
        
    total_accuracy = sum([datapoint['accuracy'] for datapoint in results_for_saving]) / len(results_for_saving)
    print(f"Total accuracy: {total_accuracy}")
    
    #@ Calculate average length
    total_llm_reasoning_token_num = sum([datapoint['avg_llm_reasoning_token_num'] for datapoint in results_for_saving]) / len(results_for_saving)
    total_llm_answer_token_num = sum([datapoint['avg_llm_answer_token_num'] for datapoint in results_for_saving]) / len(results_for_saving)
    print(f"Average reasoning token num: {total_llm_reasoning_token_num}")
    print(f"Average answer token num: {total_llm_answer_token_num}")
    
    
    #@ Save overall trend data
    if os.path.exists(overall_trend_save_path) and os.path.getsize(overall_trend_save_path) > 0:
        with open(overall_trend_save_path, 'r') as f:
            overall_trend_data = json.load(f)
    else:
        overall_trend_data = []

    new_entry = {
        "strength": steering_strength,
        "total_accuracy": total_accuracy,
        "average_reasoning_token_num": total_llm_reasoning_token_num,
        "average_answer_token_num": total_llm_answer_token_num
    }

    # Check if there is already data with the same strength.
    existing_entry = next((entry for entry in overall_trend_data if entry["strength"] == steering_strength), None)

    if existing_entry:
        # If it exists, then overwrite.
        existing_entry.update(new_entry)
    else:
        # If it does not exist, then add it.
        overall_trend_data.append(new_entry)

    # Save the updated data.
    with open(overall_trend_save_path, 'w') as f:
        json.dump(overall_trend_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    fire.Fire(generate_and_evaluate)