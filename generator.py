from typing import Any, Dict, List

import numpy as np
import ray
from packaging.version import Version
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from jinja2 import Template
from datasets import load_dataset
import json
import fire
import os

assert Version(ray.__version__) >= Version("2.22.0"), "Ray version needs to be at least 2.22.0"
from vllm.lora.request import LoRARequest

def eval(
    lora_path="cot_rec/outputs/steam/steam_star/checkpoint-550",
    model_path="model/DeepSeek-R1-Distill-Qwen-7B",
    data_path="",
    save_path="",
    batch_size=64,  # Increase batch_size to improve throughput.
    vote_num=8,
    tensor_parallel_size=1,  # Use tensor parallel.
    temperature=0.6,
    max_tokens=16384,
    mood='cot',
):
    # Initialize Ray, note that the number of GPUs is no longer specified using num_instances.
    ray.init(num_cpus=64, num_gpus=tensor_parallel_size)
    
    # Create sampling parameters.
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        n=vote_num,
    )
    
    # Load data.
    with open(data_path, 'r') as f:
        eval_pd = json.load(f)
    
    # Prepare the tokenizer and template.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    template_jinja = """\
    Please reason step by step, and put your final answer within \boxed{}
    This is the problem:
    {{prompt}}
    """
    prompt_template = Template(template_jinja)
    
    # Keep a copy of the original data to ensure consistent output format.
    original_data = eval_pd.copy()
    
    # Preprocess data.
    processed_prompts = []
    for datapoint in eval_pd:
        problem = datapoint['problem']
        prompt_temp = prompt_template.render(prompt=problem)
        message = [ {
                'role': 'user',
                'content': prompt_temp,
            }
        ]
        prompt_temp = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        processed_prompts.append(prompt_temp)
    
    
    # Create a VLLM instance using tensor_parallel_size.
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,  # Set up tensor parallelism.
        enable_lora=False,  # If LoRA is needed, change this to True.
        max_model_len=16384,
        gpu_memory_utilization=0.9,  # Increase GPU memory utilization.
        max_num_seqs=batch_size * vote_num,  # Ensure that enough sequences can be processed.
    )
    
    # Batch processing inference
    all_generated_texts = []
    for i in range(0, len(processed_prompts), batch_size):
        batch_prompts = processed_prompts[i:i+batch_size]
        
        # Printing progress
        print(f"Processing batch {i//batch_size + 1}/{(len(processed_prompts) + batch_size - 1) // batch_size}")
        
        # Batch inference using VLLM.
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Process the output results.
        for output in outputs:
            # For each output, collect all generated text.
            texts = []
            for o in output.outputs:
                texts.append(o.text)
            all_generated_texts.append(texts)
    
    # Add the generated text back to the original data, keeping the original format.
    results = []
    for i, datapoint in enumerate(original_data):
        if i < len(all_generated_texts):
            datapoint_copy = datapoint.copy()
            datapoint_copy['reasoning'] = all_generated_texts[i]
            results.append(datapoint_copy)
        else:
            # If the text is not generated for some reason, retain the original data points.
            results.append(datapoint)
    
    # Save the results.
    with open(save_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    fire.Fire(eval)