#%%
from SteerModels import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaConfig, Qwen2Config, Gemma2Config    
import torch
import numpy as np
import time
from transformers import TextStreamer
from jinja2 import Template
import pandas as pd
torch.manual_seed(42)
np.random.seed(42)
import argparse
import json
from utils import *
from mathruler.grader import extract_boxed_content, grade_answer

#%%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DeepSeek-R1-Distill-Qwen-1.5B',
                        help='model name.')
    parser.add_argument('--dataset', type=str, default='MATH500',
                        help='dataset name.')
    parser.add_argument('--strength', type=float, default=0.0,
                        help='steering strength.')
    parser.add_argument('--rollout_num', type=int, default=8,
                        help='rollout number.')
    parser.add_argument('--question_path', type=str, default='Assets/MATH500/MATH500_Level1.json',
                        help='question path.')
    args, _ = parser.parse_known_args()
    return args, _

args, _ = parse_args()
dataset, model_name, strength, rollout_num, question_path = args.dataset, args.model_name, args.strength, args.rollout_num, args.question_path
# rollout_num = 1
# strength = -0.2

print('dataset: ', dataset, 'model_name: ', model_name, 'strength: ', strength, 'rollout_num: ', rollout_num, 'question_path: ', question_path)
base_asset_path = f"Assets/{dataset}/{model_name}"
DEVICE = "cuda:0"
print("torch.cuda.device_count(): ", torch.cuda.device_count())

selected_problem_file_name = question_path.split('/')[-1].split('.')[0]
print('selected_problem_file_name: ', selected_problem_file_name)

save_path = f"Assets/{dataset}/{model_name}/steering_by_strength/{selected_problem_file_name}"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# %%
#@ Load Questions
with open(question_path, "r", encoding='utf-8') as f:
    data_problems = json.load(f)

#%%

#@ Mapping & Config
models_list = {
    "Qwen-7B-Instruct": ("qwen2.5", SteerQwen2ForCausalLM, Qwen2Config, "model/Qwen2.5-7B-Instruct"),
    "DeepSeek-R1-Distill-Qwen-1.5B": ("qwen2.5", SteerQwen2ForCausalLM, Qwen2Config, "model/DeepSeek-R1-Distill-Qwen-1.5B"),
    "DeepSeek-R1-Distill-Qwen-7B": ("qwen2.5", SteerQwen2ForCausalLM, Qwen2Config, "model/DeepSeek-R1-Distill-Qwen-7B"),
    "DeepSeek-R1-Distill-Qwen-14B": ("qwen2.5", SteerQwen2ForCausalLM, Qwen2Config, "model/DeepSeek-R1-Distill-Qwen-14B"),
    "DeepSeek-R1-Distill-Qwen-32B": ("qwen2.5", SteerQwen2ForCausalLM, Qwen2Config, "model/DeepSeek-R1-Distill-Qwen-32B"),
    "QwQ-32B": ("qwen2.5", SteerQwen2ForCausalLM, Qwen2Config, "model/Qwen/QwQ-32B"),
}

# for model_name, (_, model_class, config_class, model_id) in models_list.items():
#     print(model_class.__name__)

print('model_name: ', model_name)
_, model_class, config_class, model_id = models_list[model_name]

print('model_id: ', model_id)
config = config_class.from_pretrained(model_id)
# hidden_dim = config.hidden_size
num_layers = config.num_hidden_layers

print(config)

#%%
#@ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

#%%
#@ Read Steering Vectors
print("Loading steering vectors... from ", f"Assets/MATH/{model_name}" + f'/mean_steering_vectors.npy')
steering_vectors = np.load(f"Assets/MATH/{model_name}" + f'/mean_steering_vectors.npy')


apply_steering_indices = [True] * num_layers
layer_wise_strength = [strength] * num_layers
apply_steering_indices[0] = False

steering_vectors = torch.from_numpy(steering_vectors).to(DEVICE)
print(steering_vectors.shape)

print(apply_steering_indices)
#%%
#@ Construct Model
print("Is constructing model...")
model = model_class.from_pretrained(
    model_id,
    device_map=DEVICE,
    steering_vectors=steering_vectors,
    apply_steering_indices=apply_steering_indices,
    strength=layer_wise_strength,
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.pad_token = tokenizer.pad_token



#%%
template_jinja = """\
Please reason step by step, and put your final answer within \boxed{}
This is the problem:
{{prompt}}
"""
prompt_template = Template(template_jinja)



#%%
#@ Generate
generation_list = []
time_start = time.time()
batch_size = 4  # Add the batch_size parameter.

for i in range(0, len(data_problems), batch_size):
    # Process the data of the current batch.
    batch_problems = data_problems[i:i+batch_size]
    batch_inputs = []
    
    # Prepare the input for each question in the batch.
    for problem_dict in batch_problems:
        problem = problem_dict['problem']
        prompt_temp = prompt_template.render(prompt=problem)
        message = [{
            'role': 'user',
            'content': prompt_temp,
        }]
        template_input = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        batch_inputs.append(template_input)
    
    # Batch tokenize
    inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True).to(DEVICE)
    t1 = time.time()
    
    # Batch generation
    outputs = model.generate(**inputs,
                         max_new_tokens=16384,
                         do_sample=True,
                         temperature=0.6,
                         top_p=0.95,
                         num_return_sequences=rollout_num,
                         streamer=None  # Do not use streamer during batch processing.
                         )
    t2 = time.time()
    print(f"Batch {i//batch_size + 1} time cost: {t2 - t1}")
    
    # Process the generated results.
    all_generated_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Reorganization results, because each question has rollout_num answers.
    for j, problem_dict in enumerate(batch_problems):
        one_generation_dict = {}
        one_generation_dict['problem'] = problem_dict['problem']
        one_generation_dict['answer'] = str(problem_dict['answer'])
        one_generation_dict['reasoning'] = []
        one_generation_dict['final_answer'] = []
        one_generation_dict['reasoning_length'] = []
        
        # Obtain all generated results for the current question.
        current_answers = all_generated_answers[j * rollout_num:(j + 1) * rollout_num]
        for generated_answer in current_answers:
            think_index = generated_answer.find('<think>')
            if think_index != -1:
                processed_answer = generated_answer[think_index + len('<think>'):]
            else:
                processed_answer = generated_answer
                
            one_generation_dict['reasoning'].append(processed_answer)
            final_answer = extract_boxed_content(processed_answer)
            one_generation_dict['final_answer'].append(final_answer)
            one_generation_dict['reasoning_length'].append(len(processed_answer))
            
        generation_list.append(one_generation_dict)
    
    # Save the current progress.
    with open(save_path + f'/steering_generation_{model_name}_{strength}.json', 'w') as f:
        json.dump(generation_list, f, indent=4)
    
    print(f"Processed {min(i + batch_size, len(data_problems))}/{len(data_problems)} problems")
    # if i > 19:
    #     break
    # break  # If testing is needed, this line can be kept.

time_end = time.time()
print(f"Whole time cost: {time_end - time_start} seconds")


#%%

# print("Finish loading model")

# template_jinja = """\
# Please reason step by step, and put your final answer within \boxed{}
# This is the problem:
# {{prompt}}
# """
# prompt_template = Template(template_jinja)

# problem = "What is your name?"
# # problem = "1+1=?"
# # problem = "Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper)."

# prompt_temp = prompt_template.render(prompt=problem)

# message = [ {
#         'role': 'user',
#         'content': prompt_temp,
#     }
# ]
# template_input = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

# inputs = tokenizer(template_input, return_tensors="pt").to(DEVICE)

# Test
# t1 = time.time()
# streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
# output = model.generate(**inputs, max_new_tokens=8196, streamer=streamer)
# t2 = time.time()

# print("time cost: ", t2 - t1)
# print("length of output: ", len(output[0]))
# %%
