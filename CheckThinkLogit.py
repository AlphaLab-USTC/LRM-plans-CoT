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
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

#%%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DeepSeek-R1-Distill-Qwen-1.5B',
                        help='model name.')
    parser.add_argument('--model_path', type=str, default='model/DeepSeek-R1-Distill-Qwen-1.5B',
                        help='model path.')
    parser.add_argument('--dataset', type=str, default='MATH500',
                        help='dataset name.')
    parser.add_argument('--strength', type=float, default=0.0,
                        help='steering strength.')
    parser.add_argument('--question_path', type=str, default='Data/Questions/MATH500.json',
                        help='question path.')
    args, _ = parser.parse_known_args()
    return args, _

args, _ = parse_args()
dataset, model_name, strength, question_path = args.dataset, args.model_name, args.strength, args.question_path
# rollout_num = 1
# strength = -0.2

print('dataset: ', dataset, 'model_name: ', model_name, 'strength: ', strength, 'question_path: ', question_path)
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

# for model_name, (_, model_class, config_class, model_path) in models_list.items():
#     print(model_class.__name__)

print('model_name: ', model_name)
# _, model_class, config_class, model_path = models_list[model_name]
_, model_class, config_class, model_path = models_list[model_name]
model_path = args.model_path

print('model_path: ', model_path)
config = config_class.from_pretrained(model_path)
# hidden_dim = config.hidden_size
num_layers = config.num_hidden_layers

print(config)

#%%
#@ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

#%%
#@ Read Steering Vectors
print("Loading steering vectors... from ", f"Assets/MATH/{model_name}" + f'/mean_steering_vectors.npy')
steering_vectors = np.load(f"Assets/MATH/{model_name}" + f'/mean_steering_vectors.npy')

# strength = -0.25
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
    model_path,
    device_map=DEVICE,
    steering_vectors=steering_vectors,
    apply_steering_indices=apply_steering_indices,
    strength=layer_wise_strength,
    torch_dtype=torch.bfloat16,
    temperature=0.6
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
batch_size = 1  # Add the batch_size parameter.
eos_token = tokenizer.eos_token_id
# random_token_1 = np.random.randint(0, tokenizer.eos_token_id)
# random_token_2 = np.random.randint(0, tokenizer.eos_token_id)
# Randomly select 500 tokens.
random_tokens = np.random.randint(0, tokenizer.eos_token_id, size=2000)
think_token_logits_list = []
think_token_probs_list = []
eos_token_logits_list = []
eos_token_probs_list = []
random_token_logits_list = []
random_token_probs_list = []
last_token_logits_list = []
last_token_probs_list = []
temperature = 0.6

#%%

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
    
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True)
    last_token_logits = output.logits[:, -1, :].to(torch.float32).detach().cpu().numpy()
    last_token_probs = np.exp(last_token_logits / temperature) / np.sum(np.exp(last_token_logits / temperature))
    think_token_logits = last_token_logits[:,151649]
    think_token_probs = np.exp(think_token_logits / temperature) / np.sum(np.exp(last_token_logits / temperature))
    eos_token_logits = last_token_logits[:,eos_token]   
    eos_token_probs = np.exp(eos_token_logits / temperature) / np.sum(np.exp(last_token_logits / temperature))
    random_token_logits = last_token_logits[:,random_tokens]
    random_token_probs = np.exp(random_token_logits / temperature) / np.sum(np.exp(last_token_logits / temperature))
    random_token_logits = np.mean(random_token_logits, axis=-1)
    random_token_probs = np.mean(random_token_probs, axis=-1)
    
    # print(last_token_logits.shape)
    # print(think_token_logits)
    # print(think_token_probs)
    # print(eos_token_logits)
    # print(eos_token_probs)
    # print(random_token_logits)
    # print(random_token_probs)
    # print(random_token_logits.shape)
    # print(random_token_probs.shape)
    if i == 0:
        think_token_logits_list = think_token_logits
        think_token_probs_list = think_token_probs
        eos_token_logits_list = eos_token_logits
        eos_token_probs_list = eos_token_probs
        random_token_logits_list = random_token_logits
        random_token_probs_list = random_token_probs
        last_token_logits_list = last_token_logits
        last_token_probs_list = last_token_probs
    else:
        think_token_logits_list = np.concatenate((think_token_logits_list, think_token_logits), axis=0)
        think_token_probs_list = np.concatenate((think_token_probs_list, think_token_probs), axis=0)
        eos_token_logits_list = np.concatenate((eos_token_logits_list, eos_token_logits), axis=0)
        eos_token_probs_list = np.concatenate((eos_token_probs_list, eos_token_probs), axis=0)
        random_token_logits_list = np.concatenate((random_token_logits_list, random_token_logits), axis=0)
        random_token_probs_list = np.concatenate((random_token_probs_list, random_token_probs), axis=0)
        last_token_logits_list = np.concatenate((last_token_logits_list, last_token_logits), axis=0)
        last_token_probs_list = np.concatenate((last_token_probs_list, last_token_probs), axis=0)
    # if i == 0:
    #     save_list = [np.array([])]*len(output.hidden_states)
    
    # for index in range(len(output.hidden_states)):
    #     hidden_states = output.hidden_states[index]
    #     seq_embeds = hidden_states[:, -1, :].detach().cpu().to(torch.float32).numpy()
    #     if save_list[index].size == 0:
    #         save_list[index] = seq_embeds
    #     else:
    #         save_list[index] = np.concatenate((save_list[index], seq_embeds), axis=0)

    t2 = time.time()
    print(f"Batch {i//batch_size + 1} time cost: {t2 - t1}")
    # break

mean_last_token_logits = np.mean(last_token_logits_list, axis=0)
mean_last_token_probs = np.mean(last_token_probs_list, axis=0)

time_end = time.time()
print(f"Whole time cost: {time_end - time_start} seconds")

print("Average think token logits: ", np.mean(think_token_logits_list))
print("Average think token probs: ", np.mean(think_token_probs_list))
print("Average eos token logits: ", np.mean(eos_token_logits_list))
print("Average eos token probs: ", np.mean(eos_token_probs_list))
print(f"Average random token logits: ", np.mean(random_token_logits_list))
print(f"Average random token probs: ", np.mean(random_token_probs_list))
#%%
# Save four using the same npy file in dict form.
token_logits_probs_dict = {
    "think_token_logits": think_token_logits_list,
    "think_token_probs": think_token_probs_list,
    "eos_token_logits": eos_token_logits_list,
    "eos_token_probs": eos_token_probs_list,
    "random_token_logits": random_token_logits_list,
    "random_token_probs": random_token_probs_list,
    "mean_last_token_logits": mean_last_token_logits,
    "mean_last_token_probs": mean_last_token_probs,
}
np.save(save_path + f'/token_logits_probs_{strength}.npy', token_logits_probs_dict)
#%%