#%%
import pandas as pd
import torch
import numpy as np
import os
import os.path as op
from tqdm import tqdm
from jinja2 import Template
import json
import argparse

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# print(os.environ["HF_HOME"])
# print(os.environ["HF_HUB_CACHE"])

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='Embed text using language model')
    parser.add_argument('--model_path', type=str, default="model/DeepSeek-R1-Distill-Qwen-7B",
                        help='Path to the language model')
    parser.add_argument('--data_path', type=str, default="Data/Questions/math_all.json",
                        help='Path to the input data JSON file')
    parser.add_argument('--save_path', type=str, default="Data/Representation/MATH/",
                        help='Path to save the embeddings')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing')
    parser.add_argument('--reasoning', type=str2bool, default=True,
                    help='Whether to reason step by step')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_path.split('/')[-1]
    print(args.reasoning)
    if args.reasoning:
        save_dir = op.join(args.save_path, model_name)
    else:
        save_dir = op.join(args.save_path, model_name+"_no_reasoning")
    print(save_dir)
    if not op.exists(save_dir):
        os.makedirs(save_dir)
    args.save_path = save_dir
    with open(args.data_path, 'r') as f:
        datas = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, device_map='auto')
    model = AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16, device_map='auto')

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

# Create a save path.

    
#%%

if args.reasoning:
    template_jinja = """\
Please reason step by step, and put your final answer within \boxed{}
This is the problem:
{{prompt}}
    """
else:
    template_jinja = """\
    {{prompt}}
    """

prompt_template = Template(template_jinja)



#%%
batch_size = args.batch_size
for i in tqdm(range(0, len(datas), batch_size)):
    batch_data = datas[i:i+batch_size]
    inputs = []
    for data in batch_data:
        prompt = data['problem']
        message = [ {
            'role': 'user',
            'content': prompt_template.render(prompt=prompt),
        }
        ]
        input = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs.append(input)
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True)
    if i == 0:
        save_list = [np.array([])]*len(output.hidden_states)
    
    for index in range(len(output.hidden_states)):
        hidden_states = output.hidden_states[index]
        seq_embeds = hidden_states[:, -1, :].detach().cpu().to(torch.float32).numpy()
        if save_list[index].size == 0:
            save_list[index] = seq_embeds
        else:
            save_list[index] = np.concatenate((save_list[index], seq_embeds), axis=0)


#%%
# [np.array([])]*len(output.hidden_states)

for index in range(len(save_list)):
    embeds = save_list[index]
    np.save(op.join(args.save_path, f"embeds_{index}.npy"), embeds)


#%%
len(output.hidden_states)


#%%

print("finished")