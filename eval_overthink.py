#%%
import pandas as pd
import torch
import numpy as np
import os
import os.path as op
from tqdm import tqdm
from jinja2 import Template
# %%
# Read linear regression weights.
model_name_list = ['DeepSeek-R1-Distill-Qwen-1.5B', 'DeepSeek-R1-Distill-Qwen-7B', 
                   'DeepSeek-R1-Distill-Qwen-14B', 'DeepSeek-R1-Distill-Qwen-32B', 'QwQ-32B']
# model_name_list = ['DeepSeek-R1-Distill-Qwen-1.5B', 'DeepSeek-R1-Distill-Qwen-7B']
vanilla_mean_list = []
overthink_mean_list = []


# %%
for model_name in model_name_list:
    print(model_name)
    regresssion_coef_path = "Assets/MATH/" + model_name + "/regression_coef.npy"
    regression_coef = np.load(regresssion_coef_path, allow_pickle=True).item()
    coef = regression_coef['coef']
    intercept = regression_coef['intercept']

    # Vanilla prediction
    prediction_list = []
    for layer_idx in range(len(coef)):
        embeds = np.load("Data/Representation/alpacaeval/" + model_name + "/embeds_" + str(layer_idx) + ".npy")
        # print(f"Layer {layer_idx}:")
        # print(embeds.shape)
        prediction = embeds @ coef[layer_idx] + intercept[layer_idx]
        # print(prediction.mean())
        prediction_list.append(prediction.mean())
    mean_prediction = np.mean(prediction_list)
    print(f"Vanilla prediction: {mean_prediction}")
    
    # Overthink prediction
    overthink_prediction_list = []
    for layer_idx in range(len(coef)):
        embeds = np.load("Data/Representation/alpacaeval_overthink_attack/" + model_name + "/embeds_" + str(layer_idx) + ".npy")
        prediction = embeds @ coef[layer_idx] + intercept[layer_idx]
        overthink_prediction_list.append(prediction.mean())
    mean_overthink_prediction = np.mean(overthink_prediction_list)
    print(f"Overthink prediction: {mean_overthink_prediction}")
    print("")
    vanilla_mean_list.append(mean_prediction)
    overthink_mean_list.append(mean_overthink_prediction)
# %%
print(vanilla_mean_list)
print(overthink_mean_list)
# %%
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

# Font settings
font_path = 'Assets/Times New Roman Bold.ttf'
font_prop = font_manager.FontProperties(fname=font_path)
font_prop_large = font_manager.FontProperties(fname=font_path, size=28)
f_size = 28

model_names = model_name_list
xtick_names = ['R1-Distill-Qwen-1.5B', 'R1-Distill-Qwen-7B', 'R1-Distill-Qwen-14B', 'R1-Distill-Qwen-32B', 'QwQ-32B']
bar_width = 0.25
x = np.arange(len(model_names))

fig, ax = plt.subplots(figsize=(20, 8))

# Bar chart
b1 = ax.bar(x - bar_width/2, vanilla_mean_list, width=bar_width, 
            color="#5DADE2", # '#c4d7ef', 
            hatch='x', edgecolor='white', linewidth=0, label='Vanilla Questions')
b2 = ax.bar(x + bar_width/2, overthink_mean_list, width=bar_width, 
            color="#21618C", #'#f5ac89', 
            hatch='/', edgecolor='white', linewidth=0, label='Overthink Questions')

# Y-axis
ax.set_ylabel('Predicted Token Number', fontproperties=font_prop, fontsize=f_size, fontweight='bold')
ax.set_ylim(0, max(max(vanilla_mean_list), max(overthink_mean_list)) * 1.2)

# x-axis
ax.set_xticks(x)
ax.set_xticklabels(xtick_names, rotation=20)
ax.tick_params(axis='x', labelsize=f_size)
for label in ax.get_xticklabels():
    label.set_fontproperties(font_prop)
    label.set_fontsize(f_size)
    label.set_fontweight('bold')

ax.tick_params(axis='y', labelsize=f_size)
for label in ax.get_yticklabels():
    label.set_fontproperties(font_prop)
    label.set_fontsize(f_size)
    label.set_fontweight('bold')

# Border
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)

# Legend
# legend_elements = [
#     Patch(facecolor='#3A76A3', label='Vanilla Questions'),
#     Patch(facecolor='white', edgecolor='black', hatch='x', label='Overthink Questions')
# ]
# plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.18), 
#            ncol=2, frameon=False, fontsize=f_size, prop=font_prop_large)

plt.legend(prop=font_prop_large, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)

plt.tight_layout()
plt.savefig("Assets/overthink_detection.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
