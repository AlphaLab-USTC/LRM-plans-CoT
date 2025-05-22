#%%
import pandas as pd
import numpy as np
import os
import os.path as op
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams, font_manager

#%%
# font_path = 'Assets/Times New Roman.ttf'

# Set the font to Times New Roman.
font_path = 'Assets/Times New Roman Bold.ttf'
font_prop = font_manager.FontProperties(fname=font_path)

# rcParams['font.family'] = font_prop.get_name()
model_name_list = ['DeepSeek-R1-Distill-Qwen-1.5B', 'DeepSeek-R1-Distill-Qwen-7B', 'DeepSeek-R1-Distill-Qwen-14B', 'DeepSeek-R1-Distill-Qwen-32B', 'QwQ-32B']
# %%
performance_math_pre = [
    0.9159676744186046,
    0.9622093023255814,
    0.9505813953488372,
    0.9418604651162791,
    0.9651162790697675
]

performance_math_post = [
    0.936046511627907,
    0.9622093023255814,
    0.9447674418604651,
    0.9505813953488372,
    0.9534883720930233
]
performance_mmlu_pre = [
    0.48,   # DeepSeek-R1-Distill-Qwen-1.5B
    0.58,   # DeepSeek-R1-Distill-Qwen-7B
    0.75,   # DeepSeek-R1-Distill-Qwen-14B
    0.82,   # DeepSeek-R1-Distill-Qwen-32B
    0.82    # QwQ-32B
]
performance_mmlu_post = [
    0.53,  # DeepSeek-R1-Distill-Qwen-1.5B
    0.58,  # DeepSeek-R1-Distill-Qwen-7B
    0.73,  # DeepSeek-R1-Distill-Qwen-14B
    0.81,  # DeepSeek-R1-Distill-Qwen-32B
    0.82   # QwQ-32B
]

# %%
token_math_pre = [
    2025.6744186046512,
    1789.639534883721,
    1668.9912790697674,
    1652.3372093023256,
    1626.7761627906978
]

token_math_post = [
    681.0581395348837,
    563.8430232558139,
    707.7616279069767,
    179.5843023255814,
    1541.6424418604652
]
token_mmlu_pre = [
    1315.21,  # DeepSeek-R1-Distill-Qwen-1.5B
    965.52,   # DeepSeek-R1-Distill-Qwen-7B
    807.54,   # DeepSeek-R1-Distill-Qwen-14B
    772.64,   # DeepSeek-R1-Distill-Qwen-32B
    1026.86   # QwQ-32B
]
token_mmlu_post = [
    641.2,   # DeepSeek-R1-Distill-Qwen-1.5B
    308.42,  # DeepSeek-R1-Distill-Qwen-7B
    425.12,  # DeepSeek-R1-Distill-Qwen-14B
    545.15,  # DeepSeek-R1-Distill-Qwen-32B
    634.15   # QwQ-32B
]

# %%

# Model name
xtick_names = ['R1-Distill-Qwen-1.5B', 'R1-Distill-Qwen-7B', 'R1-Distill-Qwen-14B', 'R1-Distill-Qwen-32B', 'QwQ-32B']
f_size = 28

# Bar chart data
math_vanilla = token_math_pre
math_overthink = token_math_post
mmlu_vanilla = token_mmlu_pre
mmlu_overthink = token_mmlu_post

# Line chart data
math_perf_vanilla = performance_math_pre
math_perf_overthink = performance_math_post
mmlu_perf_vanilla = performance_mmlu_pre
mmlu_perf_overthink = performance_mmlu_post

bar_width = 0.18
x = np.arange(len(xtick_names))

fig, ax1 = plt.subplots(figsize=(20, 8))

# Bar chart
b1 = ax1.bar(x - 1.5*bar_width, math_vanilla, width=bar_width, color='#3A76A3', label='MATH Vanilla')
b2 = ax1.bar(x - 0.5*bar_width, math_overthink, width=bar_width, color='#3A76A3', hatch='x', edgecolor='white', linewidth=0, label='MATH Overthink')
b3 = ax1.bar(x + 0.5*bar_width, mmlu_vanilla, width=bar_width, color='#c4d7ef', label='MMLU Vanilla')
b4 = ax1.bar(x + 1.5*bar_width, mmlu_overthink, width=bar_width, color='#c4d7ef', hatch='x', edgecolor='white', linewidth=0, label='MMLU Overthink')

ax1.set_ylabel('Reasoning Token Number', fontsize=f_size, fontweight='bold', fontproperties=font_prop)
ax1.set_xticks(x)
ax1.set_xticklabels(xtick_names, rotation=20)
ax1.tick_params(axis='x', labelsize=f_size)

for label in ax1.get_xticklabels():
    label.set_fontproperties(font_prop)
    label.set_fontsize(24)
    label.set_fontweight('bold')

ax1.tick_params(axis='y', labelsize=f_size)
for label in ax1.get_yticklabels():
    label.set_fontproperties(font_prop)
    label.set_fontsize(f_size)
    label.set_fontweight('bold')

ax1.spines['left'].set_linewidth(3)
ax1.spines['bottom'].set_linewidth(3)
ax1.spines['right'].set_linewidth(3)
ax1.spines['top'].set_linewidth(3)
ax1.set_ylim(0, 3500)

# Second y-axis: accuracy
ax2 = ax1.twinx()
l1, = ax2.plot(x, math_perf_vanilla, color='#3A76A3', marker='o', linewidth=2, label='MATH Vanilla Accuracy', markerfacecolor='white', markeredgewidth=2)
l2, = ax2.plot(x, math_perf_overthink, color='#3A76A3', marker='o', linewidth=2, linestyle='--', label='MATH Overthink Accuracy', markerfacecolor='white', markeredgewidth=2)
l3, = ax2.plot(x, mmlu_perf_vanilla, color='#c4d7ef', marker='s', linewidth=2, label='MMLU Vanilla Accuracy', markerfacecolor='white', markeredgewidth=2)
l4, = ax2.plot(x, mmlu_perf_overthink, color='#c4d7ef', marker='s', linewidth=2, linestyle='--', label='MMLU Overthink Accuracy', markerfacecolor='white', markeredgewidth=2)
ax2.set_ylabel('Accuracy', fontsize=f_size, fontweight='bold', fontproperties=font_prop)
ax2.tick_params(axis='y', labelsize=f_size)
for label in ax2.get_yticklabels():
    label.set_fontproperties(font_prop)
    label.set_fontsize(f_size)
    label.set_fontweight('bold')
ax2.set_ylim(0, 1.05)

# Legend merge
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Create a new legend item.
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

font_prop_large = font_manager.FontProperties(fname=font_path, size=f_size)

legend_elements = [
    Patch(facecolor='#3A76A3', label='MATH500-Level 1'),
    Patch(facecolor='#c4d7ef', label='MMLU'),
    
    Patch(facecolor='white', edgecolor='black', label='Token Number before steering'),
    Patch(facecolor='white', edgecolor='black', hatch='x', label='Token Number after steering'),
    
    Line2D([0], [0], color='black', linestyle='-', label='Accuracy before steering'),
    Line2D([0], [0], color='black', linestyle='--', label='Accuracy after steering'),
]

# Modify the legend to display in two lines.
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.3), 
          ncol=3, frameon=False, fontsize=f_size, prop=font_prop_large)

plt.tight_layout()
save_path = 'Assets/efficient_reasoning.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()



# %%
