
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from utils import *
from visualization_tools import *
import os
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import argparse
import re

# %%
file_path = "Assets/MATH500/DeepSeek-R1-Distill-Qwen-1.5B/steering_by_strength/MATH500_Level5"

tokenizer = AutoTokenizer.from_pretrained("model/Qwen/QwQ-32B")
# %%
# Check all JSON files in the path.
json_files = [f for f in os.listdir(file_path) if f.endswith('.json')]
strength_list = [float(re.search(r'_(-?\d+\.?\d*)\.json$', fname).group(1)) for fname in json_files]

# %%
json_files
# %%
avg_tokens_reasoning_list, avg_tokens_answer_list, overall_accuracy_list = [], [], []

for file in json_files:
    generation_path = os.path.join(file_path, file)
    data = preprocess_data(generation_path)
    data_information = analyze_token_numbers(
        data=data,
        tokenizer=tokenizer
    )
    # data_information = [d for d in data_information if d['avg_tokens_reasoning'] < 8196]
    data_information_df = process_data_information_frame(data_information)
    avg_tokens_reasoning = data_information_df.avg_tokens_reasoning.mean()
    avg_tokens_answer = data_information_df.avg_tokens_answer.mean()
    print(avg_tokens_reasoning, avg_tokens_answer)
    
    overall_accuracy, level_accuracy, df_wrong, df_accuracy = analyze_model_accuracy(
    data=data,
    accuracy_threshold=0.2,
    samples_per_level=20
    )
    print(overall_accuracy)
    avg_tokens_reasoning_list.append(avg_tokens_reasoning)
    avg_tokens_answer_list.append(avg_tokens_answer)
    overall_accuracy_list.append(overall_accuracy)
    # break
# %%
sorted_indices = sorted(range(len(strength_list)), key=lambda i: strength_list[i])
sorted_strength_list = [strength_list[i] for i in sorted_indices]
sorted_avg_tokens_reasoning_list = [avg_tokens_reasoning_list[i] for i in sorted_indices]
sorted_avg_tokens_answer_list = [avg_tokens_answer_list[i] for i in sorted_indices]
sorted_overall_accuracy_list = [overall_accuracy_list[i] for i in sorted_indices]

print("Sorted Strength List:", sorted_strength_list)
print("Sorted Average Tokens Reasoning List:", sorted_avg_tokens_reasoning_list)
print("Sorted Average Tokens Answer List:", sorted_avg_tokens_answer_list)
print("Sorted Overall Accuracy List:", sorted_overall_accuracy_list)

# %%
# import matplotlib.pyplot as plt
# import numpy as np

# Assumed data (you can replace it with your actual data)
# sorted_strength_list = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]  # Strength
# sorted_avg_tokens_reasoning_list = [750, 1000, 1250, 1500, 1750, 2000, 1750, 1500]  # Reasoning Token Numbers
# sorted_avg_tokens_answer_list = [0, 0, 0, 0, 0, 250, 250, 250]  # Answer Token Numbers
# sorted_overall_accuracy_list = [0.5, 0.5, 0.5, 0.7, 0.9, 0.9, 0.9, 0.9]  # Accuracy
# Set the width of the bar chart.
bar_width = 0.02  # Reduce the width to fit side-by-side display and the range of Strength.

# Create graphics and dual Y-axes.
fig, ax1 = plt.subplots(figsize=(10, 5))

# Draw a side-by-side bar chart (Left Y-axis: Reasoning Token Numbers and Answer Token Numbers).
# Reasoning Token Numbers shift to the left, Answer Token Numbers shift to the right.
ax1.bar([s - bar_width/2 for s in sorted_strength_list], sorted_avg_tokens_reasoning_list, bar_width, color='skyblue', alpha=0.6, label='Reasoning Token Numbers')
ax1.bar([s + bar_width/2 for s in sorted_strength_list], sorted_avg_tokens_answer_list, bar_width, color='lightgreen', alpha=0.6, label='Answer Token Numbers')
ax1.set_xlabel('Strength')
ax1.set_ylabel('Token Numbers', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(0, max(max(sorted_avg_tokens_reasoning_list), max(sorted_avg_tokens_answer_list)) + 250)  # Dynamically set the Y-axis range.

# Create a right Y-axis and plot a line chart (Right Y-axis: Accuracy).
ax2 = ax1.twinx()
ax2.plot(sorted_strength_list, sorted_overall_accuracy_list, color='salmon', marker='o', linestyle='--', label='Accuracy')
ax2.set_ylabel('Accuracy', color='salmon')
ax2.tick_params(axis='y', labelcolor='salmon')
ax2.set_ylim(0.0, 1.0)  # Set the range of the right Y-axis.

# Add a legend.
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

# Set the title and layout.
plt.title('Reasoning and Answer Token Numbers vs Strength with Accuracy')
fig.tight_layout()

# Show the chart.
plt.show()


# %%
