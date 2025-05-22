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
model_name_list = ['DeepSeek-R1-Distill-Qwen-1.5B', 
                   'DeepSeek-R1-Distill-Qwen-7B', 
                   'DeepSeek-R1-Distill-Qwen-14B', 
                   'DeepSeek-R1-Distill-Qwen-32B', 
                   'QwQ-32B',]
dataset_list = ['MATH500', 'AIME2024', 'Minerva', 'AMC', 'OlympiadBench', 'GPQA_diamond']

base_eval_path = 'Data/Eval'


#%%
def plot_performance_vs_strength(overall_trend_results, title=None, save_path=None):
    """
    Draw the performance vs strength plot
    
    Args:
    overall_trend_results: a list of tuples, each containing (strength, metrics_dict)
    title: the title of the plot, if None, the default title will be used
    save_path: the path to save the plot, if None, the plot will be shown directly
    """
    # Extract data from overall_trend_results
    strength_list = [item[0] for item in overall_trend_results]
    accuracy_list = [item[1]['total_accuracy'] for item in overall_trend_results]
    reasoning_tokens_list = [item[1]['average_reasoning_token_num'] for item in overall_trend_results]
    answer_tokens_list = [item[1]['average_answer_token_num'] for item in overall_trend_results]

    # Create the plot with two Y-axes
    fig, ax1 = plt.subplots(figsize=(10, 5))

    bar_width = 0.02

    # Draw the bar plot (left Y-axis)
    ax1.bar([s - bar_width/2 for s in strength_list], reasoning_tokens_list, 
            bar_width, color='skyblue', alpha=0.6, label='Reasoning Token Number')
    ax1.bar([s + bar_width/2 for s in strength_list], answer_tokens_list, 
            bar_width, color='lightgreen', alpha=0.6, label='Answer Token Number')
    ax1.set_xlabel('Strength')
    ax1.set_ylabel('Token Number', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(0, max(max(reasoning_tokens_list), max(answer_tokens_list)) + 250)

    # Create the right Y-axis and draw the accuracy line plot
    ax2 = ax1.twinx()
    ax2.plot(strength_list, accuracy_list, color='salmon', 
             marker='o', linestyle='--', label='Accuracy')
    ax2.set_ylabel('Accuracy', color='salmon')
    ax2.tick_params(axis='y', labelcolor='salmon')
    ax2.set_ylim(0.0, 1.0)

    # Add the legend
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    # Set the title and layout
    if title is None:
        title = 'Reasoning and Answer Token Number vs Strength (with Accuracy)'
    plt.title(title)
    fig.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f'Save the plot to {save_path}')
        plt.show()
        plt.close()
    else:
        plt.show()

# Example usage:
# plot_performance_vs_strength(overall_trend_results)
# or
# plot_performance_vs_strength(overall_trend_results, 
#                            title='Custom Title',
#                            save_path='output.png')

#%%
from matplotlib import rcParams, font_manager

# Font settings
font_path = 'Assets/Times New Roman Bold.ttf'
font_prop = font_manager.FontProperties(fname=font_path)


def plot_performance_vs_strength(overall_trend_results, font_prop=None, figsize=(16, 8), 
                               title=None, save_path=None):
    """
    Plot the performance vs strength chart with two axes.
    
    Parameters:
        overall_trend_results: A list of tuples, each containing (strength, metrics_dict)
        font_prop: The font property, default is None
        figsize: The size of the chart, default is (12, 6)
        title: The title of the chart, default is None
        save_path: The path to save the chart, default is None (not saved)
    """
    try:
        # Extract data.
        strength_list = [item[0] for item in overall_trend_results]
        accuracy_list = [item[1]['total_accuracy'] for item in overall_trend_results]
        reasoning_tokens_list = [item[1]['average_reasoning_token_num'] for item in overall_trend_results]
        answer_tokens_list = [item[1]['average_answer_token_num'] for item in overall_trend_results]
        
        # Create a dual-axis chart.
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Set the width and position of the bar chart.
        bar_width = 0.02
        
        # Draw a bar chart for the left y-axis (Token Number).
        ax1.bar([s - bar_width/2 for s in strength_list], reasoning_tokens_list, 
                bar_width, linewidth=2, edgecolor='#515b83', color='#c4d7ef', 
                alpha=0.8, label='Reasoning Token Number')
        ax1.bar([s + bar_width/2 for s in strength_list], answer_tokens_list, 
                bar_width, linewidth=2, edgecolor='#6a7d9a', color='#a7d2cb', 
                alpha=0.8, label='Answer Token Number')
        
        # Set the left y-axis label and ticks.
        ax1.set_xlabel('Strength', fontproperties=font_prop, fontsize=32)
        ax1.set_ylabel('Token Number', fontproperties=font_prop, fontsize=32)
        
        # Increase the font size of the scale labels.
        ax1.tick_params(axis='x', labelsize=28)
        ax1.tick_params(axis='y', labelsize=28)
        
        # Set the range of the left y-axis.
        y_max = max(max(reasoning_tokens_list), max(answer_tokens_list)) * 1.15
        ax1.set_ylim(0, y_max)
        
        # Custom scale label font.
        for label in ax1.get_xticklabels():
            label.set_fontproperties(font_prop)
            label.set_fontsize(28)  # Directly set the font size of the x-axis tick labels.
        for label in ax1.get_yticklabels():
            label.set_fontproperties(font_prop)
            label.set_fontsize(28)  # Directly set the font size of the y-axis tick labels.
        
        # Create a right y-axis and plot the accuracy curve.
        ax2 = ax1.twinx()
        # ax2.plot(strength_list, accuracy_list, color='#ff8248', 
        #         marker='o', markersize=10, linestyle='--', linewidth=3.5,
        #         label='Accuracy')
        ax2.plot(strength_list, accuracy_list, color='#ff8248', 
                marker='o', markersize=10, linestyle='--', linewidth=3.5)
        
        # Set the right y-axis label and scale.
        ax2.set_ylabel('Accuracy', fontproperties=font_prop, fontsize=32)
        ax2.tick_params(axis='y', labelsize=28)  # Increase the font size of the right y-axis tick labels.
        ax2.set_ylim(0.0, 1.0)
        
        # Customize the font of the right y-axis tick labels.
        for label in ax2.get_yticklabels():
            label.set_fontproperties(font_prop)
            label.set_fontsize(28)  # Directly set the font size of the right y-axis tick labels.
        
        # Configure the axis borders and bold all borders.
        ax1.spines['left'].set_linewidth(3.5)
        ax1.spines['bottom'].set_linewidth(3.5)
        ax1.spines['right'].set_linewidth(3.5)
        ax1.spines['top'].set_linewidth(3.5)
        
        # Add grid lines.
        ax1.grid(True, linestyle='--', alpha=0.4, linewidth=2)
        
        # Create a legend.
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        
        font_prop_large = font_manager.FontProperties(fname=font_path, size=32)
        fig.legend(handles1 + handles2, labels1 + labels2, 
                   loc='upper center', bbox_to_anchor=(0.5, -0.0), 
                   ncol=3, prop=font_prop_large, fontsize=32)  # Increase the font size of the legend.
        
        # Set the title.
        # if title:
        #     plt.title(title, fontproperties=font_prop, fontsize=36, pad=20)
        
        # Adjust the layout.
        fig.tight_layout()
        
        # Save or display the chart.
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'The chart has been saved to {save_path}')
        
        plt.show()
        
    except Exception as e:
        plt.close()  # Close the chart when an error occurs.
        raise Exception(f"An error occurred during the plotting process: {e}")
#%%
for dataset in dataset_list:
    for model_name in model_name_list:
        eval_path = f'{base_eval_path}/{dataset}/{model_name}'
        if not os.path.exists(eval_path):
            continue
        print(f'{dataset} {model_name}')
        overall_trend_path = f'{eval_path}/overall_trend_results_correct_vote_num8.json'
        if not os.path.exists(overall_trend_path):
            overall_trend_path = f'{eval_path}/overall_trend_results.json'
            if not os.path.exists(overall_trend_path):
                continue
        print(overall_trend_path)
        with open(overall_trend_path, 'r') as f:
            overall_trend_results = json.load(f)
        print(overall_trend_results)
        overall_trend_results = {item['strength']: item for item in overall_trend_results}
        print(overall_trend_results)
        # Sort by key.
        overall_trend_results = sorted(overall_trend_results.items(), key=lambda x: x[0])
        # overall_trend_results = [item for item in overall_trend_results if item[0] <=0]
        print(overall_trend_results)
        plot_performance_vs_strength(overall_trend_results, 
                                    title=f'{dataset} {model_name}',
                                    save_path=f'{eval_path}/performance_vs_strength.png',
                                    font_prop=font_prop)
        # break
    # break

#%%

def plot_performance_vs_strength_only_reasoning_token_num(
    overall_trend_results, font_prop=None, figsize=(9.5, 8), title=None, save_path=None):
    """
    Plot the performance vs strength chart with two axes.
    
    Parameters:
        overall_trend_results: A list of tuples, each containing (strength, metrics_dict)
        font_prop: The font property, default is None
        figsize: The size of the chart, default is (12, 6)
        title: The title of the chart, default is None
        save_path: The path to save the chart, default is None (not saved)
    """
    try:
        # Extract data.
        strength_list = [item[0] for item in overall_trend_results]
        accuracy_list = [item[1]['total_accuracy'] for item in overall_trend_results]
        reasoning_tokens_list = [item[1]['average_reasoning_token_num'] for item in overall_trend_results]
        # answer_tokens_list = [item[1]['average_answer_token_num'] for item in overall_trend_results]
        
        # Create a dual-axis chart.
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Set the width and position of the bar chart.
        bar_width = 0.02
        
        # Draw a bar chart for the left y-axis (Token Number).
        ax1.bar(strength_list, reasoning_tokens_list, 
                bar_width, linewidth=2, edgecolor='#515b83', color='#c4d7ef', 
                alpha=0.8, label='Reasoning Token Number')
        
        # Set the left y-axis label and ticks.
        ax1.set_xlabel('Steering Strength λ', fontproperties=font_prop, fontsize=32)
        ax1.set_ylabel('Reasoning Token Number', fontproperties=font_prop, fontsize=32)
        
        # Increase the font size of the scale labels.
        ax1.tick_params(axis='x', labelsize=28)
        ax1.tick_params(axis='y', labelsize=28)
        
        # Set the range of the left y-axis.
        # y_max = max(max(reasoning_tokens_list), max(answer_tokens_list)) * 1.15
        y_max = max(reasoning_tokens_list) * 1.3
        ax1.set_ylim(0, y_max)
        
        # Customizing the font of scale labels.
        for label in ax1.get_xticklabels():
            label.set_fontproperties(font_prop)
            label.set_fontsize(28)  # Directly set the font size of the x-axis tick labels.
        for label in ax1.get_yticklabels():
            label.set_fontproperties(font_prop)
            label.set_fontsize(28)  # Directly set the font size of the y-axis tick labels.
        
        # Create a right y-axis and plot the accuracy curve.
        ax2 = ax1.twinx()
        # ax2.plot(strength_list, accuracy_list, color='#ff8248', 
        #         marker='o', markersize=10, linestyle='--', linewidth=3.5,
        #         label='Accuracy')
        ax2.plot(strength_list, accuracy_list, color='#ff8248', 
                marker='o', markersize=10, linestyle='--', linewidth=3.5,
                label='Accuracy')
        
        # Set the right y-axis label and scale.
        ax2.set_ylabel('Accuracy', fontproperties=font_prop, fontsize=32)
        ax2.tick_params(axis='y', labelsize=28)  # Increase the font size of the right y-axis tick labels.
        ax2.set_ylim(0.0, 1.0)
        
        # Customize the font of the right y-axis tick labels.
        for label in ax2.get_yticklabels():
            label.set_fontproperties(font_prop)
            label.set_fontsize(28)  # Directly set the font size of the right y-axis tick labels.
        
        # Configure the axis borders and bold all borders.
        ax1.spines['left'].set_linewidth(3.5)
        ax1.spines['bottom'].set_linewidth(3.5)
        ax1.spines['right'].set_linewidth(3.5)
        ax1.spines['top'].set_linewidth(3.5)
        
        # Add grid lines.
        ax1.grid(True, linestyle='--', alpha=0.4, linewidth=2)
        
        # Get legend handles and labels.
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        
        # Add a legend to the main image.
        font_prop_large = font_manager.FontProperties(fname=font_path, size=28)
        ax1.legend(handles1 + handles2, labels1 + labels2, 
                  loc='upper left', prop=font_prop_large, fontsize=28)
        
        # Set the title.
        # if title:
        #     plt.title(title, fontproperties=font_prop, fontsize=36, pad=20)
        
        # Adjust the layout.
        fig.tight_layout()
        
        # Save or display the chart.
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'The chart has been saved to {save_path}')
        
        plt.show()
        
    except Exception as e:
        plt.close()  # Close the chart when an error occurs.
        raise Exception(f"An error occurred during the plotting process: {e}")



for dataset in dataset_list:
    for model_name in model_name_list:
        eval_path = f'{base_eval_path}/{dataset}/{model_name}'
        if not os.path.exists(eval_path):
            continue
        print(f'{dataset} {model_name}')
        overall_trend_path = f'{eval_path}/overall_trend_results_correct_vote_num8.json'
        if not os.path.exists(overall_trend_path):
            overall_trend_path = f'{eval_path}/overall_trend_results.json'
            if not os.path.exists(overall_trend_path):
                continue
        print(overall_trend_path)
        with open(overall_trend_path, 'r') as f:
            overall_trend_results = json.load(f)
        print(overall_trend_results)
        overall_trend_results = {item['strength']: item for item in overall_trend_results}
        print(overall_trend_results)
        # Sort by key.
        overall_trend_results = sorted(overall_trend_results.items(), key=lambda x: x[0])
        overall_trend_results = [item for item in overall_trend_results if item[0] <=0]
        print(overall_trend_results)
        plot_performance_vs_strength_only_reasoning_token_num(overall_trend_results, 
                                    title=f'{dataset} {model_name}',
                                    save_path=f'{eval_path}/performance_vs_strength_only_reasoning_token_num.png',
                                    font_prop=font_prop)
        break
    break
        
        
# %%

for model_name in model_name_list:
    eval_path = f'{base_eval_path}/MATH500/{model_name}'
    results_by_strength = {}
    for strength in [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]:
        generation_path = f'{eval_path}/MATH500-{model_name}_{strength}_correct_eval_vote_num8.json'
        if not os.path.exists(generation_path):
            print(f'{generation_path} does not exist')
            continue
        print(f'{generation_path}')
        with open(generation_path, 'r') as f:
            generation_results = json.load(f)
        # print(generation_results)
        results_by_level = {1: {'average_reasoning_token_num':[], 'average_answer_token_num':[], 'total_accuracy':[]}, 
                    2: {'average_reasoning_token_num':[], 'average_answer_token_num':[], 'total_accuracy':[]}, 
                    3: {'average_reasoning_token_num':[], 'average_answer_token_num':[], 'total_accuracy':[]}, 
                    4: {'average_reasoning_token_num':[], 'average_answer_token_num':[], 'total_accuracy':[]}, 
                    5: {'average_reasoning_token_num':[], 'average_answer_token_num':[], 'total_accuracy':[]}}
        
        for result in generation_results:
            level = result['level']
            results_by_level[level]['average_reasoning_token_num'].append(result['avg_llm_reasoning_token_num'])
            results_by_level[level]['average_answer_token_num'].append(result['avg_llm_answer_token_num'])
            results_by_level[level]['total_accuracy'].append(result['accuracy'])
        
        for level in results_by_level:
            results_by_level[level]['average_reasoning_token_num'] = np.mean(results_by_level[level]['average_reasoning_token_num'])
            results_by_level[level]['average_answer_token_num'] = np.mean(results_by_level[level]['average_answer_token_num'])
            results_by_level[level]['total_accuracy'] = np.mean(results_by_level[level]['total_accuracy'])
        results_by_strength[strength] = results_by_level
        # break
    results_for_display_by_level = {}
    for level in [1, 2, 3, 4, 5]:
        results_for_display_by_level[level] = {}
        for strength in results_by_strength:
            results_for_display_by_level[level][strength] = results_by_strength[strength][level]
        results_for_display_by_level[level] = sorted(results_for_display_by_level[level].items(), key=lambda x: x[0])
    # Draw a diagram for each level using the function above.
    for level in results_for_display_by_level:
        plot_performance_vs_strength(results_for_display_by_level[level], 
                                    title=f'{model_name} Level {level}',
                                    save_path=f'{eval_path}/level_{level}_performance_vs_strength.png')
    # break


results_by_strength

#%%
original_results_list = []
for model_name in model_name_list:
    eval_path = f'{base_eval_path}/MATH500/{model_name}'
    results_by_strength = {}
    for strength in [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]:
        generation_path = f'{eval_path}/MATH500-{model_name}_{strength}_correct_eval_vote_num8.json'
        if not os.path.exists(generation_path):
            print(f'{generation_path} does not exist')
            continue
        print(f'{generation_path}')
        with open(generation_path, 'r') as f:
            generation_results = json.load(f)
        # print(generation_results)
        results_by_level = {1: {'average_reasoning_token_num':[], 'average_answer_token_num':[], 'total_accuracy':[]}, 
                    2: {'average_reasoning_token_num':[], 'average_answer_token_num':[], 'total_accuracy':[]}, 
                    3: {'average_reasoning_token_num':[], 'average_answer_token_num':[], 'total_accuracy':[]}, 
                    4: {'average_reasoning_token_num':[], 'average_answer_token_num':[], 'total_accuracy':[]}, 
                    5: {'average_reasoning_token_num':[], 'average_answer_token_num':[], 'total_accuracy':[]}}
        
        for result in generation_results:
            level = result['level']
            results_by_level[level]['average_reasoning_token_num'].append(result['avg_llm_reasoning_token_num'])
            results_by_level[level]['average_answer_token_num'].append(result['avg_llm_answer_token_num'])
            results_by_level[level]['total_accuracy'].append(result['accuracy'])
        
        for level in results_by_level:
            results_by_level[level]['average_reasoning_token_num'] = np.mean(results_by_level[level]['average_reasoning_token_num'])
            results_by_level[level]['average_answer_token_num'] = np.mean(results_by_level[level]['average_answer_token_num'])
            results_by_level[level]['total_accuracy'] = np.mean(results_by_level[level]['total_accuracy'])
        results_by_strength[strength] = results_by_level
        # break
    results_for_display_by_level = {}
    for level in [1, 2, 3, 4, 5]:
        results_for_display_by_level[level] = {}
        for strength in results_by_strength:
            results_for_display_by_level[level][strength] = results_by_strength[strength][level]
        results_for_display_by_level[level] = sorted(results_for_display_by_level[level].items(), key=lambda x: x[0])
    print("Original results:")
    print(results_for_display_by_level[1][4])
    original_results_list.append(results_for_display_by_level[1][4])
    print(model_name)
    print(results_for_display_by_level)
    # Draw a diagram for each level using the function above.
    # break

# results_by_strength

# %%
# print(len(results_by_level[1]['average_reasoning_token_num']))
# print(len(results_by_level[2]['average_reasoning_token_num']))
# print(len(results_by_level[3]['average_reasoning_token_num']))
# print(len(results_by_level[4]['average_reasoning_token_num']))
# print(len(results_by_level[5]['average_reasoning_token_num']))
# %%
original_results_list

# %%
