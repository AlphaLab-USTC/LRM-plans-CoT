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
import random


#%%
from matplotlib import rcParams, font_manager

# font_path = 'Assets/Times New Roman.ttf'

font_path = 'Assets/Times New Roman Bold.ttf'
font_prop = font_manager.FontProperties(fname=font_path)
# rcParams['font.family'] = font_prop.get_name()

# %load_ext autoreload
# %autoreload 2
#%%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DeepSeek-R1-Distill-Qwen-14B',
                        help='model name.')
    parser.add_argument('--dataset', type=str, default='MATH',
                        help='dataset name.')
    args, _ = parser.parse_known_args()
    return args, _

args, _ = parse_args()
model_name = args.model_name
dataset = args.dataset

# model_name = "Qwen2.5-3B-instruct"
# model_name = 'DeepSeek-R1-Distill-Qwen-1.5B'
# model_name = 'DeepSeek-R1-Distill-Qwen-7B'
# model_name = 'DeepSeek-R1-Distill-Qwen-14B'
model_name = 'DeepSeek-R1-Distill-Qwen-32B'
# model_name = 'QwQ-32B'
dataset = 'MATH'
# dataset = 'MATH500'
# dataset = 'AIME2024'

# model_name_split = model_name.split('-')

# tmp_model_name = model_name_split[0] + '-' + model_name_split[-1]

# if model_name != 'QwQ-32B':
#     generation_path = f"Data/Response/{dataset}/{dataset}-DeepSeek-R1-Distill-{tmp_model_name}_vote_num8.json"
# else:
#     generation_path = f"Data/Response/{dataset}/{dataset}-QwQ-32B_vote_num8.json"

generation_path = f"Data/Response/{dataset}/{dataset}-{model_name}_vote_num8.json"

representation_path = f"Data/Representation/{dataset}/{model_name}"
if not os.path.exists(representation_path):
    # create the directory
    os.makedirs(representation_path)

model_path = "model/Qwen/QwQ-32B"

# save_path = f'Data_gen/Info/{model_name}_data_information_split.json'
base_asset_path = f"Assets/{dataset}/{model_name}"

representation_files = os.listdir(representation_path)
print("Total representation files: ", len(representation_files))

base_asset_path, grouped_token_number_path, layer_wise_regression_path, layer_wise_cosine_similarity_path, layer_trend_path, tsne_path, layer_wise_coefficients_path = prepare_save_path(base_asset_path)

#%%
#@ Preperation
# data = preprocess_data("Data/Response/MATH/MATH-DeepSeek-R1-Distill-Qwen-7B_vote_num8.json")
# data = preprocess_data(f"Data/Response/{dataset}/{dataset}-DeepSeek-R1-Distill-{model_name}_vote_num8.json")
data = preprocess_data(generation_path)
#%%
overall_accuracy, level_accuracy, df_wrong, df_accuracy = analyze_model_accuracy(
    data=data,
    model_name=model_name,
    accuracy_threshold=0.2,
    samples_per_level=20
)
overall_accuracy
# #%%
# #@ Optional: Select problems with level 1
# selected_problem_list = [i for i in range(len(data)) if data[i]['level'] == 3]
# Randomly select 50 samples.
# selected_problem_list = random.sample(selected_problem_list, 50)
# #%%
# selected_problem_file_name = f"{dataset}_Level3.json"

#%%
# Convert data_math to the format of data.


# %%
#@ Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_path)
data_information = analyze_token_numbers(
    data=data,
    tokenizer=tokenizer,
    model_name=model_name,
    save_path=base_asset_path + f'/{model_name}_data_information_split.json'
)
#%%
#@ Analyze Token Numbers
data_information
data_information_df = process_data_information_frame(data_information)
#%%
data_information_df = data_information_df.merge(df_accuracy[['problem', 'accuracy']], on='problem', how='left')
data_information_df
#%%
# Filter out samples with an accuracy of less than 0.2 and add a column named is_correct.
df_accuracy['is_correct'] = df_accuracy['accuracy'] >= 0.2
incorrect_samples = df_accuracy[df_accuracy['accuracy'] < 0.2]
print("Incorrect samples number:", len(incorrect_samples))
# Output detailed information of incorrect samples.
print(incorrect_samples)

#%%
def visualize_level_statistics(df, model_name, group_col='level', value_col='avg_tokens_reasoning', 
                               figsize=(10, 6), save_path=None):
    """
    Visualize the average token number of different difficulty levels.

    Parameters:
        df: The DataFrame containing the analysis data.
        model_name: The name of the model, used for the chart title.
        group_col: The column name used for grouping, default is 'level'.
        value_col: The column name used for calculating the average value, default is 'avg_tokens_reasoning'.
        figsize: The size of the chart, default is (10, 6).
        save_path: The path to save the chart, default is None (not saved).

    Returns:
        level_avg: The DataFrame containing the average token number of each difficulty level.
    """
    # Calculate the average number of tokens for each difficulty level.
    level_avg = df.groupby(group_col)[value_col].mean().reset_index()
    level_avg.columns = [group_col, 'mean_tokens']

    # Print statistical results.
    print(f"Each {group_col} average token number:")
    print(level_avg)

    # Visualization
    plt.figure(figsize=figsize)
    # Thicken the edge lines of the bar chart.
    plt.bar(level_avg[group_col], level_avg['mean_tokens'],
            linewidth=2, edgecolor='#515b83', color='#c4d7ef')

    # Configure the axis borders, bold the left and bottom borders, and hide the top and right borders.
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1)    # Bold the left border.
    ax.spines['bottom'].set_linewidth(1)  # Bold the bottom border.

    # Set labels and scales.
    plt.ylabel('Average Token Number', fontproperties=font_prop, fontsize=24)
    plt.xticks(fontproperties=font_prop, fontsize=20)
    plt.yticks(fontproperties=font_prop, fontsize=20)
    plt.tight_layout()

    # If a save path is specified, save the chart.
    if save_path:
        plt.savefig(save_path)

    plt.show()

    return level_avg

#%%
#@ Visualize Token Numbers
reasoning_level_avg = visualize_level_statistics(data_information_df, model_name, value_col='avg_tokens_reasoning', save_path=grouped_token_number_path + f'/average_token_reasoning_level_statistics.png')
answer_level_avg = visualize_level_statistics(data_information_df, model_name, value_col='avg_tokens_answer', save_path=grouped_token_number_path + f'/average_token_answer_level_statistics.png')
#%%
print(reasoning_level_avg.mean_tokens.mean())
print(answer_level_avg.mean_tokens.mean())

#%%
#@ Visualize Token Numbers Difference
diff_reasoning_level_avg = visualize_diff_level_statistics(data_information_df, model_name, value_col='avg_tokens_reasoning', save_path=grouped_token_number_path + f'/diff_average_token_reasoning_level_statistics.png')
diff_answer_level_avg = visualize_diff_level_statistics(data_information_df, model_name, value_col='avg_tokens_answer', save_path=grouped_token_number_path + f'/diff_average_token_answer_level_statistics.png')


# %%
#@ One single layer
# layer_number = int(0.8 * len(representation_files))
layer_number = 53
file = representation_files[layer_number]
data_path = representation_path + '/' + file
X = np.load(data_path)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# X_scaled = X
indices = np.arange(len(X_scaled)) 
y = np.array([item['avg_tokens_reasoning'] for item in data_information])
# Normalize y.
# y = (y - y.min()) / (y.max() - y.min())
# y = y / 100
# y = np.log(y)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_scaled, y, indices, test_size=0.1, random_state=42)

    
# Initialize Lasso model with regularization parameter alpha
alpha = 10  # Adjust alpha as needed
# model = Ridge(alpha=alpha)
model = Lasso(alpha=alpha)
# model = LinearRegression()
# model = MLPRegressor(hidden_layer_sizes=(128, 64), 
# activation='relu',        # The activation function can be 'relu', 'tanh', 'logistic'
# solver='adam',            # optimizer
# max_iter=500, # maximum number of iterations
#                     early_stopping=True,
#                     validation_fraction=0.1,
#                     learning_rate_init=5e-4,
#                     verbose=True,
#                     random_state=42)


y_train_pred, y_test_pred, mse_train, r2_train, spearman_train, mse_test, r2_test, spearman_test = regression_at_each_layer(X_train, X_test, y_train, y_test, model)

#%%
# plot_regression_results(y_train, y_test, y_train_pred, y_test_pred, mse_train, r2_train, spearman_train, mse_test, r2_test, spearman_test, model_name, layer_number=layer_number, save_path=layer_wise_regression_path + f'/layer_{layer_number}.png', correction_df=df_accuracy, indices_train=indices_train, indices_test=indices_test)
plot_regression_results(y_train, y_test, y_train_pred, y_test_pred, mse_train, r2_train, spearman_train, mse_test, r2_test, spearman_test, model_name, layer_number=layer_number, save_path=layer_wise_regression_path + f'/layer_{layer_number}.png')

#%%
def plot_regression_results_single(y_test, y_test_pred, 
                            mse_test, r2_test, spearman_test,
                            model_name,
                            figsize=(8, 8), point_size=14,
                            layer_number=None,
                            save_path=None):
    """
    Visualize the prediction results of the regression model on the test set, only show the test set.
    
    Parameters:
        y_test: The actual value of the test set
        y_test_pred: The predicted value of the test set
        mse_test: The MSE of the test set
        r2_test: The R² of the test set
        spearman_test: The Spearman correlation coefficient of the test set
        model_name: The name of the model
        figsize: The size of the chart, default is (8, 8)
        alpha_value: The transparency of the scatter plot, default is 1
        point_size: The size of the scatter plot, default is 14
        layer_number: The current layer number, used for the title display, default is None
        save_path: The path to save the chart, default is None (not saved)
    """
    plt.figure(figsize=figsize)
    
    # Draw a scatter plot of the test set.
    plt.scatter(y_test, y_test_pred, alpha=0.4, s=point_size, 
                edgecolor='#515b83', color='#3a76a3', linewidth=0.5)
    
    # Draw reference diagonal.
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=1, color='#eab299', linewidth=3.5)
    
    # Configure the axis borders, thicken the left and bottom borders, and hide the top and right borders.
    ax = plt.gca()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(3.5)    # Bold the left border.
    ax.spines['bottom'].set_linewidth(3.5)  # Bold the bottom border.
    ax.spines['right'].set_linewidth(3.5)    # Bold the right border.
    ax.spines['top'].set_linewidth(3.5)  # Bold the top border.
    
    # Set the display range of the x-axis and y-axis to show only the part greater than 0.
    plt.xlim(0, 16384)
    plt.ylim(0, 16384)
    
    # Set tags and titles.
    plt.xlabel('Actual Token Number', fontproperties=font_prop, fontsize=32)
    plt.ylabel('Predicted Token Number', fontproperties=font_prop, fontsize=32)
    # plt.title(f'Test Set: Actual vs Predicted\n{model_name} (Layer {layer_number})')
    # plt.xticks(fontproperties=font_prop, fontsize=24)
    # plt.yticks(fontproperties=font_prop, fontsize=24)

    fixed_ticks = [0, 2500, 5000, 7500, 10000, 12500, 15000]
    plt.xticks(fixed_ticks, fontproperties=font_prop, fontsize=24)
    plt.yticks(fixed_ticks, fontproperties=font_prop, fontsize=24)

    # Add evaluation metric text box.
    text_x = min_val + 0.12 * (max_val - min_val)
    text_y = max_val - 0.1 * (max_val - min_val)
    metrics_text = (
        f'Spearman R: {spearman_test:.4f}'
    )
    plt.text(text_x, text_y, metrics_text, bbox=dict(facecolor='white', alpha=0.7),
             fontproperties=font_prop, fontsize=24)
    
    plt.grid(True, linestyle='--', alpha=0.8, linewidth=2.5)
    plt.tight_layout()
    
    # If a save path is specified, save the chart.
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
plot_regression_results_single(y_test, y_test_pred, mse_test, r2_test, spearman_test, model_name, save_path=layer_wise_regression_path + f'/layer_{layer_number}.png')

#%%
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import ticker

# def plot_regression_results_single_hexbin(
#     y_test, y_test_pred, 
#     mse_test, r2_test, spearman_test,
#     model_name,
#     figsize=(6, 6),
#     gridsize=40,
#     layer_number=None,
#     save_path=None
# ):
#     """
# The prediction results of the visualization regression model on the test set, 2D frequency map (hexbin), without fitting line.
#     """
#     plt.figure(figsize=figsize)

# 2D Frequency Map (hexbin)
#     hb = plt.hexbin(
#         y_test, y_test_pred, 
#         gridsize=gridsize, 
#         cmap='Blues', 
#         mincnt=1, 
#         # linewidths=0.5,
#         # edgecolors='gray',
# bins='log',  # Color depth is based on a logarithmic distribution.
# alpha=0.7    # Increase the alpha value to make low probability areas more visible.
#     )
#     cb = plt.colorbar(hb, format=ticker.PercentFormatter(xmax=hb.get_array().max(), decimals=1))
#     cb.set_label('Frequency', fontsize=14)

# Draw the diagonal of perfect prediction.
#     min_val = min(y_test.min(), y_test_pred.min())
#     max_val = max(y_test.max(), y_test_pred.max())
#     plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

# Axis Settings
#     plt.xlim(min_val, max_val)
#     plt.ylim(min_val, max_val)
#     plt.xlabel('Predicted Labels', fontsize=16)
#     plt.ylabel('Real Labels', fontsize=16)

# Title and Legend
#     plt.title(model_name, fontsize=18)
#     plt.legend(loc='upper left', fontsize=12)

# Indicator Text
#     metrics_text = (
#         f'S: {spearman_test:.2f}±0.00, P: {r2_test:.2f}±0.00'
#     )
#     plt.text(
#         0.05 * max_val, 0.95 * max_val, metrics_text,
#         fontsize=12, bbox=dict(facecolor='white', alpha=0.7)
#     )

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
# Usage example
# plot_regression_results_single_hexbin(y_test, y_test_pred, mse_test, r2_test, spearman_test, model_name, save_path=layer_wise_regression_path + f'/layer_{layer_number}.png')


# y_train_pred, y_test_pred, mse_train, r2_train, spearman_train, mse_test, r2_test, spearman_test = regression_at_each_layer(X_scaled, X_test, y, y_test, model)
# plot_regression_results(y, y_test, y_train_pred, y_test_pred, mse_train, r2_train, spearman_train, mse_test, r2_test, spearman_test, model_name, layer_number=layer_number, save_path=layer_wise_regression_path + f'/layer_{layer_number}.png')

# %%
#@ All layers
param_list = {'coef': [], 'intercept': []}
train_MSEs, test_MSEs, train_R2s, test_R2s, train_spearman, test_spearman = [], [], [], [], [], []

for idx, file in enumerate(representation_files):
    print("Is processing layer: ", idx)
    print("Processing file: ", file)
    data_path = representation_path + '/' + file
    
    X = np.load(data_path)
    y = np.array([item['avg_tokens_reasoning'] for item in data_information])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    alpha = 10  # Adjust alpha as needed
    model = Lasso(alpha=alpha)

    y_train_pred, y_test_pred, mse_train, r2_train, spearman_train, mse_test, r2_test, spearman_test = regression_at_each_layer(X_train, X_test, y_train, y_test, model)
    # plot_regression_results(y_train, y_test, y_train_pred, y_test_pred, mse_train, r2_train, spearman_train, mse_test, r2_test, spearman_test, model_name, layer_number=idx, save_path=layer_wise_regression_path + f'/layer_{idx}.png')
    plot_regression_results_single(y_test, y_test_pred, mse_test, r2_test, spearman_test, model_name, layer_number=idx, save_path=layer_wise_regression_path + f'/layer_{idx}.png')
    param_list['coef'].append(model.coef_)
    param_list['intercept'].append(model.intercept_)
    train_MSEs.append(mse_train)
    test_MSEs.append(mse_test)
    train_R2s.append(r2_train)
    test_R2s.append(r2_test)
    train_spearman.append(spearman_train)
    test_spearman.append(spearman_test)
    # break

# Save the regression coefficients to a npy file.
np.save(os.path.join(base_asset_path, 'regression_coef.npy'), param_list)

# %%
def plot_layer_wise_metrics_single(test_metrics, metric_name, 
                          model_name=None, figsize=(8, 4), 
                          test_marker='s',
                          save_path=None):
    """
    Visualize the change trend of the evaluation index of the model on the test set at different layers.
    
    Parameters:
        test_metrics: The evaluation index list of each layer of the test set
        metric_name: The name of the evaluation index (e.g., 'MSE', 'R²', 'Spearman' etc.)
        model_name: The name of the model, used for the title, default is None
        figsize: The size of the chart, default is (10, 6)
        test_marker: The marker style of the test set data point, default is 's'
        save_path: The path to save the chart, default is None (not saved)

    返回:
        None
    """
    plt.figure(figsize=figsize)
    
    # Draw a trend line.
    x = range(len(test_metrics))
    plt.plot(x, test_metrics, marker=test_marker, linestyle='-', 
            markersize=7, label=f'Testing {metric_name}',
            color='#3a76a3', linewidth=3.5)
    
    # Set the x-axis scale to display one every 5 layers.
    all_layers = range(len(test_metrics))
    visible_layers = [i for i in all_layers if i % 5 == 0]
    plt.xticks(visible_layers, [f'{i}' for i in visible_layers], 
               fontproperties=font_prop, fontsize=24)
    plt.yticks(fontproperties=font_prop, fontsize=24)
    
    # Set labels and titles.
    plt.xlabel('Layer Index', fontproperties=font_prop, fontsize=32)
    plt.ylabel(metric_name, fontproperties=font_prop, fontsize=32)
    
    # Configure the axis borders and bold all borders.
    ax = plt.gca()
    ax.spines['left'].set_linewidth(3.5)
    ax.spines['bottom'].set_linewidth(3.5)
    ax.spines['right'].set_linewidth(3.5)
    ax.spines['top'].set_linewidth(3.5)
    
    # Add grid.
    plt.grid(True, linestyle='--', alpha=0.8, linewidth=2.5)
    
    # Add a legend.
    # plt.legend(prop=font_prop, fontsize=24)
    
    plt.tight_layout()
    
    # If a save path is specified, save the chart.
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    
plot_layer_wise_metrics_single(test_spearman, 'Spearman R', model_name, save_path=layer_trend_path + f'/layer_wise_spearman.png')

# %%
#@ Extract steering vectors
#@ Single layer
layer_number = int(0.8 * len(representation_files))
file = representation_files[layer_number]
data_path = representation_path + '/' + file
X = np.load(data_path)

df = pd.DataFrame(data_information)

# Only keep the necessary columns.
df = df[['problem', 'level', 'avg_tokens_reasoning']]
df.loc[df['level'] == 'Level ?', 'level'] = 'Level 3'

# Add embedding column to DataFrame.
df['embedding'] = list(X)

df = df.merge(df_accuracy[['problem', 'accuracy']], on='problem', how='left')
# df = df[df['accuracy'] >= 0.8]
# df = df[df['accuracy'] < 0.5]
diff_df, level_names = construct_diff_df(df, X)
diff_df

# %%
def plot_cosine_similarity_matrix(diff_embeddings, level_names, 
                                figsize=(10, 8), cmap=None,
                                sim_range=(-1, 1), text_threshold=0.5,
                                title=None, save_path=None):
    """
    Visualize the cosine similarity matrix between difficulty level difference vectors.

    Parameters:
        diff_embeddings: The matrix of difference vectors, shape is (n_levels, embedding_dim)
        level_names: The list of difficulty level names
        figsize: The size of the chart, default is (10, 8)
        cmap: The color mapping of the heatmap, default is None (use the custom color mapping)
        sim_range: The range of the similarity value, default is (-1, 1)
        text_threshold: The threshold for text color switching, default is 0.5
        title: The title of the chart, default is None
        save_path: The path to save the chart, default is None (not saved)

    Returns:
        cosine_sim_matrix: The cosine similarity matrix
        mean_similarity: The average similarity value
    """
    # Calculate the cosine similarity matrix.
    cosine_sim_matrix = cosine_similarity(diff_embeddings)
    
    # Create a chart.
    plt.figure(figsize=figsize)
    
    # Create a custom color mapping from #3a76a3 to #eab299.
    if cmap is None:
        colors = ['#fffffb', '#fcedd1'] # [#ef8a57  #f5c278]#3a76a3 #eab299
        from matplotlib.colors import LinearSegmentedColormap
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        cmap = custom_cmap
    
    # Draw a heatmap.
    im = plt.imshow(cosine_sim_matrix, cmap=cmap, 
                    vmin=sim_range[0], vmax=sim_range[1])
    cbar = plt.colorbar(im)
    # cbar.set_label('Cosine Similarity', fontproperties=font_prop, fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    
    # Set the title and tags.
    if title is None:
        title = 'Cosine Similarity Matrix of Difficulty Level Differences'
    # plt.title(title, fontproperties=font_prop, fontsize=32)
    
    # Set scale labels.
    plt.xticks(range(len(level_names)), level_names, rotation=45, 
               fontproperties=font_prop, fontsize=24)
    plt.yticks(range(len(level_names)), level_names, 
               fontproperties=font_prop, fontsize=24)
    
    # Add similarity values in each cell.
    for i in range(len(level_names)):
        for j in range(len(level_names)):
            sim_value = cosine_sim_matrix[i, j]
            # text_color = "white" if sim_value < text_threshold else "black"
            text_color = "black"
            plt.text(j, i, f'{sim_value:.2f}',
                    ha="center", va="center", 
                    color=text_color, fontproperties=font_prop, fontsize=20)
    
    # Configure the axis borders and bold all borders.
    ax = plt.gca()
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.tight_layout()
    
    # If a save path is specified, save the chart.
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Calculate and return the average similarity.
    mean_similarity = np.mean(cosine_sim_matrix)
    
    return cosine_sim_matrix, mean_similarity

level_names = [r'$\mathbf{r}_{\text{2} \leftarrow \text{1}}$', r'$\mathbf{r}_{\text{3} \leftarrow \text{1}}$', r'$\mathbf{r}_{\text{4} \leftarrow \text{1}}$', r'$\mathbf{r}_{\text{5} \leftarrow \text{1}}$']
plot_cosine_similarity_matrix(
    np.vstack(diff_df['diff_embedding'].values), 
    level_names, 
    save_path=layer_wise_cosine_similarity_path + f'/layer_{layer_number}.png')

#%%
from mpl_toolkits.mplot3d import Axes3D  # Note: This module needs to be imported.
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def plot_cosine_similarity_3d_bar(diff_embeddings, level_names, 
                                 figsize=(10, 10), sim_range=(-1, 1),
                                 cmap=None, title=None, save_path=None):
    cosine_sim_matrix = cosine_similarity(diff_embeddings)
    n = len(level_names)
    xpos, ypos = np.meshgrid(np.arange(n), np.arange(n))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dz = cosine_sim_matrix.flatten()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    dx = dy = 0.7 * np.ones_like(zpos)


    if cmap is None:
        colors = ['#eab299', '#ffe7d9']
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        cmap = custom_cmap

    norm = plt.Normalize(sim_range[0], sim_range[1])
    colors = cmap(norm(dz))

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)

    ax.set_xticks(np.arange(n) + 0.35)
    ax.set_yticks(np.arange(n) + 0.35)
    ax.set_xticklabels(level_names, fontproperties=font_prop, fontsize=24, rotation=45, ha='right')
    ax.set_yticklabels(level_names, fontproperties=font_prop, fontsize=24)
    # Increase the font size of the z-axis coordinates and set the font to font_prop.
    for tick in ax.get_zticklabels():
        tick.set_fontproperties(font_prop)
    ax.tick_params(axis='z', labelsize=20)
    ax.set_zlabel('Cosine Similarity', fontproperties=font_prop, fontsize=24)
    ax.zaxis.labelpad = 15

    ax.set_zlim(0, 1.02)

    z_ticks = ax.get_zticks()
    filtered_ticks = [t for t in z_ticks if t < 1.02]
    ax.set_zticks(filtered_ticks)

    ax.set_box_aspect(aspect=None, zoom=0.85)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
        # Calculate and return the average similarity.
    mean_similarity = np.mean(cosine_sim_matrix)
    
    
    return cosine_sim_matrix, mean_similarity

level_names = [r'$\mathbf{r}_{\text{2} \leftarrow \text{1}}$', r'$\mathbf{r}_{\text{3} \leftarrow \text{1}}$', r'$\mathbf{r}_{\text{4} \leftarrow \text{1}}$', r'$\mathbf{r}_{\text{5} \leftarrow \text{1}}$']
cosine_sim_matrix, mean_similarity = plot_cosine_similarity_3d_bar(np.vstack(diff_df['diff_embedding'].values), level_names, save_path=layer_wise_cosine_similarity_path + f'/layer_{layer_number}_3d_bar.png')
# %%
# for acc_type in ['all', 'correct', 'incorrect']:
# for acc_type in ['all']:
#@ All layers
diff_dfs = []
cosine_sims = []
avg_cosine_sims = []


files = os.listdir(representation_path)

for idx, file in enumerate(files):
    # if idx != 20:
        # continue
    print("Is processing layer: ", idx)
    print("Processing file: ", file)
    data_path = representation_path + '/' + file
    rep = np.load(data_path)
    print(rep.shape)
    
    X = rep
    
    df = pd.DataFrame(data_information)
    df = df[['problem', 'level', 'avg_tokens_reasoning']]
    df.loc[df['level'] == 'Level ?', 'level'] = 'Level 3'
    df['embedding'] = list(X)
    df = df.merge(df_accuracy[['problem', 'accuracy']], on='problem', how='left')

    diff_df, level_names = construct_diff_df(df, X)
    diff_dfs.append(diff_df)
    
    # cosine_sim_matrix, mean_similarity = plot_cosine_similarity_matrix(np.vstack(diff_df['diff_embedding'].values), level_names, save_path=layer_wise_cosine_similarity_path + f'/layer_{idx}_{acc_type}.png')
    level_names = [r'$\mathbf{r}_{\text{2} \leftarrow \text{1}}$', r'$\mathbf{r}_{\text{3} \leftarrow \text{1}}$', r'$\mathbf{r}_{\text{4} \leftarrow \text{1}}$', r'$\mathbf{r}_{\text{5} \leftarrow \text{1}}$']
    # cosine_sim_matrix, mean_similarity = plot_cosine_similarity_3d_bar(np.vstack(diff_df['diff_embedding'].values), level_names, save_path=layer_wise_cosine_similarity_path + f'/layer_{idx}.png')
    cosine_sim_matrix, mean_similarity = plot_cosine_similarity_matrix(
        np.vstack(diff_df['diff_embedding'].values), level_names, 
        save_path=layer_wise_cosine_similarity_path + f'/matrix_layer_{idx}.png')
    cosine_sims.append(cosine_sim_matrix)
    avg_cosine_sims.append(mean_similarity)
    # break

#%%
# break
########################################################################################

def plot_layer_wise_cosine_similarity(avg_cosine_sims, model_name=None, 
                                    figsize=(8, 4), marker='o',
                                    grid=True, save_path=None):
    """
    Visualize the change trend of the average cosine similarity between different layers.
    
    Parameters:
        avg_cosine_sims: The list or array of the average cosine similarity of each layer
        model_name: The name of the model, used for the title, default is None
        figsize: The size of the chart, default is (8, 4)
        marker: The marker style of the data point, default is 'o'
        grid: Whether to display the grid line, default is True
        save_path: The path to save the chart, default is None (not saved)
        
    Returns:
        max_sim_layer: The layer index of the maximum similarity
        max_sim_value: The maximum similarity value
    """
    try:
        # Ensure the input is a numpy array.
        avg_cosine_sims = np.array(avg_cosine_sims)
        
        # Create a chart.
        plt.figure(figsize=figsize)
        
        # Draw a trend line.
        x = range(len(avg_cosine_sims))
        plt.plot(x, avg_cosine_sims, marker=marker, linestyle='-', 
                markersize=7, label='Average Similarity',
                color='#ff8248', linewidth=3.5)
        
        # Set the x-axis scale to display one every 5 layers.
        all_layers = range(len(avg_cosine_sims))
        visible_layers = [i for i in all_layers if i % 5 == 0]
        plt.xticks(visible_layers, [f'{i}' for i in visible_layers], 
                  fontproperties=font_prop, fontsize=24)
        plt.yticks(fontproperties=font_prop, fontsize=24)
        
        # Set labels and titles.
        plt.xlabel('Layer Index', fontproperties=font_prop, fontsize=32)
        plt.ylabel('Mean Similarity', fontproperties=font_prop, fontsize=32)
        
        # Configure the axis borders and bold all borders.
        ax = plt.gca()
        ax.spines['left'].set_linewidth(3.5)
        ax.spines['bottom'].set_linewidth(3.5)
        ax.spines['right'].set_linewidth(3.5)
        ax.spines['top'].set_linewidth(3.5)
        
        # Add grid.
        if grid:
            plt.grid(True, linestyle='--', alpha=0.8, linewidth=2.5)
        
        # Label the maximum and minimum values.
        max_idx = np.argmax(avg_cosine_sims)
        min_idx = np.argmin(avg_cosine_sims)
        
        # plt.annotate(f'Max: {avg_cosine_sims[max_idx]:.4f}', 
        #             xy=(max_idx, avg_cosine_sims[max_idx]),
        #             xytext=(10, -15), textcoords='offset points',
        #             bbox=dict(facecolor='white', alpha=0.7),
        #             fontproperties=font_prop, fontsize=20)
        
        # plt.annotate(f'Min: {avg_cosine_sims[min_idx]:.4f}', 
        #             xy=(min_idx, avg_cosine_sims[min_idx]),
        #             xytext=(10, -15), textcoords='offset points',
        #             bbox=dict(facecolor='white', alpha=0.7),
        #             fontproperties=font_prop, fontsize=20)
        
        plt.tight_layout()
        
        # If a save path is specified, save the chart.
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return max_idx, avg_cosine_sims[max_idx]
    
    except Exception as e:
        plt.close()  # Close the chart when an error occurs.
        raise Exception(f"Error in plotting: {e}")

plot_layer_wise_cosine_similarity(avg_cosine_sims, model_name, save_path=layer_trend_path + f'/layer_wise_cosine_similarity.png')

#%%
font_prop_large = font_manager.FontProperties(fname=font_path, size=24)
#%%
import matplotlib
        # Set up a two-column legend and adjust the position.
def plot_layer_wise_norms(diff_norms, figsize=(10, 12), cmap_name='Blues', 
                         title=None, save_path=None):
    """
    Visualize the L2 norm of the embedding difference between different layers.
    
    Parameters:
        diff_norms: The list of the L2 norm of the embedding difference of each layer
        figsize: The size of the chart, default is (10, 12)
        cmap_name: The name of the color mapping, default is 'Blues'
        title: The title of the chart, default is None
        save_path: The path to save the chart, default is None
    """
    try:
        # Set color mapping.
        # Create a custom color map using ff8248 orange as the base color.
        base_color = '# ff8248'  # Orange
        rgb_base = matplotlib.colors.to_rgb(base_color)
        
        # Select the layers to display: the first layer, every 5 layers, and the last layer.
        selected_layers = [0]  # First layer.
        selected_layers.extend(range(5, len(diff_norms), 5))  # One every five floors.
        if len(diff_norms) - 1 not in selected_layers:  # Ensure the last layer is included.
            selected_layers.append(len(diff_norms) - 1)
        
        # Create a continuous color mapping: maintain a brighter color system.
        # Create a brighter color gradient.
        brightest_color = tuple(min(1.0, c * 1.5) for c in rgb_base)  # The brightest orange.
        lighter_color = tuple(min(1.0, c * 1.2) for c in rgb_base)  # Brighter orange.
        light_color = rgb_base  # Original orange
        
        # Create a continuous bright color mapping (gradient of three points).
        color_map = matplotlib.colors.LinearSegmentedColormap.from_list(
            'custom_bright_cmap', [brightest_color, lighter_color, light_color], N=256)
        
        # Create a chart.
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create continuous standardized objects using the entire layer range.
        min_layer = 0
        max_layer = len(diff_norms) - 1
        norm = matplotlib.colors.Normalize(vmin=min_layer, vmax=max_layer)
        
        # Draw the curve of the selected layer, using color mapping and normalizers to set the color.
        for layer_idx in selected_layers:
            # Get the color corresponding to the current layer.
            color = color_map(norm(layer_idx))
            ax.plot(diff_norms[layer_idx], color=color, linewidth=3.5)
        
        # Set the x-axis scale.
        level_names = [r'$\mathbf{r}_{\text{2} \leftarrow \text{1}}$', r'$\mathbf{r}_{\text{3} \leftarrow \text{1}}$', r'$\mathbf{r}_{\text{4} \leftarrow \text{1}}$', r'$\mathbf{r}_{\text{5} \leftarrow \text{1}}$']
        ax.set_xticks(np.arange(len(diff_norms[0])))
        ax.set_xticklabels(level_names, fontproperties=font_prop, fontsize=32)
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(ax.get_yticks(), fontproperties=font_prop, fontsize=32)
        
        # Set label.
        # ax.set_xlabel("Level Index", fontproperties=font_prop, fontsize=32)
        ax.set_ylabel("L2 Norm", fontproperties=font_prop, fontsize=36)
        
        # Configure the axis borders and bold all borders.
        ax.spines['left'].set_linewidth(3.5)
        ax.spines['bottom'].set_linewidth(3.5)
        ax.spines['right'].set_linewidth(3.5)
        ax.spines['top'].set_linewidth(3.5)
        
        # Set the y-axis to start from 0.
        y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max)
        
        # Add grid.
        ax.grid(True, linestyle='--', alpha=0.8, linewidth=2.5)
        
        # Create a ScalarMappable object to generate a continuous colorbar.
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        
        # Add a continuous colorbar.
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Layer Index', fontproperties=font_prop, fontsize=36)
        
        # Set the colorbar ticks, using the selected layers as the main tick points.
        cbar.set_ticks(selected_layers)
        cbar.set_ticklabels([str(layer) for layer in selected_layers])
        cbar.ax.tick_params(labelsize=32)
        
        # Set the font of the colorbar tick labels.
        for t in cbar.ax.get_yticklabels():
            t.set_fontproperties(font_prop)
            t.set_fontsize(28)
        
        # Adjust the layout to ensure the colorbar is fully visible.
        plt.tight_layout()
        
        # If a save path is specified, save the chart.
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except Exception as e:
        plt.close()  # Close the chart when an error occurs.
        raise Exception(f"Error in plotting: {e}")

diff_norms = []
for i in range(len(diff_dfs)):
    df_norms = []
    for embedding in diff_dfs[i]['diff_embedding'].values:
        # Calculate the L2 norm of each embedding.
        norm = np.linalg.norm(embedding)
        df_norms.append(norm)
    diff_norms.append(df_norms)

plot_layer_wise_norms(diff_norms, figsize=(12, 6), cmap_name='Blues', title='Custom Title', save_path=layer_trend_path + f'/layer_wise_norms.png')

# #@ TSNE
# df_tsne_full, X_tsne, level_avg_embedding_array = prepare_tsne_data(X, data_information)

# plot_tsne_by_difficulty(
#     df_tsne_full,
#     model_name=model_name,
#     figsize=(12, 8),
#     point_size=30,
#     mean_point_size=1000,
#     alpha=0.7,
#     mean_alpha=1.0,
#     save_path=tsne_path + f'/layer_{layer_number}_tsne.png'
# )

#%%

#@ Calculate the mean steering vector
mean_steering_vectors = []
for i in range(len(diff_dfs)):
    mean_steering_vectors.append(np.mean(diff_dfs[i]['diff_embedding'].values, axis=0))
mean_steering_vectors = np.array(mean_steering_vectors)
mean_steering_vectors[0]

# Print length.
for i in range(len(mean_steering_vectors)):
    print(np.linalg.norm(mean_steering_vectors[i]))

# Save.
np.save(f'{base_asset_path}/mean_steering_vectors.npy', mean_steering_vectors[:-1])
#%%

def plot_effect_list(effect_list, figsize=(10, 6), font_prop=None, save_path=None):
    """
    Visualize the regression effect of different layers.
    
    Parameters:
        effect_list: The list of the regression effect of each layer
        figsize: The size of the chart, default is (10, 6)
        font_prop: The font property, default is None
        save_path: The path to save the chart, default is None (not saved)
    """
    try:
        # Create a chart.
        plt.figure(figsize=figsize)
        
        # Calculate the average effect.
        mean_effect = np.mean(effect_list)
        
        # Draw a bar chart.
        plt.bar(range(len(effect_list)), effect_list, 
                linewidth=2, edgecolor='#515b83', color='#c4d7ef')
        
        # Add average reference line.
        plt.axhline(y=mean_effect, color='#ff8248', linestyle='--', 
                   linewidth=3.5, label='Average $\hat{y}^{(l)}$')
        
        # Set the x-axis scale to display every 5 layers.
        all_layers = range(len(effect_list))
        visible_layers = [i for i in all_layers if i % 5 == 0]
        plt.xticks(visible_layers, [f'{i}' for i in visible_layers], 
                  fontproperties=font_prop, fontsize=32)
        plt.yticks(fontproperties=font_prop, fontsize=32)
        
        # Set label.
        plt.xlabel('Layer Index', fontproperties=font_prop, fontsize=36)
        plt.ylabel('$\hat{y}^{(l)}$', fontproperties=font_prop, fontsize=36, fontweight='bold') # \hat{y}^{(l)} Regression Effect
        
        # Configure the axis borders and bold all borders.
        ax = plt.gca()
        ax.spines['left'].set_linewidth(3.5)
        ax.spines['bottom'].set_linewidth(3.5)
        ax.spines['right'].set_linewidth(3.5)
        ax.spines['top'].set_linewidth(3.5)
        
        # Add grid.
        plt.grid(True, linestyle='--', alpha=0.8, linewidth=2.5)
        
        # Add a legend.
        font_prop_large = font_manager.FontProperties(fname=font_path, size=32)
        plt.legend(prop=font_prop_large)
        
        plt.tight_layout()
        
        # If a save path is specified, save the chart.
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return mean_effect
        
    except Exception as e:
        plt.close()  # Close the chart when an error occurs.
        raise Exception(f"Error in plotting: {e}")

# Calculate regression effect
effect_list = []
for i in range(len(mean_steering_vectors)):
    steering_vector = mean_steering_vectors[i]
    coef = param_list['coef'][i]
    intercept = param_list['intercept'][i]
    # print(coef)
    # print(intercept)
    # Calculate regression performance.
    y_pred = coef @ steering_vector + intercept
    print("Layer ", i, " regression effect: ", y_pred * 0.2)
    effect_list.append(y_pred * 0.2)

# Draw a bar chart.
# mean_effect = np.mean(effect_list)
# plt.bar(range(len(effect_list)), effect_list)
# plt.axhline(y=mean_effect, color='r', linestyle='--', label='Mean Effect')
# plt.legend()
# plt.show()

mean_effect = plot_effect_list(effect_list, font_prop=font_prop, 
                              save_path=layer_trend_path + f'/layer_wise_effect.png')


########################################################
# Save.
# np.save(f'{base_asset_path}/effect_list.npy', effect_list)
# %%
def plot_tsne_by_difficulty(df_tsne, model_name=None, figsize=(12, 8), 
                          point_size=50, mean_point_size=200,
                          alpha=0.7, mean_alpha=1.0,
                          save_path=None):
    """
    Visualize the data points and the corresponding mean points using t-SNE.
    
    Parameters:
        df_tsne: The DataFrame containing the reduced-dimensional data, needs to include columns:
                 ['tsne_x', 'tsne_y', 'difficulty', 'is_mean']
        model_name: The name of the model, used for the title
        figsize: The size of the chart, default is (12, 8)
        point_size: The size of the ordinary data point, default is 50
        mean_point_size: The size of the mean point, default is 200
        alpha: The transparency of the ordinary data point, default is 0.7
        mean_alpha: The transparency of the mean point, default is 1.0
        save_path: The path to save the chart, default is None
    """
    # Set color mapping for different difficulty levels using a blue gradient.
    unique_levels = sorted(df_tsne[~df_tsne['is_mean']]['difficulty'].unique())
    # Use the Blues color map, from light to dark.
    # print(np.linspace(-0.1, 0.6, len(unique_levels)))
    blue_scale = [0.0, 0.12, 0.2, 0.37, 0.6]
    print(blue_scale)
    colors_scatter = plt.cm.Blues(blue_scale)
    colors_mean = plt.cm.Reds(np.linspace(0.05, 0.35, len(unique_levels)))
    color_map_scatter = dict(zip(unique_levels, colors_scatter))
    color_map_mean = dict(zip(unique_levels, colors_mean))
    
    # Create the main chart.
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # First, plot the ordinary data points.
    for level in unique_levels:
        mask = (df_tsne['difficulty'] == level) & (~df_tsne['is_mean'])
        ax.scatter(df_tsne.loc[mask, 'tsne_x'], 
                   df_tsne.loc[mask, 'tsne_y'],
                   c=[color_map_scatter[level]],
                   label=level,
                   alpha=alpha,
                   s=point_size,
                   marker='o',
                   edgecolors='none',
                   linewidth=0,
                   )  # Use circular markers for ordinary points.
    
    # Then plot the mean point.
    for level in unique_levels:
        print(level)
        mask = df_tsne['difficulty'] == f'Mean {level}'
        if mask.any():
            ax.scatter(df_tsne.loc[mask, 'tsne_x'],
                       df_tsne.loc[mask, 'tsne_y'],
                       c=color_map_mean[level],  # Use blue of the corresponding difficulty level.
                       marker='*',  # The mean point is marked with a star symbol.
                       label=f'Mean {level}',
                       alpha=mean_alpha,
                       s=mean_point_size,
                       linewidth=2
                       )  # Bold border.
    
    # Show borders but do not display scales.
    ax.spines['top'].set_linewidth(3.8)
    ax.spines['right'].set_linewidth(3.8)
    ax.spines['bottom'].set_linewidth(3.8)
    ax.spines['left'].set_linewidth(3.8)
    
    # Set blue dashed border
    # border_color = colors_scatter[2]  # Use the third color from the tab10 color map.
    # for spine in ax.spines.values():
    #     spine.set_edgecolor(border_color)
    #     spine.set_linestyle('--')
    #     spine.set_linewidth(3.5)
    
    # Set rounded corners for the border.
    # Note: Matplotlib does not directly support rounded borders, but it can be simulated by adjusting the appearance of the chart.
    # ax.patch.set_facecolor('white')
    # ax.patch.set_alpha(0.8)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    # If a save path is specified, save the main chart.
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # Create separate legend charts.
    figlegend = plt.figure(figsize=(5, 3))
    font_prop_large = font_manager.FontProperties(fname=font_path, size=12)
    
    # Get the legend handles and labels from the main figure.
    handles, labels = ax.get_legend_handles_labels()
    
    # Create a legend in a separate chart.
    legend = figlegend.legend(handles, labels, 
                    title='Difficulty Level', 
                    loc='center',
                    ncol=2,
                    prop=font_prop_large)
    
    # Set the font of the legend title to be consistent with the other labels.
    plt.setp(legend.get_title(), fontproperties=font_prop_large)
    # Save the legend chart.
    if save_path:
        legend_path = save_path.replace('.png', '_legend.png')
        figlegend.savefig(legend_path, dpi=300, bbox_inches='tight')
    
    plt.show()

#@ All layers tsne

files = os.listdir(representation_path)

for idx, file in enumerate(files):
    # if idx != 20:
        # continue
    if idx == 0:
        continue
    
    print("Is processing layer: ", idx)
    print("Processing file: ", file)
    data_path = representation_path + '/' + file
    rep = np.load(data_path)
    print(rep.shape)
    
    X = rep
    df = pd.DataFrame(data_information)
    df = df[['problem', 'level', 'avg_tokens_reasoning']]
    df.loc[df['level'] == 'Level ?', 'level'] = 'Level 3'
    df['embedding'] = list(X)
    df = df.merge(df_accuracy[['problem', 'accuracy']], on='problem', how='left')

    # #@ TSNE
    df_tsne_full, X_tsne, level_avg_embedding_array = prepare_tsne_data(X, data_information)

    plot_tsne_by_difficulty(
        df_tsne_full,
        model_name=model_name,
        figsize=(8, 8),
        point_size=40,
        mean_point_size=500,
        alpha=0.45,
        mean_alpha=1.0,
        save_path=tsne_path + f'/layer_{idx}_tsne.png'
    )
# %%

# plot_tsne_by_difficulty(
#     df_tsne_full,
#     model_name=model_name,
#     figsize=(9, 8),
#     point_size=40,
#     mean_point_size=500,
#     alpha=0.45,
#     mean_alpha=1.0,
#     # save_path=tsne_path + f'/layer_{idx}_tsne.png'
# )