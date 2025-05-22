import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap
from sklearn.metrics.pairwise import cosine_similarity

def visualize_level_statistics(df, model_name, group_col='level', value_col='avg_tokens_reasoning', 
                             figsize=(10, 6), save_path=None):
    """
    Calculate and visualize the average token number of different difficulty levels.
    
    Parameters:
        df: The DataFrame containing the analysis data.
        model_name: The name of the model, used for the chart title
        group_col: The column name used for grouping, default is 'level'
        value_col: The column name used for calculating the average value, default is 'avg_tokens_reasoning'
        figsize: The size of the chart, default is (10, 6)
        save_path: The path to save the chart, default is None (not saved)
        
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
    plt.bar(level_avg[group_col], level_avg['mean_tokens'])
    plt.title(f'Average Token Numbers for Each {group_col} ({model_name})')
    plt.xlabel(group_col.title())
    plt.ylabel('Average Token Number')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # If a save path is specified, save the chart.
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return level_avg

# Usage example:
# Basic usage
# level_stats = visualize_level_statistics(df, model_name)

# Custom parameter usage.
# level_stats = visualize_level_statistics(
#     df, 
#     model_name,
#     group_col='level',
#     value_col='avg_tokens_reasoning',
#     figsize=(12, 8),
#     save_path='figures/level_statistics.png'
# )
def visualize_diff_level_statistics(df, model_name, group_col='level', value_col='mean_tokens', 
                                    figsize=(10, 6), save_path=None):
    """
    Calculate and visualize the difference in average token number between different difficulty levels and the first difficulty level.
    
    Parameters:
        df: The DataFrame containing the analysis data.
        model_name: The name of the model, used for the chart title
        group_col: The column name used for grouping, default is 'level'
        value_col: The column name used for calculating the average value, default is 'mean_tokens'
        figsize: The size of the chart, default is (10, 6)
        save_path: The path to save the chart, default is None (not saved)
        
    Returns:
        diff_level_avg: The DataFrame containing the difference in average token number between different difficulty levels and the first difficulty level.
    """
    # Calculate the average number of tokens for each difficulty level.
    level_avg = df.groupby(group_col)[value_col].mean().reset_index()
    level_avg.columns = [group_col, 'mean_tokens']
    
    # Calculate the difference with the first difficulty level.
    base_value = level_avg['mean_tokens'].iloc[0]
    diff_level_avg = level_avg['mean_tokens'] - base_value
    diff_level_avg = diff_level_avg[1:]  # Remove the first value (the difference with itself is 0).
    
    # Visualization
    plt.figure(figsize=figsize)
    plt.plot(range(1, len(diff_level_avg) + 1), diff_level_avg.values, 
             marker='o', label='Differences from First Level')
    plt.title(f'Differences from Level 1 in Average Token Numbers ({model_name})')
    plt.xlabel(f'{group_col.title()} (compared to {level_avg[group_col].iloc[0]})')
    plt.ylabel('Difference in Average Token Number')
    plt.xticks(range(1, len(diff_level_avg) + 1), 
               level_avg[group_col].iloc[1:],
               rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # If a save path is specified, save the chart.
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return pd.DataFrame({
        group_col: level_avg[group_col].iloc[1:],
        'diff_from_first': diff_level_avg
    })

# Usage example:
# diff_stats = visualize_diff_level_statistics(df, model_name)

def plot_regression_results(y_train, y_test, y_train_pred, y_test_pred, 
                          mse_train, r2_train, spearman_train,
                          mse_test, r2_test, spearman_test,
                          model_name,
                          figsize=(12, 6), alpha_value=1, point_size=1,
                          correction_df = None,
                          layer_number=None,
                          save_path=None,
                          indices_train=None,
                          indices_test=None
                          ):
    """
    Visualize the prediction results of the regression model on the training set and the test set.
    
    Parameters:
        y_train: The actual value of the training set
        y_test: The actual value of the test set
        y_train_pred: The predicted value of the training set
        y_test_pred: The predicted value of the test set
        mse_train: The MSE of the training set
        r2_train: The R² of the training set
        spearman_train: The Spearman correlation coefficient of the training set
        mse_test: The MSE of the test set
        r2_test: The R² of the test set
        spearman_test: The Spearman correlation coefficient of the test set
        model_name: The name of the model
        file: The name of the file
        figsize: The size of the chart, default is (12, 6)
        alpha_value: The transparency of the scatter plot, default is 1
        point_size: The size of the scatter plot, default is 1
        correction_df: The correction data frame, default is None
        save_path: The path to save the chart, default is None (not saved)
    """
    plt.figure(figsize=figsize)
    
    # Define a drawing function to reduce code duplication.
    def plot_subplot(position, y_true, y_pred, mse, r2, spearman, set_type, correction_df, indices):
        plt.subplot(1, 2, position)
        
        # Draw a scatter plot.
        if correction_df is not None:
            scatter = plt.scatter(y_true, y_pred, alpha=alpha_value, s=point_size, c=correction_df['is_correct'].iloc[indices], cmap='coolwarm')
            plt.colorbar(scatter, label='Is Correct')
        else:
            scatter = plt.scatter(y_true, y_pred, alpha=alpha_value, s=point_size)
        
        # Draw the diagonal.
        plt.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 
                'r--', alpha=0.3)
        
        # Set labels and titles.
        plt.xlabel('Actual Token Number')
        plt.ylabel('Predicted Token Number')
        plt.title(f'{set_type} Set: Actual vs Predicted\n{model_name} (Layer {layer_number})')
        
        
        # Add evaluation metric text box.
        text_x = y_true.min() + 0.05 * (y_true.max() - y_true.min())
        text_y = y_true.max() - 0.1 * (y_true.max() - y_true.min())
        metrics_text = (f'MSE: {mse:.2f}\n'
                       f'R²: {r2:.4f}\n'
                       f'Spearman: {spearman:.4f}')
        
        plt.text(text_x, text_y, metrics_text,
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Add grid.
        plt.grid(True, linestyle='--', alpha=0.3)
    
    # Draw the training set results.
    plot_subplot(1, y_train, y_train_pred, 
                mse_train, r2_train, spearman_train, 'Training', correction_df, indices_train)
    
    # Plot the test set results.
    plot_subplot(2, y_test, y_test_pred,
                mse_test, r2_test, spearman_test, 'Test', correction_df, indices_test)
    
    plt.tight_layout()
    
    # If a save path is specified, save the chart.
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Usage example:
# metrics_train = {
#     'mse': mse_train,
#     'r2': r2_train,
#     'spearman': spearman_train
# }
# metrics_test = {
#     'mse': mse_test,
#     'r2': r2_test,
#     'spearman': spearman_test
# }
# 
# plot_regression_results(
#     y_train, y_test, y_train_pred, y_test_pred,
#     metrics_train, metrics_test,
#     model_name, file,
#     save_path='figures/regression_results.png'
# )


def plot_layer_wise_metrics(train_metrics, test_metrics, metric_name, 
                          model_name=None, figsize=(12, 6), 
                          train_marker='o', test_marker='s',
                          save_path=None):
    """
    Visualize the change trend of the evaluation metrics of the model on the training set and the test set at different layers.
    
    Parameters:
        train_metrics: The list of evaluation metrics of each layer of the training set
        test_metrics: The list of evaluation metrics of each layer of the test set
        metric_name: The name of the evaluation metric (e.g., 'MSE', 'R²', 'Spearman' etc.)
        model_name: The name of the model, used for the chart title, default is None
        figsize: The size of the chart, default is (12, 6)
        train_marker: The marker style of the training set data points, default is 'o'
        test_marker: The marker style of the test set data points, default is 's'
        save_path: The path to save the chart, default is None (not saved)
        
    Returns:
        None
    """
    plt.figure(figsize=figsize)
    
    def plot_subplot(position, metrics, set_type, marker_style):
        plt.subplot(1, 2, position)
        
        # Draw a trend line.
        x = range(len(metrics))
        plt.plot(x, metrics, marker=marker_style, linestyle='-', 
                markersize=6, label=f'{set_type} {metric_name}')
        
        # Set the x-axis scale.
        plt.xticks(x, [f'Layer {i+1}' for i in x], rotation=45)
        
        # Set labels and titles.
        plt.xlabel('Layer Index')
        plt.ylabel(metric_name)
        title = f'{set_type} Set: Layer-wise {metric_name}'
        if model_name:
            title += f'\n{model_name}'
        plt.title(title)
        
        # Add grid.
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add a legend.
        plt.legend()
        
        # Add maximum and minimum value labels.
        max_idx = np.argmax(metrics)
        min_idx = np.argmin(metrics)
        plt.annotate(f'Max: {metrics[max_idx]:.4f}', 
                    xy=(max_idx, metrics[max_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(facecolor='white', alpha=0.7))
        plt.annotate(f'Min: {metrics[min_idx]:.4f}', 
                    xy=(min_idx, metrics[min_idx]),
                    xytext=(10, -15), textcoords='offset points',
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot the training set results.
    plot_subplot(1, train_metrics, 'Training', train_marker)
    
    # Plot the test set results.
    plot_subplot(2, test_metrics, 'Testing', test_marker)
    
    plt.tight_layout()
    
    # If a save path is specified, save the chart.
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Usage example:
# plot_layer_wise_metrics(
#     train_metrics=train_mse_list,
#     test_metrics=test_mse_list,
#     metric_name='Mean Squared Error',
#     model_name='Qwen-7B',
#     save_path='figures/layer_wise_mse.png'
# )


def plot_cosine_similarity_matrix(diff_embeddings, level_names, 
                                figsize=(10, 8), cmap='viridis',
                                sim_range=(-1, 1), text_threshold=0.5,
                                title=None, save_path=None):
    """
    Visualize the cosine similarity matrix between the difficulty level difference vectors.
    
    Parameters:
        diff_embeddings: The matrix of difference vectors, shape is (n_levels, embedding_dim)
        level_names: The list of difficulty level names
        figsize: The size of the chart, default is (10, 8)
        cmap: The color map of the heatmap, default is 'viridis'
        sim_range: The range of the similarity value, default is (-1, 1)
        text_threshold: The threshold of the text color, default is 0.5
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
    
    # Draw a heatmap.
    im = plt.imshow(cosine_sim_matrix, cmap=cmap, 
                    vmin=sim_range[0], vmax=sim_range[1])
    plt.colorbar(im, label='Cosine Similarity')
    
    # Set the title and tags.
    if title is None:
        title = 'Cosine Similarity Matrix of Difficulty Level Differences'
    plt.title(title)
    
    # Set scale labels.
    plt.xticks(range(len(level_names)), level_names, rotation=45)
    plt.yticks(range(len(level_names)), level_names)
    
    # Add similarity values in each cell.
    for i in range(len(level_names)):
        for j in range(len(level_names)):
            sim_value = cosine_sim_matrix[i, j]
            text_color = "white" if sim_value < text_threshold else "black"
            plt.text(j, i, f'{sim_value:.2f}',
                    ha="center", va="center", 
                    color=text_color)
    
    plt.tight_layout()
    
    # If a save path is specified, save the chart.
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Calculate and return the average similarity.
    mean_similarity = np.mean(cosine_sim_matrix)
    
    return cosine_sim_matrix, mean_similarity

# Usage example:
# Extract the difference vector from the DataFrame.
# diff_embeddings = np.vstack(diff_df['diff_embedding'].values)
# level_names = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
# 
# Basic Usage
# sim_matrix, mean_sim = plot_cosine_similarity_matrix(
#     diff_embeddings=diff_embeddings,
#     level_names=level_names
# )
# 
# Custom Parameter Usage
# sim_matrix, mean_sim = plot_cosine_similarity_matrix(
#     diff_embeddings=diff_embeddings,
#     level_names=level_names,
#     figsize=(12, 10),
#     cmap='RdYlBu',
#     sim_range=(-0.5, 1),
#     text_threshold=0.3,
#     title='Custom Similarity Matrix',
#     save_path='figures/similarity_matrix.png'
# )


def plot_layer_wise_cosine_similarity(avg_cosine_sims, model_name=None, 
                                    figsize=(10, 6), marker='o',
                                    grid=True, save_path=None):
    """
    Visualize the change trend of the average cosine similarity between different layers.
    
    Parameters:
        avg_cosine_sims: The list or array of the average cosine similarity of each layer
        model_name: The name of the model, used for the chart title, default is None
        figsize: The size of the chart, default is (10, 6)
        marker: The marker style of the data points, default is 'o'
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
                markersize=6, label='Average Similarity')
        
        # Set the title.
        title = "Average Cosine Similarity Across Layers"
        if model_name:
            title += f"\n{model_name}"
        plt.title(title)
        
        # Set axis labels.
        plt.xlabel("Layer Index")
        plt.ylabel("Average Cosine Similarity")
        
        # Set the x-axis scale.
        plt.xticks(x, [f'Layer {i+1}' for i in x], rotation=45)
        
        # Add grid.
        if grid:
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add a legend.
        plt.legend()
        
        # Label the maximum and minimum values.
        max_idx = np.argmax(avg_cosine_sims)
        min_idx = np.argmin(avg_cosine_sims)
        
        plt.annotate(f'Max: {avg_cosine_sims[max_idx]:.4f}', 
                    xy=(max_idx, avg_cosine_sims[max_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(facecolor='white', alpha=0.7))
        
        plt.annotate(f'Min: {avg_cosine_sims[min_idx]:.4f}', 
                    xy=(min_idx, avg_cosine_sims[min_idx]),
                    xytext=(10, -15), textcoords='offset points',
                    bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        # If a save path is specified, save the chart.
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return max_idx, avg_cosine_sims[max_idx]
    
    except Exception as e:
        plt.close()  # Close the chart when an error occurs.
        raise Exception(f"绘图过程中发生错误: {e}")

# Usage example:
# Basic Usage
# max_layer, max_sim = plot_layer_wise_cosine_similarity(avg_cosine_sims)
#
# Custom Parameter Usage
# max_layer, max_sim = plot_layer_wise_cosine_similarity(
#     avg_cosine_sims=avg_cosine_sims,
#     model_name='Qwen-7B',
#     figsize=(12, 8),
#     marker='s',
#     grid=True,
#     save_path='figures/layer_cosine_similarity.png'
# )

def plot_layer_wise_norms(diff_norms, figsize=(10, 12), cmap_name='Blues', 
                         title=None, save_path=None):
    """
    Visualize the norm of the embedding difference between different layers.
    
    Parameters:
        diff_norms: The list of the norm of the embedding difference of each layer
        figsize: The size of the chart, default is (10, 12)
        cmap_name: The name of the color map, default is 'Blues'
        title: The title of the chart, default is None
        save_path: The path to save the chart, default is None
    """
    # Set color mapping.
    cmap = get_cmap(cmap_name)
    num_lines = len(diff_norms)
    
    # Create a chart and set a larger right margin to accommodate two columns of legends.
    plt.figure(figsize=figsize)
    
    # Draw the curve for each layer.
    for i in range(num_lines):
        color = cmap(0.3 + 0.7 * i / (num_lines - 1))  # The color gradually changes from light to dark.
        plt.plot(diff_norms[i], label=f"Layer {i+1}", color=color)
    
    # Set the title and tags.
    if title is None:
        title = "Embedding Norm Difference Across Levels"
    plt.title(title)
    plt.xlabel("Level Index")
    plt.ylabel("L2 Norm of Embedding Difference")
    
    # Set the x-axis scale.
    plt.xticks(np.arange(len(diff_norms[0])))
    
    # Set up a two-column legend and adjust the position.
    plt.legend(loc='center left', 
              bbox_to_anchor=(1.05, 0.5),
              ncol=2)  # ncol=2 makes the legend split into two columns.
    
    # Adjust the layout to ensure the legend is fully visible.
    plt.tight_layout()
    
    # If a save path is specified, save the chart.
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Usage example:
# plot_layer_wise_norms(
#     diff_norms=diff_norms,
#     figsize=(12, 8),
#     cmap_name='Blues',
#     title='Custom Title',
#     save_path='figures/layer_norms.png'
# )

def plot_tsne_by_difficulty(df_tsne, model_name=None, figsize=(12, 8), 
                          point_size=50, mean_point_size=200,
                          alpha=0.7, mean_alpha=1.0,
                          save_path=None):
    """
    Visualize the data points and the corresponding mean points using t-SNE.
    
    Parameters:
        df_tsne: The DataFrame containing the reduced-dimensional data, needs to include the columns:
                 ['tsne_x', 'tsne_y', 'difficulty', 'is_mean']
        model_name: The name of the model, used for the chart title
        figsize: The size of the chart, default is (12, 8)
        point_size: The size of the ordinary data points, default is 50
        mean_point_size: The size of the mean points, default is 200
        alpha: The transparency of the ordinary data points, default is 0.7
        mean_alpha: The transparency of the mean points, default is 1.0
        save_path: The path to save the chart, default is None
    """
    # Set color mapping for different difficulty levels.
    unique_levels = sorted(df_tsne[~df_tsne['is_mean']]['difficulty'].unique())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_levels)))
    color_map = dict(zip(unique_levels, colors))
    
    # Create a chart.
    plt.figure(figsize=figsize)
    
    # First, plot the ordinary data points.
    for level in unique_levels:
        mask = (df_tsne['difficulty'] == level) & (~df_tsne['is_mean'])
        plt.scatter(df_tsne.loc[mask, 'tsne_x'], 
                   df_tsne.loc[mask, 'tsne_y'],
                   c=[color_map[level]],
                   label=level,
                   alpha=alpha,
                   s=point_size,
                   marker='o')  # Use circular markers in a regular way.
    
    # Then plot the mean point.
    for level in unique_levels:
        mask = df_tsne['difficulty'] == f'Mean {level}'
        if mask.any():
            plt.scatter(df_tsne.loc[mask, 'tsne_x'],
                       df_tsne.loc[mask, 'tsne_y'],
                       c='black',  # The mean point is represented in black.
                       marker='*',  # The mean point is marked with a star symbol.
                       label=f'Mean {level}',
                       alpha=mean_alpha,
                       s=mean_point_size,
                       edgecolors=color_map[level],  # Use colors corresponding to the difficulty level as borders.
                       linewidth=2)  # Bold border.
    
    # Set the title and tags.
    title = 't-SNE Visualization of Embeddings by Difficulty Level'
    if model_name:
        title += f'\n{model_name}'
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Add a legend and display it in two columns.
    plt.legend(title='Difficulty Level', 
              bbox_to_anchor=(1.05, 1.0),
              loc='upper left',
              ncol=2)  # The legend is divided into two columns.
    
    plt.tight_layout()
    
    # If a save path is specified, save the chart.
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Usage example:
# plot_tsne_by_difficulty(
#     df_tsne=df_tsne_full,
#     model_name='Qwen-7B',
#     figsize=(12, 8),
#     point_size=50,
#     mean_point_size=200,
#     alpha=0.7,
#     mean_alpha=1.0,
#     save_path='figures/tsne_difficulty_with_means.png'
# )