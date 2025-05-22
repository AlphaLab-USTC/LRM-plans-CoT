import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
import json
from tqdm import tqdm
import pandas as pd
import os
from sklearn.manifold import TSNE
import re
from math_verify import parse, verify
from mathruler.grader import extract_boxed_content, grade_answer

def preprocess_data(generation_path):
    """
    Preprocess the data function, read the JSON file and process the reasoning and answer.
    
    Parameters:
        generation_path: The path of the JSON file
        
    Returns:
        data: The processed data list
    """
    # Read JSON file.
    with open(generation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Display basic information of the data.
    print(f"The type of the data: {type(data)}")
    if isinstance(data, list):
        print(f"The number of data entries: {len(data)}")
        if len(data) > 0:
            print(f"The first data entry example:")
            print(json.dumps(data[0], indent=2, ensure_ascii=False)[:500] + "...")
    elif isinstance(data, dict):
        print(f"The number of data keys: {len(data.keys())}")
        print(f"The data keys: {list(data.keys())}")
    
    # Handle reasoning and answer.
    for idx, d in tqdm(enumerate(data)):
        reasoning_data = d['reasoning']
        
        if 'solution' in d:
            solution = d['solution']
        else:
            solution = ""
        
        if 'answer' in d:
            gt_answer = d['answer']
        else:
            gt_answer = extract_boxed_content(solution)
        data[idx]['gt_answer'] = gt_answer

        gold = parse(f"${gt_answer}$")
            
        data[idx]['answer'] = []
        data[idx]['final_answer'] = []
        data[idx]['is_correct'] = []
        for reasoning in reasoning_data:
            final_answer = extract_boxed_content(reasoning)
            data[idx]['final_answer'].append(final_answer)
            prediction = parse(f"${final_answer}$")
            
            # verify
            if verify(gold, prediction):
                data[idx]['is_correct'].append(1)
            else:
                data[idx]['is_correct'].append(0)
                
            
            think_idx = reasoning.find('</think>')
            if think_idx != -1:
                data[idx]['answer'].append(reasoning[think_idx:])
                reasoning = reasoning[:think_idx]
            else:
                data[idx]['answer'].append("</think>")
        
        data[idx]['accuracy'] = sum(data[idx]['is_correct']) / len(data[idx]['is_correct'])
    
    return data

def analyze_model_accuracy(data, model_name=None, accuracy_threshold=0.01, 
                         samples_per_level=20, save_path=None):
    """
    Analyze the accuracy of the model on different difficulty levels, and extract the wrong samples.
    
    Parameters:
        data: The list or DataFrame containing the problem information
        model_name: The name of the model, used to save the file, default is None
        accuracy_threshold: The threshold for filtering wrong samples, default is 0.01
        samples_per_level: The number of samples to extract per difficulty level, default is 20
        save_path: The path to save the results, default is None
        
    Returns:
        dict: The dictionary containing the following key-value pairs:
            - 'overall_accuracy': The overall accuracy
            - 'level_accuracy': The accuracy of each difficulty level
            - 'wrong_samples': The wrong samples DataFrame
            - 'accuracy_df': The complete accuracy DataFrame
    """
    try:
        # Process according to the input data type.
        if isinstance(data, pd.DataFrame):
            df_accuracy = data
        else:
            # Check the data format and extract the necessary fields.
            required_fields = ['problem', 'accuracy', 'gt_answer', 'final_answer', 'level']
            if not all(field in data[0] for field in required_fields):
                # If the level field is missing, try the basic format.
                required_fields = ['problem', 'accuracy', 'gt_answer', 'final_answer']
                if not all(field in data[0] for field in required_fields):
                    raise ValueError("The data format is incorrect, missing the necessary fields")
                
                # Create a basic DataFrame.
                df_accuracy = pd.DataFrame([
                    [d['problem'], d['accuracy'], d['gt_answer'], d['final_answer']]
                    for d in data
                ], columns=['problem', 'accuracy', 'gt_answer', 'final_answer'])
            else:
                # Create a complete DataFrame that includes difficulty levels.
                df_accuracy = pd.DataFrame([
                    [d['problem'], d['level'], d['accuracy'], d['gt_answer'], d['final_answer']]
                    for d in data
                ], columns=['problem', 'level', 'accuracy', 'gt_answer', 'final_answer'])
        
        # Calculate overall accuracy.
        overall_accuracy = df_accuracy['accuracy'].mean()
        print(f"The overall accuracy: {overall_accuracy:.4f}")
        
        # If there is difficulty level information, calculate the grading accuracy.
        level_accuracy = None
        if 'level' in df_accuracy.columns:
            # Handle special difficulty level markers.
            df_accuracy.loc[df_accuracy['level'] == 'Level ?', 'level'] = 'Level 3'
            
            # Calculate the accuracy for each difficulty level.
            level_accuracy = df_accuracy.groupby('level')['accuracy'].mean()
            print("\nThe accuracy of each difficulty level:")
            print(level_accuracy)
        
        # Extract error samples.
        df_wrong = df_accuracy.query(f"accuracy < {accuracy_threshold}")
        
        # If there is difficulty level information and a specified sample size, conduct stratified sampling.
        if 'level' in df_accuracy.columns and samples_per_level:
            print("\nThe number of wrong samples for each difficulty level:")
            print(df_wrong.groupby('level').size())
            
            # Ensure that there are enough samples for each difficulty level.
            df_wrong = df_wrong.groupby('level').apply(
                lambda x: x.sample(n=min(len(x), samples_per_level))
                if len(x) > 0 else x
            ).reset_index(drop=True)
        
        # Save error samples.
        if save_path and model_name:
            save_file = f'{save_path}/{model_name}_wrong_problems.csv'
            df_wrong.to_csv(save_file, index=False)
            print(f"\nThe wrong samples have been saved to: {save_file}")
        
        return overall_accuracy, level_accuracy, df_wrong, df_accuracy
        
    except Exception as e:
        raise Exception(f"An error occurred during the analysis: {e}")

# Usage example:
# try:
# Basic Usage
#     results = analyze_model_accuracy(
#         data=data,
#         model_name='Qwen-7B',
#         accuracy_threshold=0.01,
#         samples_per_level=20,
#         save_path='results'
#     )
#     
# Access Results
#     overall_acc = results['overall_accuracy']
#     level_acc = results['level_accuracy']
#     wrong_samples = results['wrong_samples']
#     
# except Exception as e:
# print(f"Error: {e}")

def analyze_token_numbers(data, tokenizer, model_name=None, save_path=None):
    """
    Analyze the number of tokens in the data, if the cache file exists, read it directly, otherwise calculate and save it.
    
    Parameters:
        data: The processed data list
        tokenizer: The tokenizer for encoding text
        model_name: The name of the model
        save_path: The path to save/read the analysis results
        
    Returns:
        data_information: The list containing the token analysis results
    """
    # If the cache file exists, read it directly.
    if save_path:
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                print(f"Found the cache file: {save_path}, directly read the analysis results")
                return json.load(f)
        except FileNotFoundError:
            print(f"The cache file is not found, start calculating the token number analysis...")
    
    # Calculate the number of tokens.
    data_information = []
    for d in tqdm(data):
        problem = d['problem']
        if 'level' in d:
            level = d['level']
        else:
            level = 'Level ?'
        num_tokens_reasoning = []
        num_tokens_answer = []
        
        for reasoning in d['reasoning']:
            encoded = tokenizer.encode(reasoning)
            num_tokens_reasoning.append(len(encoded))
        
        for answer in d['answer']:
            encoded = tokenizer.encode(answer)
            num_tokens_answer.append(len(encoded))
            
        data_information.append({
            'problem': problem,
            'level': level,
            'num_tokens_reasoning': num_tokens_reasoning,
            'num_tokens_answer': num_tokens_answer,
            'avg_tokens_reasoning': sum(num_tokens_reasoning) / len(num_tokens_reasoning),
            'avg_tokens_answer': sum(num_tokens_answer) / len(num_tokens_answer)
        })
    
    # Save the analysis results.
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data_information, f, ensure_ascii=False, indent=2)
        
    print(f"Analysis completed, the results have been saved to: {save_path}")
    return data_information

# Usage example:
# data_information = analyze_token_numbers(
#     data=data,
#     tokenizer=tokenizer,
#     model_name=model_name,
#     save_path=f'Data_gen/Info/{model_name}_data_information_split.json'
# )

def check_and_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"Generate the directory: {dir_path}")
    else:
        print(f"Warning: {dir_path} already exists")
    return dir_path

def prepare_save_path(base_asset_path):
    check_and_create_dir(base_asset_path)
    grouped_token_number_path = f"{base_asset_path}/grouped_token_number"
    check_and_create_dir(grouped_token_number_path)
    layer_wise_regression_path = f"{base_asset_path}/layer_wise_regression"
    check_and_create_dir(layer_wise_regression_path)
    layer_wise_cosine_similarity_path = f"{base_asset_path}/layer_wise_cosine_similarity"
    check_and_create_dir(layer_wise_cosine_similarity_path)
    layer_trend_path = f"{base_asset_path}/layer_trend"
    check_and_create_dir(layer_trend_path)  
    tsne_path = f"{base_asset_path}/tsne"
    check_and_create_dir(tsne_path)
    layer_wise_coefficients_path = f"{base_asset_path}/layer_wise_coefficients"
    check_and_create_dir(layer_wise_coefficients_path)
    return base_asset_path, grouped_token_number_path, layer_wise_regression_path, layer_wise_cosine_similarity_path, layer_trend_path, tsne_path, layer_wise_coefficients_path

def process_data_information_frame(data_information, columns=['problem', 'level', 'avg_tokens_reasoning', 'avg_tokens_answer']):
    """
    Process the analysis data, create a DataFrame and clean the data
    
    Parameters:
        data_information: The list containing the token analysis results
        columns: The list of column names to keep, default is ['problem', 'level', 'avg_tokens_reasoning', 'avg_tokens_answer']
        
    Returns:
        df: The processed DataFrame
    """
    # Create a DataFrame.
    df = pd.DataFrame(data_information)
    
    # Only keep the specified columns.
    df = df[columns]
    
    # Change 'Level ?' to 'Level 3'.
    if 'level' in columns:
        df.loc[df['level'] == 'Level ?', 'level'] = 'Level 3'
    
    print(f"The number of data entries after processing: {len(df)}")
    
    return df

# Usage example:
# Use default columns.
# df = process_data_informationframe(data_information)

# Or specify a custom column.
# df = process_data_informationframe(data_information, columns=['problem', 'level', 'avg_tokens_reasoning'])


# Use example.
# data = preprocess_data(generation_path)

def select_and_save_problems(selected_indices, problem_path, save_path=None, 
                           filename="selected_problems.json"):
    """
    Select problems from the problem dataset based on the given index list and save them.
    
    Parameters:
        selected_indices: The list of indices of the selected problems
        problem_path: The path of the original problem dataset JSON file
        save_path: The path of the directory to save the results, default is None (not saving)
        filename: The name of the file to save, default is "selected_problems.json"
        
    Returns:
        selected_problems: The list of selected problems
    """
    try:
        # Verify input.
        if not selected_indices:
            raise ValueError("The selected index list is empty")
        
        if not os.path.exists(problem_path):
            raise FileNotFoundError(f"The problem dataset file is not found: {problem_path}")
        
        # Read the original question dataset.
        print(f"Reading the original question dataset: {problem_path}")
        with open(problem_path, "r", encoding='utf-8') as f:
            data_problems = json.load(f)
            
        # Verify if the index is valid.
        max_index = len(data_problems) - 1
        invalid_indices = [i for i in selected_indices if i < 0 or i > max_index]
        if invalid_indices:
            raise ValueError(f"Invalid indices found: {invalid_indices}")
            
        # Choose a question.
        selected_problems = [data_problems[i] for i in selected_indices]
        print(f"Selected {len(selected_problems)} problems")
        
        # Save the results.
        if save_path:
            # Ensure the save directory exists.
            os.makedirs(save_path, exist_ok=True)
            
            # Build a complete save path.
            full_save_path = os.path.join(save_path, filename)
            
            # Save the file.
            print(f"Saving the selected problems to: {full_save_path}")
            with open(full_save_path, "w", encoding='utf-8') as f:
                json.dump(selected_problems, f, ensure_ascii=False, indent=2)
            print("Saving completed")
            
        return selected_problems
        
    except Exception as e:
        raise Exception(f"An error occurred when selecting and saving the problems: {e}")

# Usage example:
# try:
# Basic Usage
#     selected_problems = select_and_save_problems(
#         selected_indices=selected_problem_list,
#         problem_path="/path/to/problems.json",
#         save_path="results",
#         filename="level1_problems.json"
#     )
#     
# Only select, do not save.
#     selected_problems = select_and_save_problems(
#         selected_indices=selected_problem_list,
#         problem_path="/path/to/problems.json"
#     )
#     
# except Exception as e:
# print(f"Error: {e}")



def regression_at_each_layer(X_train, X_test, y_train, y_test, model):
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on training and testing sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Compute evaluation metrics for the training set
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    spearman_train, _ = spearmanr(y_train, y_train_pred)
    print(f"Training MSE: {mse_train:.2f}")
    print(f"Training R²: {r2_train:.4f}")
    print(f"Training Spearman correlation: {spearman_train:.4f}")
    
    # Compute evaluation metrics for the testing set
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    spearman_test, _ = spearmanr(y_test, y_test_pred)
    print(f"Testing MSE: {mse_test:.2f}")
    print(f"Testing R²: {r2_test:.4f}")
    print(f"Testing Spearman correlation: {spearman_test:.4f}")
    
    return y_train_pred, y_test_pred, mse_train, r2_train, spearman_train, mse_test, r2_test, spearman_test


def construct_diff_df(df, X):
    # df['embedding'] = list(X)

    # Group by level and calculate the average embedding for each level.
    level_avg_embedding = df.groupby('level')['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()

    print("Shape of Average Embeddings for Each Level:")
    for _, row in level_avg_embedding.iterrows():
        print(f"{row['level']}: {row['embedding'].shape}")


    # Calculate the embedding difference between adjacent difficulty levels.
    level_diff_embeddings = []      
    level_names = []

    # Ensure that level_avg_embedding is sorted by level.
    level_avg_embedding = level_avg_embedding.sort_values('level')

    # Constructing difference embedding.
    for i in range(1, len(level_avg_embedding)):
        current_level = level_avg_embedding.iloc[i]['level']
        previous_level = level_avg_embedding.iloc[0]['level']
        
        current_embedding = level_avg_embedding.iloc[i]['embedding']
        previous_embedding = level_avg_embedding.iloc[0]['embedding']
        
        # Calculate the difference.
        diff_embedding = current_embedding - previous_embedding
        
        level_diff_embeddings.append(diff_embedding)
        level_names.append(f"{current_level} - {previous_level}")

    # Create a DataFrame for the difference embedding.
    diff_df = pd.DataFrame({
        'level_diff': level_names,
        'diff_embedding': level_diff_embeddings
    })
    return diff_df, level_names

def prepare_tsne_data(X, data_information, n_components=2, random_state=42):
    """
    Prepare the data for t-SNE visualization, including calculating the average embedding vector and dimensionality reduction.
    
    Parameters:
        X: The original embedding vector matrix
        data_information: The list or dictionary containing the problem information
        n_components: The target dimension of t-SNE dimensionality reduction, default is 2
        random_state: The random seed, default is 42
        
    Returns:
        df_tsne_full: The DataFrame containing the original points and the mean points, including the columns:
                      ['tsne_x', 'tsne_y', 'difficulty', 'is_mean']
        level_avg_embedding: The average embedding vector for each difficulty level
    """
    try:
        # Create a basic DataFrame.
        df = pd.DataFrame(data_information)
        
        # Keep only the necessary columns and handle special cases.
        df = df[['problem', 'level', 'avg_tokens_reasoning']]
        df.loc[df['level'] == 'Level ?', 'level'] = 'Level 3'
        
        # Add embedding to DataFrame.
        df['embedding'] = list(X)
        
        # Calculate the average embedding for each difficulty level.
        level_avg_embedding = (df.groupby('level')['embedding']
                             .apply(lambda x: np.mean(np.vstack(x), axis=0))
                             .reset_index())
        level_avg_embedding_array = np.vstack(level_avg_embedding['embedding'].values)
        
        # Add the average embedding to X.
        X_combined = np.vstack((X, level_avg_embedding_array))
        
        # Perform t-SNE dimensionality reduction.
        tsne = TSNE(n_components=n_components, random_state=random_state)
        X_tsne = tsne.fit_transform(X_combined)
        
        # Create a DataFrame of raw data points.
        df_tsne = pd.DataFrame(
            X_tsne[:-len(level_avg_embedding_array)],
            columns=['tsne_x', 'tsne_y']
        )
        df_tsne['difficulty'] = df['level']
        
        # Create a DataFrame for average embedding points.
        df_tsne_means = pd.DataFrame(
            X_tsne[-len(level_avg_embedding_array):],
            columns=['tsne_x', 'tsne_y']
        )
        df_tsne_means['difficulty'] = [
            f'Mean {level}' for level in sorted(df['level'].unique())
        ]
        
        # Merge the data and add an identifier column.
        df_tsne_full = pd.concat([df_tsne, df_tsne_means], ignore_index=True)
        df_tsne_full['is_mean'] = [False] * len(df_tsne) + [True] * len(df_tsne_means)
        
        return df_tsne_full, X_tsne, level_avg_embedding_array
        
    except Exception as e:
        raise Exception(f"An error occurred during the data processing: {e}")

# Usage example:
# try:
# Prepare data
#     df_tsne_full, level_avg_embedding = prepare_tsne_data(
#         X=X,
#         data_information=data_information,
#         n_components=2,
#         random_state=42
#     )
#     
# Visualize Data
#     plot_tsne_by_difficulty(
#         df_tsne=df_tsne_full,
#         model_name='Qwen-7B',
#         figsize=(12, 8),
#         point_size=50,
#         mean_point_size=200
#     )
#     
# except Exception as e:
# print(f"Error: {e}")


# @ tsne dimensionality reduction

# file = representation_files[40]
# data_path = representation_path + '/' + file
# X = np.load(data_path)

# df = pd.DataFrame(data_information)

# Keep only the necessary columns.
# df = df[['problem', 'level', 'avg_tokens_reasoning']]
# df.loc[df['level'] == 'Level ?', 'level'] = 'Level 3'


# Add the mean embedding of each group at the end of X.
# df['embedding'] = list(X)
# Calculate the average embedding for each difficulty level.
# level_avg_embedding = df.groupby('level')['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()
# level_avg_embedding = np.vstack(level_avg_embedding['embedding'].values)


# Add the average embedding to the last column of X.
# X = np.vstack((X, level_avg_embedding))


# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(X)

# Create a DataFrame to store the dimensionality-reduced data.
# df_tsne = pd.DataFrame(X_tsne[:-len(level_avg_embedding)], columns=['tsne_x', 'tsne_y'])  # Original data points
# df_tsne['difficulty'] = df['level']  # Difficulty labels from the original data

# Create a DataFrame for average embeddings
# df_tsne_means = pd.DataFrame(X_tsne[-len(level_avg_embedding):], columns=['tsne_x', 'tsne_y'])  # Average embedding points
# df_tsne_means['difficulty'] = [f'Mean {level}' for level in sorted(df['level'].unique())]  # Add "Mean Level X" labels

# Merge raw data points and average points
# df_tsne_full = pd.concat([df_tsne, df_tsne_means], ignore_index=True)

# Add an identifier column to distinguish whether it is an average point.
# df_tsne_full['is_mean'] = [False] * len(df_tsne) + [True] * len(df_tsne_means)
