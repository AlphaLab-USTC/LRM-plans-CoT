#%%
from datasets import load_dataset
import pandas as pd
import numpy as np
import random
import json
# Set a random seed to ensure reproducibility.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
#%%
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", token="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")
# %%
# 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
ds
# %%
df_ds = ds['train'].to_pandas()

#%%
# Extract key fields.
remained_df = df_ds[[
    'Question', 
    'Correct Answer',
    'Incorrect Answer 1',
    'Incorrect Answer 2',
    'Incorrect Answer 3',
    'Explanation',
    'High-level domain',
    'Subdomain',
    'Question Difficulty_EV_1'
]]

#%%
# Rename column
remained_df.rename(columns={
    'Question': 'raw_problem',
    'Correct Answer': 'raw_answer',
    'Explanation': 'solution',
    'High-level domain': 'domain',
    'Subdomain': 'subdomain',
    'Question Difficulty_EV_1': 'difficulty'
}, inplace=True)

#%%
# Create a list of options and shuffle it.
def create_shuffled_choices(row):
    choices = [
        row['raw_answer'],
        row['Incorrect Answer 1'],
        row['Incorrect Answer 2'],
        row['Incorrect Answer 3']
    ]
    # Record the index of the correct answer.
    correct_idx = 0
    # Shuffle the options.
    shuffled_indices = list(range(4))
    random.shuffle(shuffled_indices)
    # Get the new index of the correct answer after shuffling.
    new_correct_idx = shuffled_indices.index(0)
    # Rearrange the options according to the shuffled order.
    shuffled_choices = [choices[i] for i in shuffled_indices]
    return {
        'choices': shuffled_choices,
        'correct_idx': new_correct_idx
    }

#%%
# Apply shuffle and add a new column.
shuffled_data = remained_df.apply(create_shuffled_choices, axis=1)
remained_df['choices'] = shuffled_data.apply(lambda x: x['choices'])
remained_df['correct_idx'] = shuffled_data.apply(lambda x: x['correct_idx'])

#%%
# Delete the original incorrect answer column.
remained_df = remained_df.drop([
    'Incorrect Answer 1',
    'Incorrect Answer 2',
    'Incorrect Answer 3'
], axis=1)


question_template = ("What is the correct answer to this question: <problem>"
                     "\n\nChoices:\n"
                     "(A) <choice1>\n"
                     "(B) <choice2>\n"
                     "(C) <choice3>\n"
                     "(D) <choice4>\n"
                    "Only put the option letter in the box, e.g. \\boxed{A}. There is only one correct answer"
                     )
#%%
def create_prompt(row):
    prompt = question_template.replace('<problem>', row['raw_problem'])
    for i, choice in enumerate(row['choices']):
        prompt = prompt.replace(f'<choice{i+1}>', choice)
    return prompt

remained_df['problem'] = remained_df.apply(create_prompt, axis=1)
# The answer is a mapping of 0-3 to (A), (B), (C), (D).
remained_df['answer'] = remained_df['correct_idx'].map({0:'(A)', 1:'(B)', 2:'(C)', 3:'(D)'})
#%%
# Remove unnecessary columns and reorder
final_df = remained_df[['problem', 'answer', 'raw_answer', 'choices', 'solution', 'domain', 'subdomain', 'difficulty']]
final_df

#%%
# Save as a JSON file.
with open('GPQA_diamond.json', 'w') as f:
    json.dump(final_df.to_dict(orient='records'), f, indent=4)

#%% 

#%%
# # %%
# df_ds.head()
# # %%
# remained_df = df_ds[['Question', 'Explanation', 'Correct Answer']]
# # %%
# remained_df.rename(columns={'Question':'problem', 'Correct Answer':'answer', 'Explanation':'solution'}, inplace=True)
# # %%
# remained_df.head()
# # %%
# import json
# with open('GPQA_diamond.json', 'w') as f:
#     json.dump(remained_df.to_dict(orient='records'), f, indent=4)
# # %%
# Read JSON file
# with open('GPQA_diamond.json', 'r') as f:
#     data = json.load(f)
# # %%
# data
# # %%
# len(data)
# # %%
