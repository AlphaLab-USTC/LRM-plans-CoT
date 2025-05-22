#%%
import json
import pandas as pd
import numpy as np
import random
# %%
from datasets import load_dataset

# Set a random seed to ensure reproducibility.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load the dataset.
ds = load_dataset("cais/mmlu", "all")

# Randomly select 100 samples from the test set.
random_indices = random.sample(range(len(ds["test"])), 100)
picked_ds = ds["test"].select(random_indices)

# Convert to pandas DataFrame.
df_ds = picked_ds.to_pandas()

#%%
df_ds

#%%
# Only keep the required fields.
remained_df = df_ds[[
    'question',
    'choices',
    'answer',
    'subject'
]]

# Rename column
remained_df.rename(columns={
    'question': 'raw_problem',
    'answer': 'raw_answer'
}, inplace=True)

# Option letter mapping
idx2letter = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}

# Question template
question_template = (
    "What is the correct answer to this question: <problem>\n\nChoices:\n"
    "(A) <choice1>\n"
    "(B) <choice2>\n"
    "(C) <choice3>\n"
    "(D) <choice4>\n"
    "Only put the option letter in the box, e.g. \\boxed{A}. There is only one correct answer"
)

def create_prompt(row):
    prompt = question_template.replace('<problem>', str(row['raw_problem']))
    for i, choice in enumerate(row['choices']):
        prompt = prompt.replace(f'<choice{i+1}>', str(choice))
        print(prompt)
    return prompt

# Construct a new DataFrame.
final_df = pd.DataFrame()
final_df['problem'] = remained_df.apply(create_prompt, axis=1)
final_df['answer'] = remained_df['raw_answer'].map(idx2letter)
final_df['raw_answer'] = remained_df.apply(lambda row: row['choices'][row['raw_answer']], axis=1)
final_df['choices'] = remained_df['choices']
final_df['solution'] = ''  # No explanation, leave it blank.
final_df['domain'] = remained_df['subject']
final_df['subdomain'] = ''  # No subfield, leave blank.
final_df['difficulty'] = ''  # No difficulty, leave it blank.
# Key: Convert the choices column to a list to prevent ndarray from being unserializable.
final_df['choices'] = final_df['choices'].apply(lambda x: list(x))


# Save as a JSON file.
with open('MMLU.json', 'w') as f:
    json.dump(final_df.to_dict(orient='records'), f, indent=4)

# %%