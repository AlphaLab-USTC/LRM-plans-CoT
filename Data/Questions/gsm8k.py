#%%
from datasets import load_dataset
import json

def process_gsm8k_data(split='train'):
    ds = load_dataset("openai/gsm8k", "main")
    df_ds = ds[split].to_pandas()
    
    for i in range(len(df_ds)):
        solution = df_ds.iloc[i]['answer']
        # Separator
        split_solution = solution.split('\n#### ')
        if len(split_solution) > 1:
            solution = split_solution[0]
            answer = split_solution[1]
        else:
            answer = ''
            print("error")
        df_ds.at[i, 'solution'] = solution
        df_ds.at[i, 'answer'] = answer

    df_ds.rename(columns={'question': 'problem'}, inplace=True)
    datasets = df_ds.to_dict(orient='records')

    with open(f'gsm8k_{split}.json', 'w') as f:
        json.dump(datasets, f, indent=4)

# Call the function to process the training set and the test set.
process_gsm8k_data('train')
process_gsm8k_data('test')

# %%
