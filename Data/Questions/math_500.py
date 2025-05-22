#%%
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/MATH-500")
# %%
ds
# %%
ds['test'][0]
# %%
df_ds = ds['test'].to_pandas()
# %%
df_ds.head()
# %%
# Remove unique_id.
df_ds = df_ds.drop(columns=['unique_id'])
# %%
df_ds.head()
#%% 
df_ds.rename(columns={'subject':'type'}, inplace=True)
# %%
datasets = df_ds.to_dict(orient='records')
# %%
datasets[0]
# %%
len(datasets)
# %%
import json
with open('math_500.json', 'w') as f:
    json.dump(datasets, f,indent=4)
# %%
#%%
# Read math_500.json.
with open('math_500.json', 'r') as f:
    datasets = json.load(f)
# %%
datasets[0]
# %%
