#%%
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("Maxwell-Jia/AIME_2024")
# %%
ds
# %%
df_ds = ds['train'].to_pandas()
# %%
df_ds.head()
# %%
df_ds.rename(columns={'Problem':'problem', 'Solution':'solution', 'Answer':'answer'}, inplace=True)
# %%
df_ds.head()
# %%
import json
with open('AIME2024.json', 'w') as f:
    json.dump(df_ds.to_dict(orient='records'), f, indent=4)
# %%
