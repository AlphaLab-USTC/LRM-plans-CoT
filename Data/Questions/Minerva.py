#%%
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# %%
# Read the JSON file of Minerva.
with open('minerva.json', 'r') as f:
    minerva_data = json.load(f)

# %%
print(minerva_data[0])

# %%
new_minerva_data = []
for question in minerva_data:
    new_question = {}
    new_question['question'] = question['question']
    new_question['answer'] = question['answer'][0]
    new_question['options'] = question['options']
    new_minerva_data.append(new_question)

# %%
from math_verify import parse, verify
from mathruler.grader import extract_boxed_content, grade_answer
# %%
ans1 = parse("$2.19 \\times 10^6$")
ans2 = parse("$2.19 \\times 10^{6}$")

res = verify(ans1, ans2)
print(res)
# %%
ans1 = parse("$4080\n$")
ans2 = parse("$4080\n$")

res = verify(ans1, ans2)
print(res)
# %%
