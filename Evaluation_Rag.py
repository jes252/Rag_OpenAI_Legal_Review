#!/usr/bin/env python
# coding: utf-8

# In[6]:


from datasets import load_dataset
 

import pandas as pd
from datasets import Dataset, Value, Sequence
df = pd.read_csv("data.csv")

df['contexts'] = df['contexts'].apply(lambda x: eval(x) if isinstance(x, str) else [])


# In[7]:


from datasets import Dataset 

import numpy as np

def replace_none_with_default(lst, default='None'):
    return [x if x is not None and x==x else default for x in lst] # x==x is used to check for nan values 

data_dict = {
    'question': replace_none_with_default(df['question'].tolist()),
    'ground_truth': replace_none_with_default(df['ground_truth'].tolist()),
    'contexts': replace_none_with_default(df['contexts'].tolist()),
    'answer': replace_none_with_default(df['answer'].tolist()),
    'evolution_type': replace_none_with_default(df['evolution_type'].tolist()),
    'episode_done': replace_none_with_default(df['episode_done'].tolist())
}

dataset = Dataset.from_dict(data_dict)


# In[8]:


import os

os.environ["OPENAI_API_KEY"] = "xxxxxxx"
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
result = evaluate(
    dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

print(result)


# In[9]:


import matplotlib.pyplot as plt

data = result

# Create bar chart
plt.bar(data.keys(), data.values())
plt.title('Bar Chart')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.xticks(rotation=45) # This makes the x-axis labels easier to read
plt.show()


# In[9]:




