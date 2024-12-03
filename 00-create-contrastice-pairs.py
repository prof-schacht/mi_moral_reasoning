# %%

import pandas as pd
import requests
import json
from tqdm import tqdm
import os
import openai
import pickle
# %%
df = pd.read_csv('data/cm_train.csv')
df = df[df['is_short'] == True]
# randomly sample 1000 rows
df = df.sample(n=1000, random_state=42)

# load contrastive if exists
if os.path.exists('data/cm_train_contrastive.csv'):
    df_with_contrastive = pd.read_csv('data/cm_train_contrastive.csv')
else:
    df_with_contrastive = None

# instructions
instructions = "Answer whether the following answer is moral (0) or immoral (1). Answer with the number only, not with an explanation:"

df['instructions'] = instructions

# %%

def generate_contrastive_pairs(df):
    instruction_contrastive = "Generate a contrastive example to this sentence. Answer only with the sentence:"
    model = "accounts/fireworks/models/llama-v3p1-405b-instruct"
    
    # Set Fireworks API as the base
    openai.api_base = "https://api.fireworks.ai/inference/v1"
    openai.api_key = "fw_3ZMpHHB51dkY23RhnsUi6t3H"
    results = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"{instruction_contrastive} \"\"\"{row['input']}\"\"\""
                    }
                ],
                max_tokens=4096,
                temperature=0.6
            )
            
            contrastive_example = response.choices[0].message.content
            
            # add contrastive label 1 if label is 0 and 0 if label is 1
            contrastive_label = 1 if row['label'] == 0 else 0

            results.append({
                'input': row['input'],
                'instructions': row['instructions'],
                'label': row['label'],
                'contrastive_input': contrastive_example,
                'contrastive_label': contrastive_label
            })
        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            continue
            
    return pd.DataFrame(results)
#%%

def generate_moral_tupels(df):
    # Tubel element is moral input, second tubel element is contrastive input

    # select all elements with label 0 for the first tubel element and all elements of the same row as moral wrong
    # for the unmoral elements do it the oposite

    df_moral = df[df['label'] == 0]
    df_unmoral = df[df['label'] == 1]

    list_moral_tupels = [(df_moral['instructions'].iloc[i] + df_moral['input'].iloc[i], df_moral['instructions'].iloc[i] + df_moral['contrastive_input'].iloc[i]) for i in range(len(df_moral))]
    list_unmoral_tupels = [(df_unmoral['instructions'].iloc[i] + df_unmoral['contrastive_input'].iloc[i], df_unmoral['instructions'].iloc[i] + df_unmoral['input'].iloc[i]) for i in range(len(df_unmoral))]

    # concatenate the lists
    list_moral_tupels.extend(list_unmoral_tupels)

    return list_moral_tupels

# %%
# Usage
df_with_contrastive = generate_contrastive_pairs(df)

# save the data
df_with_contrastive.to_csv('data/cm_train_contrastive.csv', index=False)

# %%
df_with_contrastive.head()
# %%

list_moral_tupels = generate_moral_tupels(df_with_contrastive)

# %%
# save lst_moral_tupels as pickle
with open('data/cm_train_contrastive_tupels.pkl', 'wb') as f:
    pickle.dump(list_moral_tupels, f)
# %%


