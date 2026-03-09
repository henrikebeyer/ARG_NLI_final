import json
import ollama
import pandas as pd
import re
from tqdm import tqdm
import os

INPUT_FILE = "MyDataset/test.csv"
OUTPUT_FILE = "llama_predictions_output_test.csv"
MODEL_NAME = "llama3.1:8b"

def get_llama_prediction(source, target):
    prompt = f"""
    Analyze the argumentative relationship between the following two sentences.
    
    Source: "{source}"
    Target: "{target}"
    
    Task:
    1. Classify the relationship as: support, attack, or unrelated.
    2. Provide a 1-2 sentence explanation of your logic.
    
    Output strictly in JSON format with keys "label" and "reasoning".
    Example: {{"label": "support", "reasoning": "The target provides evidence for the claim in the source."}}
    """
    
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt, format='json')
        result = json.loads(response['response'])
        return result.get('label'), result.get('reasoning')
    except Exception as e:
        return "error", str(e)

# --- EXECUTION ---
df = pd.read_csv(INPUT_FILE)

# Check if we are resuming from a previous crash
if os.path.exists(OUTPUT_FILE):
    df_output = pd.read_csv(OUTPUT_FILE)
    start_index = len(df_output)
    print(f"Resuming from index {start_index}...")
else:
    df_output = pd.DataFrame(columns=list(df.columns) + ['llm_prediction', 'llm_explanation'])
    start_index = 0

for i in tqdm(range(start_index, len(df))):
    row = df.iloc[i]
    label, reasoning = get_llama_prediction(row['source'], row['target'])
    
    # Append new data
    new_row = row.to_dict()
    new_row['llm_prediction'] = label
    new_row['llm_explanation'] = reasoning
    
    # Save to CSV immediately to prevent data loss
    pd.DataFrame([new_row]).to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)

print(f"Done! Results saved to {OUTPUT_FILE}")
