import ollama
import pandas as pd
import re
from tqdm import tqdm

# Ensure you have the model loaded
# Command in terminal: ollama pull deepseek-r1:32b
MODEL = "deepseek-r1:32b"

def get_deepseek_pred(source, target):
    """
    Queries DeepSeek R1 and parses out the reasoning trace (<think> tags).
    """
    prompt = f"""Determine the relation between the following two texts.

Source: "{source}"
Target: "{target}"

- If the Target supports the Source, reply "Support".
- If the Target attacks or contradicts the Source, reply "Attack".
- If they are unrelated or neutral, reply "Unrelated".

Return the and an explanation of maximum 2 sentences in the format: Label:\nExplanation:"""

    try:
        response = ollama.chat(model=MODEL, messages=[
            {'role': 'user', 'content': prompt}
        ])
        
        raw_content = response['message']['content']
        
        # 1. Remove the <think>...</think> block to get the final answer
        # DeepSeek R1 puts its reasoning inside these tags.
        clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).replace("**","").strip().lower()
        
        label = "None"
        if "label: unrelated" in clean_content:
            label = "unrelated"
        elif "label: support" in clean_content:
            label = "support"
        elif "label: attack" in clean_content:
            label = "attack"
        
        # 3. Return a Pandas Series containing both items
        return pd.Series([label, raw_content])
        
    except Exception as e:
        print(f"Error: {e}")
        return pd.Series(["None", f"Error: {str(e)}"])
# --- Test Run ---
# df_test = pd.DataFrame([
#     {"source": "Nuclear energy is dangerous.", "target": "Nuclear accidents are rare.", "label": "attacks"},
#     {"source": "We should buy this car.", "target": "It is very expensive.", "label": "attacks"}
# ])

# print(f"Running baseline with {MODEL}...")
# tqdm.pandas()
# df_test['pred'] = df_test.progress_apply(lambda row: get_deepseek_pred(row['source'], row['target']), axis=1)

# print(df_test)