from re import U

import pandas as pd

CORPUS_PATH = "/home/henrike/ARG-NLI_project/code/ID_model_training/corpus_nli_preds_threshold_hard_0.999_noWebis_cleaned.csv"
UNRELATED_PATH = "/home/henrike/ARG-NLI_project/code/ID_model_training/unrelated_only.csv"


corpus_df = pd.read_csv(CORPUS_PATH)
unrelated_df = pd.read_csv(UNRELATED_PATH)
full_df = pd.concat([corpus_df[["ID", "antecedent", "consequent", "relation"]], unrelated_df[["ID", "antecedent", "consequent", "relation"]]])

df_full = full_df.rename(columns={"antecedent":"source", "consequent":"target", "relation":"relation_type"})
df_full["relation_type"] = ["unrelated" if label=="neutral" else label for label in df_full["relation_type"]]
print(df_full.value_counts("relation_type"))


def create_balanced_subset(df, n_per_class=2000):
    # Separate the classes
    support = df[df['relation_type'] == 'support']
    attack = df[df['relation_type'] == 'attack']
    neutral = df[df['relation_type'] == 'unrelated']
    
    # Randomly sample from each
    # (Using random_state=42 for reproducibility)
    s_sample = support.sample(n=n_per_class, random_state=42)
    a_sample = attack.sample(n=n_per_class, random_state=42)
    n_sample = neutral.sample(n=n_per_class, random_state=42)
    
    # Combine and shuffle
    df_balanced = pd.concat([s_sample, a_sample, n_sample]).sample(frac=1).reset_index(drop=True)
    
    print(f"Balanced Dataset Created: {len(df_balanced)} rows.")
    return df_balanced

def make_train_test_split(df):
        full_df = df#self.parse_directory(file_path)
        
        train = full_df.sample(frac=0.8,random_state=200)
        test = full_df.drop(train.index)
        
        print(f"""The train sample contains:
              - {train.value_counts("relation_type")["support"]} Support pairs,
              - {train.value_counts("relation_type")["attack"]} Attack pairs,
              - {train.value_counts("relation_type")["unrelated"]} Unrelated pairs,""")
        
        print(f"""The train sample contains:
              - {test.value_counts("relation_type")["support"]} Support pairs,
              - {test.value_counts("relation_type")["attack"]} Attack pairs,
              - {test.value_counts("relation_type")["unrelated"]} Unrelated pairs,""")
        
        train.to_csv("MyDataset/train.csv", index=False)
        test.to_csv("MyDataset/test.csv", index=False)

# Apply it
df_deepseek_input = create_balanced_subset(df_full, n_per_class=2000)
make_train_test_split(df_deepseek_input)