import pandas as pd
import math

qwen_path = "/home/oenni/Dokumente/NLI-Argumentation-project/NLI_labeling/full_corpus_qwen2_predictions.tsv"
gemma_path = "/home/oenni/Dokumente/NLI-Argumentation-project/NLI_labeling/full_corpus_gemma3-4b_predictions.tsv"
llama_path = "/home/oenni/Dokumente/NLI-Argumentation-project/NLI_labeling/full_corpus_llama3.1-8b_predictions.tsv"
deberta_path = "/home/oenni/Dokumente/NLI-Argumentation-project/NLI_labeling/full_corpus_deberta_large_v3_nli_tasksource_predictions.tsv"

paths = [qwen_path,
         gemma_path,
         llama_path,
         deberta_path]

df_list = []
pred_cols = []
columns = ["nli_label_qwen2"]

for path in paths:
    df = pd.read_csv(path, sep="\t")
    df.rename(columns={"qwen2-preds":"nli_label_qwen2", "nli_label":"nli_label_llama3.1-8b"}, inplace=True)
    column = list(df.columns)[-1]
    columns.append(column)
    df_list.append(df)
    pred_cols.append(df[column])

target_list = df_list[:1]+pred_cols[1:]
all_pred_df = pd.concat(target_list, axis=1)

columns = columns[:1] + columns[2:-1]
print(columns)
for column in columns:
    all_pred_df[column] = [label.replace("<<","").replace(">>","") for label in all_pred_df[column]]

majority_label = []
gold = []
majority_votes = {"simple_majority":[],
                  2:[], 
                  3:[], 
                  4:[]
                  }

for index, row in all_pred_df.iterrows():
    classifs = [row[col] for col in columns]
    count_dict = {"entailment":classifs.count("entailment"),
                "contradiction":classifs.count("contradiction"),
                "neutral":classifs.count("neutral")}
    
    #print(count_dict)
    max_label = max(count_dict, key=count_dict.get)
    max_agree = max(count_dict.values())
    # if max_agree == 2:
    #     print(count_dict, max_label)
    # print(max_agree)

    majority_votes["simple_majority"].append(max_label)


    for threshold in [2,3,4]:
        if max_agree >= threshold:
            majority_votes[threshold].append(max_label)
        else:
            majority_votes[threshold].append(None)


out_path = "/home/oenni/Dokumente/NLI-Argumentation-project/NLI_labeling/NLI_dist"

for key, value in majority_votes.items():
    all_pred_df[key] = value
    all_pred_df.value_counts(key).to_csv(f"{out_path}/{key}_ensemble_NLI_dist.csv", index=False)
    filter_df = all_pred_df.loc[all_pred_df["relation"].isin(["support", "attack"])].dropna(subset=key)

    filter_df.value_counts("Corpus_ID").to_csv(f"{out_path}/{key}_ensemble_Corpus_dist_argOnly.csv", index=False)

all_pred_df.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/corpus/full_corpus_ensemble_preds.tsv", sep="\t", index=False)