import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
import math
import itertools

annot_file = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/full_annotation_df.tsv"

data = pd.read_csv(annot_file, sep="\t")
data["Corpus_ID"] = [str(id).split("_")[0] for id in data["ID"]]
data["relation"] = [relation.replace("supports", "support").replace("attacks", "attack").lower() for relation in data["relation"]]
# print(data["relation"].unique())
# for relation in ["support", "attack", "neutral"]:
#     print(relation)
#     print(data.loc[data["relation"] == relation].value_counts("Corpus_ID"))

dir = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/noUnrelated"

def weighted_majority_vote(df, cols, weights):
    # Normalize weights for the current subset
    norm_weights = [w / sum(weights) for w in weights]
    votes = df[cols]
    weighted_counts = pd.DataFrame(0.0, index=votes.index, columns=['entailment', 'contradiction', 'neutral'])
    for col, w in zip(cols, norm_weights):
        for label in ['entailment', 'contradiction', 'neutral']:
            weighted_counts[label] += (votes[col] == label) * w
    max_label = weighted_counts.idxmax(axis=1)
    return max_label, df["nli"]

def majority_vote(df, cols, threshold):
    votes = df[cols]
    counts = votes.apply(lambda x: x.value_counts(), axis=1).fillna(0)
    max_label = counts.idxmax(axis=1)
    max_agree = counts.max(axis=1)
    mask = max_agree >= threshold
    return max_label[mask], df.loc[mask, "nli"]

df_list = []
pred_list = []
for file in os.listdir(dir):
    if file.startswith("all_"):
        file_path = f"{dir}/{file}"
        df = pd.read_csv(file_path, sep="\t")
        # print(df.value_counts("relation"))
        # print(file)
        df.rename(columns={"gold":"nli"})
        # print(df.columns)
        df["nli"] = ["neutral" if str(nli).lower() in ["neu", "unknown", "nan"] else str(nli).lower() for nli in df["nli"]]

        #print(df.columns)
        if len(df[df.columns[-1]].unique()) > 3:
            df[df.columns[-1]] = ["entailment" if "entailment" in str(pred)
                     else "contradiction" if "contradiction" in str(pred)
                     else "neutral" for pred in df[df.columns[-1]]]
            # print("\n\n",file)
            # print(df[df.columns[-1]].unique())

        pred_list.append(df[df.columns[-1]])
        df_list.append(df)

all_df = pd.concat(df_list[:1]+pred_list[1:], axis=1)
print(all_df.columns[5:])
models = list(all_df.columns[5:])
# for model in models:
#     print(all_df.value_counts(model))

# 1. Evaluate individual model F1
model_cols = all_df.columns[6:]

# for col in model_cols:
#     print(all_df[col].value_counts())
f1s = {col: f1_score(all_df["nli"], all_df[col], average="macro") for col in model_cols}

for k, v in f1s.items():
    print(k,v)
top_models = sorted(f1s, key=f1s.get, reverse=True)[:10]  # Take top 8

# Calculate weights (e.g., F1 scores for each model)
weights = [f1s[col] for col in top_models]
print(weights)

# 2. Only use top models for combinations
from itertools import combinations
subsets = [list(sub) for k in range(3, 6) for sub in combinations(top_models, k)]

# print(top_models)

def greedy_ensemble(all_df, model_cols, gold_col="nli", max_size=5):
    selected = []
    best_f1 = 0
    for _ in range(max_size):
        candidates = [m for m in model_cols if m not in selected]
        scores = []
        for c in candidates:
            cols = selected + [c]
            maj, gold = weighted_majority_vote(all_df, cols, weights)
            #maj, gold = majority_vote(all_df, cols, threshold=math.ceil(0.5*len(cols)))
            if len(gold) == 0:
                continue
            f1 = f1_score(gold, maj, average="macro")
            scores.append((f1, c))
        if not scores:
            break
        scores.sort(reverse=True)
        if scores[0][0] > best_f1:
            best_f1 = scores[0][0]
            selected.append(scores[0][1])
        else:
            break
    return selected, best_f1

selected_models, best_f1 = greedy_ensemble(all_df, top_models)
print("Best greedy ensemble:", selected_models, "F1:", best_f1)

def findSubsets(v, idx, subset, result):
  
    # If the current subset is not empty insert it into
    # the result
    if (len(subset) != 0):
        if (subset not in result):
            result.append(subset[:])

    # Iterate over every element in the array
    for j in range(idx, len(v)):
      
        # Pick the element and move ahead
        subset.append(v[j])
        findSubsets(v, j + 1, subset, result)

        # Backtrack to drop the element
        subset.pop()

def solve(v):
  
    # To store the resulting subsets.
    result = []
    subset = []

    # Helper function call
    findSubsets(v, 0, subset, result)

    res = []
    for i in range(len(result)):
        res.append(list(result[i]))

    return res

def select_models():
    model_cols = list(all_df.columns[5:])
    max_subset_size = 5  # or 4
    result = []
    for k in range(3, max_subset_size + 1):
        subsets = [list(subset) for subset in itertools.combinations(model_cols, k)]
        # print(k)
        # for subs in subsets:
        #     print(subs)        

    result += subsets
    # result = [list(subset) for k in range(3, max_subset_size+1) for subset in itertools.combinations(model_cols, k)]
    
    max_f1_dict = {}
    max_f1_num = {}

    for subs in result:
        threshold = math.ceil(0.5*len(subs))

        majority_label, gold = majority_vote(all_df, subs, threshold)
        if len(gold) == 0:
            continue
        f1 = f1_score(y_true=gold, y_pred=majority_label, average="macro")
        key = " ".join(subs)+f"_{threshold} of {len(subs)}"
        max_f1_dict[key] = f1
        max_f1_num[key] = len(gold)

    max_models = max(max_f1_dict, key=max_f1_dict.get)
    max_models_num = max_f1_num[max_models]
    print("Best subset and threshold:", max_models)
    print("Number of samples:", max_models_num)
    print("Best F1:", max(max_f1_dict.values()))

def select_models_optimised():
    model_cols = list(all_df.columns[6:])
    min_size = 3
    max_size = 5  # Adjust as needed
    subsets = [list(subset) for k in range(min_size, max_size+1) for subset in itertools.combinations(model_cols, k)]
    
    max_f1_dict = {}
    max_f1_num = {}

    for subs in subsets:
        threshold = math.ceil(0.5*len(subs))
        
        votes = all_df[subs]
        counts = votes.apply(lambda x: x.value_counts(), axis=1).fillna(0)
        max_label = counts.idxmax(axis=1)
        max_agree = counts.max(axis=1)
        mask = max_agree >= threshold
        majority_label = max_label[mask]
        gold = all_df.loc[mask, "nli"]
        if len(gold) == 0:
            continue
        f1 = f1_score(y_true=gold, y_pred=majority_label, average="macro")
        key = " ".join(subs)+f"_{threshold} of {len(subs)}"
        max_f1_dict[key] = f1
        max_f1_num[key] = len(gold)

    max_models = max(max_f1_dict, key=max_f1_dict.get)
    max_models_num = max_f1_num[max_models]
    print(max_models)
    print(max_models_num)
    print(max(max_f1_dict.values()))

# select_models_optimised()

    # for subs in result:
    #     if len(subs) >= 3 and len(subs) <= 5:
    #         thresholds = [0] + list(range(math.ceil(0.5*len(subs)), len(subs)+1))
    #         for threshold in thresholds:

    #             majority_label = []
    #             gold = []

    #             for index, row in all_df.iterrows():
    #                 classifs = [row[col] for col in subs]
    #                 gold_label = row["nli"]
    #                 #print(list(classifs))
    #                 count_dict = {"entailment":classifs.count("entailment"),
    #                             "contradiction":classifs.count("contradiction"),
    #                             "neutral":classifs.count("neutral")}
                    
    #                 #print(count_dict)
    #                 max_label = max(count_dict, key=count_dict.get)
    #                 max_agree = max(count_dict.values())

    #                 if max_agree >= threshold:
    #                     #print(max_label, row["nli"])

    #                     majority_label.append(max_label)
    #                     gold.append(gold_label)

    #             f1 = f1_score(y_true=gold, y_pred=majority_label, average="macro")
                
    #             max_f1_dict[" ".join(subs)+f"_{threshold} of {len(subs)}"] = f1
    #             max_f1_num[" ".join(subs)+f"_{threshold} of {len(subs)}"] = len(gold) 

    # max_models = max(max_f1_dict, key=max_f1_dict.get)
    # max_models_num = max_f1_num[max_models]
    # print(max_models)
    # print(max_models_num)
    # print(max(max_f1_dict.values()))

# select_models()

# all_df["relation"] = [str(relation).lower() for relation in all_df["relation"]]
# all_df["Corpus_ID"] = [id.split("_")[0] for id in all_df["ID"]]
# print(all_df.Corpus_ID.unique())
# for relation in ["support", "attack", "nan"]:
#     print(relation)
#     print(all_df.loc[all_df["relation"]== relation].value_counts("Corpus_ID"))


# models = ["Qwen2.5:7b_16Shot_nli_label", "Llama3.1:8b_4Shot_nli_label", "deberta-v3-large-tasksource-nli_nli_label", "bart-large-mnli_nli_label"]
# majority_label = []
# gold = []
# max_label_dict = {0:[],
#                   2:[],
#                   3:[],
#                   4:[]}
# gold_label_dict = {0:[],
#                   2:[],
#                   3:[],
#                   4:[]}


# all_df["relation"] = [str(relation).lower() for relation in all_df["relation"]]
# for index, row in all_df.loc[all_df["relation"] == "attack"].iterrows():
#     classifs = [row[col] for col in models]
#     # print(classifs)
#     gold_label = row["nli"]
#     count_dict = {"entailment":classifs.count("entailment"),
#                 "contradiction":classifs.count("contradiction"),
#                 "neutral":classifs.count("neutral")}
    
#     #print(count_dict)
#     max_label = max(count_dict, key=count_dict.get)
#     max_agree = max(count_dict.values())
#     # print(max_agree)
#     for threshold in [0, 3]:
#         if max_agree >= threshold:
#             # print(max_label, row["nli"])

#             max_label_dict[threshold].append(max_label)
#             gold_label_dict[threshold].append(gold_label)

# for threshold in [0, 3]:
#     f1 = f1_score(y_true=gold_label_dict[threshold], y_pred=max_label_dict[threshold], average="macro")
#     acc = accuracy_score(y_true=gold_label_dict[threshold], y_pred=max_label_dict[threshold])
#     cm = confusion_matrix(y_true=gold_label_dict[threshold], y_pred=max_label_dict[threshold])
#     print(len(max_label_dict[threshold]), acc, f1)
#     print(cm)

# TODO Next step: get majority vote to see if the classifier is better then; the gold standard is in the nli column
# import itertools

# def select_models_optimised():
#     model_cols = list(all_df.columns[6:])
#     min_size = 3
#     max_size = 5  # Adjust as needed
#     subsets = [list(subset) for k in range(min_size, max_size+1) for subset in itertools.combinations(model_cols, k)]
    
#     max_f1_dict = {}
#     max_f1_num = {}

#     for subs in subsets:
#         thresholds = [0] + list(range(math.ceil(0.5*len(subs)), len(subs)+1))
#         for threshold in thresholds:
#             votes = all_df[subs]
#             counts = votes.apply(lambda x: x.value_counts(), axis=1).fillna(0)
#             max_label = counts.idxmax(axis=1)
#             max_agree = counts.max(axis=1)
#             mask = max_agree >= threshold
#             majority_label = max_label[mask]
#             gold = all_df.loc[mask, "nli"]
#             if len(gold) == 0:
#                 continue
#             f1 = f1_score(y_true=gold, y_pred=majority_label, average="macro")
#             key = " ".join(subs)+f"_{threshold} of {len(subs)}"
#             max_f1_dict[key] = f1
#             max_f1_num[key] = len(gold)

#     max_models = max(max_f1_dict, key=max_f1_dict.get)
#     max_models_num = max_f1_num[max_models]
#     print(max_models)
#     print(max_models_num)
#     print(max(max_f1_dict.values()))

# select_models_optimised()
