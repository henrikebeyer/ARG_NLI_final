import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import os
import math
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# PREDICTION_DIR = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/noUnrelated"
PROB_PREDICTION_DIR = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/noUnrelated_probabilities"
# PROB_PREDICTION_DIR = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/noUnrelated_decoderOnly/precision_tuned"

TOP = 8
SUBSET_SIZE = 10

def read_input_files(dir):
    files = [f for f in os.listdir(dir) if f.endswith('.tsv')]
    dataframes = []
    predictions = []
    for file in files:
        df = pd.read_csv(os.path.join(dir, file), sep="\t")
        df.rename(columns={"gold":"nli"})
        df[df.columns[-1]] = df[df.columns[-1]].astype(str)
        df["nli"] = ["neutral" if str(nli).lower() in ["neu", "unknown", "nan"] else str(nli).lower() for nli in df["nli"]]

        if len(df[df.columns[-1]].unique()) > 3:
            df[df.columns[-1]] = ["entailment" if "entailment" in str(pred)
                     else "contradiction" if "contradiction" in str(pred)
                     else "neutral" for pred in df[df.columns[-1]]]
        dataframes.append(df)
        predictions.append(df[df.columns[-1]])

    all_df = pd.concat(dataframes[:1]+predictions[1:], axis=1)
    return all_df

def read_prob_input_files(dir):
    files = [f for f in os.listdir(dir) if f.endswith('.tsv')]
    dataframes = []
    predictions = []
    probs = []
    for file in files:
        df = pd.read_csv(os.path.join(dir, file), sep="\t")
        df.rename(columns={"gold":"nli"})
        # df[df.columns[-4]] = df[df.columns[-4]].astype(str)
        df["nli"] = ["neutral" if str(nli).lower() in ["neu", "unknown", "nan"] else str(nli).lower() for nli in df["nli"]]
        dataframes.append(df[df.columns[:-3]]) 
        predictions.append(df[df.columns[-4]])
        probs.append(df[df.columns[-3:]])
    all_df = pd.concat(dataframes[:1]+predictions[1:], axis=1).astype(str)
    prob_df = pd.concat(probs, axis=1)
    return all_df, prob_df

def majority_vote(df, cols, threshold):
    votes = df[cols]
    counts = votes.apply(lambda x: x.value_counts(), axis=1).fillna(0)
    max_label = counts.idxmax(axis=1)
    max_agree = counts.max(axis=1)
    mask = max_agree >= threshold
    return max_label[mask], df.loc[mask, "nli"]

def weighted_majority_vote(df, cols, weights):
    votes = df[cols]
    norm_weights = [w / sum(weights) for w in weights]
    weighted_counts = pd.DataFrame(0.0, index=votes.index, columns=['entailment', 'contradiction', 'neutral'])
    for col, weight in zip(cols, norm_weights):
        for label in ["entailment", "contradiction", "neutral"]:
            weighted_counts[label] += (votes[col] == label) #* weight
    print(weighted_counts)
    max_label = weighted_counts.idxmax(axis=1)
    return max_label, df["nli"]


def get_best_models_and_weights(df, top=8):
    model_cols = df.columns[6:]
    # for col in model_cols:
        # print(f"Model: {col}")
        # print(df[col].unique())
    f1s = {col: f1_score(df["nli"], df[col], average="macro") for col in model_cols}
    
    best_models = sorted(f1s, key=f1s.get, reverse=True)[:top]
    weights = [f1s[col] for col in best_models]

    return {best_models[i]: weights[i] for i in range(len(best_models))}

def greedy_ensemble(all_df, top_dict, gold_col="nli", max_size=SUBSET_SIZE):
    selected = []
    best_f1 = 0
    for _ in range(max_size):
        candidates = [m for m in top_dict.keys() if m not in selected]
        scores = []
        for c in candidates:
            cols = selected + [c]
            current_weights = [top_dict[col] for col in cols]
            # maj, gold = weighted_majority_vote(all_df, cols, current_weights)
            maj, gold = majority_vote(all_df, cols, threshold=math.ceil(0.5*len(cols)))
            # if len(gold) == 0:
            #     continue
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

def stacking_ensemble(df, model_cols, gold_col="nli"):
    # Encode string labels to integers for sklearn
    le = LabelEncoder()
    y = le.fit_transform(df[gold_col])
    X = df[list(model_cols)].apply(le.transform)
    # Train meta-classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    f1 = f1_score(y, y_pred, average="macro")
    print("Stacking F1:", f1)
    # Optionally, return decoded predictions
    return le.inverse_transform(y_pred), le.inverse_transform(y)

def exhaustive_ensemble_search(df, top_models, min_size=2, max_size=10):
    from itertools import combinations
    best_f1 = 0
    best_subset = None
    for k in range(min_size, max_size+1):
        for subset in combinations(top_models.keys(), k):
            maj, gold = majority_vote(df, list(subset), threshold=math.ceil(0.5*len(subset)))
            if len(gold) == 0:
                continue
            f1 = f1_score(gold, maj, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_subset = subset
    return best_subset, best_f1


def mixture_of_experts(df, model_cols, gold_col="nli"):
    from sklearn.metrics import precision_score
    label_experts = {}
    labels = ["entailment", "contradiction", "neutral"]
    for label in labels:
        best_score = 0
        best_model = None
        for col in model_cols:
            # Precision for this label
            score = precision_score(df[gold_col], df[col], labels=[label], average="macro", zero_division=0)
            if score > best_score:
                best_score = score
                best_model = col
        label_experts[label] = best_model
    print("Experts per label:", label_experts)
    # For each row, use the expert's prediction for its predicted label
    preds = []
    for idx, row in df.iterrows():
        # Get each expert's prediction for this row
        expert_preds = {label: row[label_experts[label]] for label in labels}
        # Choose the label whose expert is most confident (here: just use the expert's prediction)
        # You could also use probabilities if available
        # For simplicity, pick the label whose expert predicts that label for this row
        for label in labels:
            if expert_preds[label] == label:
                preds.append(label)
                break
        else:
            # If no expert predicts their own label, fallback to majority vote
            votes = [row[col] for col in model_cols]
            preds.append(pd.Series(votes).mode()[0])
    f1 = f1_score(df[gold_col], preds, average="macro")
    print("Mixture of Experts F1:", f1)
    return preds

def mixture_of_experts_classwise_accuracy(df, model_cols, gold_col="nli"):
    labels = ["entailment", "contradiction", "neutral"]
    label_experts = {}
    for label in labels:
        best_acc = 0
        best_model = None
        # Only consider rows where the gold label is the current label
        mask = df[gold_col] == label
        for col in model_cols:
            if mask.sum() == 0:
                continue
            acc = (df.loc[mask, col] == label).mean()
            if acc > best_acc:
                best_acc = acc
                best_model = col
        label_experts[label] = best_model
    print("Experts per label (by classwise accuracy):", label_experts)
    # For each row, use the expert's prediction for its gold label
    preds = []
    for idx, row in df.iterrows():
        # Use the expert for the predicted label of each model
        expert_preds = {label: row[label_experts[label]] for label in labels}
        # If any expert predicts their own label, use that label
        for label in labels:
            if expert_preds[label] == label:
                preds.append(label)
                break
        else:
            # Fallback: majority vote
            votes = [row[col] for col in model_cols]
            preds.append(pd.Series(votes).mode()[0])
    f1 = f1_score(df[gold_col], preds, average="macro")
    print("Mixture of Experts (classwise accuracy) F1:", f1)
    return preds

def mixture_of_experts_classwise_f1(df, model_cols, gold_col="nli"):
    from sklearn.metrics import f1_score
    labels = ["entailment", "contradiction", "neutral"]
    label_experts = {}
    for label in labels:
        best_f1 = 0
        best_model = None
        mask = df[gold_col] == label
        for col in model_cols:
            if mask.sum() == 0:
                continue
            # F1 for this label as positive class
            f1 = f1_score(df.loc[mask, gold_col], df.loc[mask, col], labels=[label], average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_model = col
        label_experts[label] = best_model
    print("Experts per label (by classwise F1):", label_experts)
    preds = []
    for idx, row in df.iterrows():
        expert_preds = {label: row[label_experts[label]] for label in labels}
        for label in labels:
            if expert_preds[label] == label:
                preds.append(label)
                break
        else:
            votes = [row[col] for col in model_cols]
            preds.append(pd.Series(votes).mode()[0])
    f1 = f1_score(df[gold_col], preds, average="macro")
    print("Mixture of Experts (classwise F1) F1:", f1)
    return preds


def mixture_of_experts_probabilities(df, prob_df, gold_col="nli"):
    """
    model_prob_cols: dict mapping model name to dict of label->column name, e.g.
        {
            "model1": {"entailment": "model1_entailment_prob", ...},
            ...
        }
    """
    labels = ["entailment", "contradiction", "neutral"]
    label_experts = {}
    for label in labels:
        best_score = 0
        best_model = None
        mask = df[gold_col] == label
        for col in model_cols:
            if mask.sum() == 0:
                continue
            # Accuracy for this label as gold
            score = precision_score(df[gold_col], df[col], labels=[label], average="macro", zero_division=0)
            # print(col, "accuracy for label", label, ":", acc)
            if score > best_score:
                best_acc = score
                best_model = col
        label_experts[label] = best_model
    print("Experts per label (by mean prob):", label_experts)
    preds = []
    for idx, row in df.iterrows():
        expert_preds = {label: row[label_experts[label]] for label in labels}
        expert_probs = {label: prob_df.loc[idx, f"{label_experts[label]}_prob_{expert_preds[label]}"] for label in labels}
        candidates = []
        for label, pred in expert_preds.items():
            if pred == label:
                candidates.append(pred)
        # print(candidates)
        if len(candidates) == 0:
            # Fallback: majority vote among all model predictions for this row
            votes = [row[col] for col in model_cols]
            preds.append(pd.Series(votes).mode()[0])
        elif len(candidates) == 1:
            preds.append(candidates[0])
        else:
            # If multiple candidates, choose the one with highest probability
            highest_prob_label = max(candidates, key=lambda l: expert_probs[l])
            preds.append(highest_prob_label)

    print(len(preds), len(df["nli"]))


    f1 = f1_score(df[gold_col], preds, average="macro")
    print("Mixture of Experts (probabilities) F1:", f1)
    return preds

import pandas as pd
from sklearn.metrics import f1_score, precision_score

import pandas as pd
from sklearn.metrics import f1_score

def thresholded_predictions(df, prob_df, model_cols, labels=None, gold_col="nli", threshold=0.5):
    """
    Chooses the label with the highest probability among all models,
    if it is above the threshold. Gold labels remain unchanged.

    Args:
        df: DataFrame with model predictions
        prob_df: DataFrame with model probability columns
        model_cols: list of model prediction columns
        labels: list of possible labels (default = unique values in gold_col)
        gold_col: name of gold label column
        threshold: probability threshold (0-1)

    Returns:
        preds: list of predictions (or None if no label passed threshold)
        coverage: fraction of rows with a valid prediction
        f1: macro-F1 score on subset with valid predictions
    """
    if labels is None:
        labels = df[gold_col].unique().tolist()

    preds = []

    for idx, row in df.iterrows():
        candidate_probs = {}

        # collect all probabilities across models
        for model in model_cols:
            pred = row[model]  # predicted label from that model
            prob_col = f"{model}_prob_{pred}"
            if prob_col in prob_df.columns:
                prob = prob_df.loc[idx, prob_col]
                # take max prob for each label across models
                candidate_probs[pred] = max(candidate_probs.get(pred, 0), prob)

        # filter by threshold
        valid_candidates = {l: p for l, p in candidate_probs.items() if p >= threshold}
        if not valid_candidates:
            preds.append(None)   # abstain
        else:
            best_label = max(valid_candidates, key=valid_candidates.get)
            preds.append(best_label)

    # coverage = % rows with a prediction
    coverage = sum(p is not None for p in preds) / len(preds)

    # evaluation only on rows with valid predictions
    eval_mask = pd.Series(preds).notna()
    preds_eval = [p for p, keep in zip(preds, eval_mask) if keep]
    gold_eval = df.loc[eval_mask, gold_col]

    if len(preds_eval) > 0:
        f1 = f1_score(gold_eval, preds_eval, average="macro")
        print(f"Thresholded models F1: {f1:.4f}, Coverage: {coverage:.2%}")
    else:
        f1 = None
        print("No predictions above threshold – cannot compute F1.")

    return preds, coverage, f1


def thresholded_majority_vote(df, prob_df, model_cols, labels=None, gold_col="nli", threshold=0.5):
    """
    Majority vote among model predictions that pass the threshold.

    Args:
        df: DataFrame with model predictions
        prob_df: DataFrame with model probability columns
        model_cols: list of model prediction columns
        labels: list of possible labels (default = unique values in gold_col)
        gold_col: name of gold label column
        threshold: probability threshold (0-1)

    Returns:
        preds: list of predictions (or None if no label passed threshold)
        coverage: fraction of rows with a valid prediction
        f1: macro-F1 score on subset with valid predictions
    """
    if labels is None:
        labels = df[gold_col].unique().tolist()

    preds = []

    for idx, row in df.iterrows():
        candidate_probs = {}
        candidate_labels = []

        # collect predictions across models
        for model in model_cols:
            pred = row[model]
            prob_col = f"{model}_prob_{pred}"
            if prob_col in prob_df.columns:
                prob = prob_df.loc[idx, prob_col]
                if prob >= threshold:
                    candidate_labels.append(pred)
                    candidate_probs[pred] = max(candidate_probs.get(pred, 0), prob)

        if not candidate_labels:
            preds.append(None)  # abstain
        else:
            # majority vote
            vote_counts = Counter(candidate_labels)
            top_labels = vote_counts.most_common()
            if len(top_labels) == 1 or top_labels[0][1] > top_labels[1][1]:
                # clear majority
                preds.append(top_labels[0][0])
            else:
                # tie → break by highest probability
                tied_labels = [lbl for lbl, cnt in top_labels if cnt == top_labels[0][1]]
                best_label = max(tied_labels, key=lambda l: candidate_probs.get(l, 0))
                preds.append(best_label)

    # coverage = % rows with a prediction
    coverage = sum(p is not None for p in preds) / len(preds)

    # evaluation only on rows with valid predictions
    eval_mask = pd.Series(preds).notna()
    preds_eval = [p for p, keep in zip(preds, eval_mask) if keep]
    gold_eval = df.loc[eval_mask, gold_col]

    if len(preds_eval) > 0:
        f1 = f1_score(gold_eval, preds_eval, average="macro")
        print(f"Thresholded majority-vote F1: {f1:.4f}, Coverage: {coverage:.2%}")
    else:
        f1 = None
        print("No predictions above threshold – cannot compute F1.")

    return preds, coverage, f1

def dynamic_confidence_expert(df, prob_df, model_names, labels, gold_col = "nli"):
    # model_names = df.columns[6:]
    labels = ["entailment", "contradiction", "neutral"]
    label_experts = {}
    for label in labels:
        best_acc = 0
        best_model = None
        mask = df[gold_col] == label
        for col in model_cols:
            if mask.sum() == 0:
                continue
            # Accuracy for this label as gold
            acc = (df.loc[mask, col] == label).mean()
            # print(col, "accuracy for label", label, ":", acc)
            if acc > best_acc:
                best_acc = acc
                best_model = col
        label_experts[label] = best_model
    preds = []
    for idx, row in df.iterrows():
        best_model = None
        best_label = None
        best_prob = -1
        for model in model_names:
            for label in labels:
                prob = prob_df.loc[idx, f"{model}_prob_{label}"]
                if prob > best_prob:
                    best_prob = prob
                    best_model = model
                    best_label = label
        preds.append(best_label)

    print("Dynamic confidence expert F1:", f1_score(df["nli"], preds, average="macro"))
    return preds


def dynamic_per_class_expert(df, prob_df, model_names, gold_col="nli"):
    labels = ["entailment", "contradiction", "neutral"]
    label_experts = {}
    for label in labels:
        best_acc = 0
        best_model = None
        mask = df[gold_col] == label
        for col in model_cols:
            if mask.sum() == 0:
                continue
            # Accuracy for this label as gold
            acc = (df.loc[mask, col] == label).mean()
            # print(col, "accuracy for label", label, ":", acc)
            if acc > best_acc:
                best_acc = acc
                best_model = col
        label_experts[label] = best_model
    preds = []
    for idx, row in df.iterrows():
        # For each label, get the expert's probability for that label
        expert_probs = {label: prob_df.loc[idx, f"{label_experts[label]}_prob_{label}"] for label in labels}
        # Pick the label with the highest expert probability
        pred_label = max(expert_probs, key=expert_probs.get)
        preds.append(pred_label)

    print("Dynamic per-class expert F1:", f1_score(df[gold_col], preds, average="macro"))
    return preds


# prediction_dataframe = read_input_files(PREDICTION_DIR)
prediction_dataframe, probability_dataframe = read_prob_input_files(PROB_PREDICTION_DIR)
print(prediction_dataframe.columns)
top_models = get_best_models_and_weights(prediction_dataframe, top=len(prediction_dataframe.columns[6:]))

selected_models, best_f1 = selected_models, best_f1 = greedy_ensemble(prediction_dataframe, top_models, max_size=SUBSET_SIZE)
print("Best greedy ensemble:", selected_models, "F1:", best_f1)

model_cols = list(top_models.keys())
stacking_preds, stacking_gold = stacking_ensemble(prediction_dataframe, model_cols)

stacking_f1 = f1_score(stacking_gold, stacking_preds, average="macro")
print("Stacking F1:", stacking_f1)

best_subset, best_f1 = exhaustive_ensemble_search(prediction_dataframe, top_models, min_size=2, max_size=10)
print("Best exhaustive ensemble:", best_subset, "F1:", best_f1)

# model_cols = list(top_models.keys())
moe_preds_precision = mixture_of_experts(prediction_dataframe, model_cols)
# # subsets = list(list(subset) for k in range (3, SUBSET_SIZE+1) for subset in itertools.combinations(top_models.keys(), k))

# model_cols = list(top_models.keys())
moe_preds_acc = mixture_of_experts_classwise_accuracy(prediction_dataframe, model_cols)

moe_preds_f1 = mixture_of_experts_classwise_f1(prediction_dataframe, model_cols)

moe_preds_prob = mixture_of_experts_probabilities(prediction_dataframe, probability_dataframe)

dynamic_conf_preds = dynamic_confidence_expert(prediction_dataframe, probability_dataframe, model_cols, labels=["entailment", "contradiction", "neutral"])
dynamic_per_class_expert_preds = dynamic_per_class_expert(prediction_dataframe, probability_dataframe, model_cols)

def confusion_matrix_accuracies_and_counts(df, model_cols, gold_col="nli"):
    labels = ["entailment", "contradiction", "neutral"]
    acc_matrix = pd.DataFrame(index=labels, columns=model_cols)
    count_matrix = pd.DataFrame(index=labels, columns=model_cols)
    for label in labels:
        mask = df[gold_col] == label
        for col in model_cols:
            total = mask.sum()
            if total == 0:
                acc = float('nan')
                count_str = "0/0"
            else:
                correct = (df.loc[mask, col] == label).sum()
                acc = correct / total
                count_str = f"{correct}/{total}"
            acc_matrix.loc[label, col] = acc
            count_matrix.loc[label, col] = count_str
    print("Confusion matrix of per-class accuracies:")
    print(acc_matrix)
    print("\nConfusion matrix of raw counts (correct/total):")
    print(count_matrix)
    return acc_matrix, count_matrix

threshold_preds, updated_df, f1 = thresholded_predictions(prediction_dataframe, probability_dataframe, model_cols, threshold=0.9975)

threshold_preds, updated_df, f1 = thresholded_predictions(prediction_dataframe, probability_dataframe, model_cols, threshold=0.9968)

threshold_preds, updated_df, f1 = thresholded_predictions(prediction_dataframe, probability_dataframe, model_cols, threshold=0.8)

threshold_preds, updated_df, f1 = thresholded_predictions(prediction_dataframe, probability_dataframe, model_cols, threshold=0)


threshold_preds, updated_df, f1 = thresholded_majority_vote(prediction_dataframe, probability_dataframe, model_cols, threshold=0.9975)

threshold_preds, updated_df, f1 = thresholded_majority_vote(prediction_dataframe, probability_dataframe, model_cols, threshold=0.9968)

threshold_preds, updated_df, f1 = thresholded_majority_vote(prediction_dataframe, probability_dataframe, model_cols, threshold=0)

# threshold_preds, updated_df = thresholded_predictions(prediction_dataframe, probability_dataframe, model_cols, threshold=0.75)

# threshold_preds, updated_df = thresholded_predictions(prediction_dataframe, probability_dataframe, model_cols, threshold=0.7)

# threshold_preds, updated_df = thresholded_predictions(prediction_dataframe, probability_dataframe, model_cols, threshold=0.65)

# threshold_preds, updated_df = thresholded_predictions(prediction_dataframe, probability_dataframe, model_cols, threshold=0.6)

# threshold_preds, updated_df = thresholded_predictions(prediction_dataframe, probability_dataframe, model_cols, threshold=0.55)

# threshold_preds, updated_df = thresholded_predictions(prediction_dataframe, probability_dataframe, model_cols, threshold=0.5)

# threshold_preds, updated_df = thresholded_predictions(prediction_dataframe, probability_dataframe, model_cols, threshold=0.45)

# threshold_preds, updated_df = thresholded_predictions(prediction_dataframe, probability_dataframe, model_cols, threshold=0)
                                                      
# Usage example:
# confusion_matrix_accuracies_and_counts(prediction_dataframe, model_cols)

# dynamic_confidence_preds = dynamic_confidence_expert(prediction_dataframe, probability_dataframe, model_cols, ["entailment", "contradiction", "neutral"])
# dynamic_per_class_preds = dynamic_per_class_expert(prediction_dataframe, probability_dataframe, model_cols)

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import numpy as np

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# # Encode predictions for each model (label-encoded)
# le = LabelEncoder()
# X_preds = prediction_dataframe[model_cols].apply(le.fit_transform)

# # Optionally, one-hot encode predictions (better for linear models)
# ohe = OneHotEncoder(sparse_output=False)
# X_preds_onehot = ohe.fit_transform(X_preds)

# # Use probabilities as features
# X_probs = probability_dataframe[[col for col in probability_dataframe.columns if any(m in col for m in model_cols)]]
# # print(X_probs)
# # Concatenate predictions (one-hot) and probabilities
# import numpy as np
# meta_features = X_probs.values #np.concatenate([X_preds_onehot, X_probs.values], axis=1)
# y = prediction_dataframe["nli"]
# y_encoded = le.fit_transform(y)

# # Split data (50/25/25)
# from sklearn.model_selection import train_test_split
# X_train, X_temp, y_train, y_temp = train_test_split(meta_features, y_encoded, test_size=0.5, random_state=42)
# X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# # Train meta-classifier
# from sklearn.linear_model import LogisticRegression
# meta_clf = LogisticRegression(max_iter=1000)
# meta_clf.fit(X_train, y_train)

# # Evaluate on validation and test sets
# valid_preds = meta_clf.predict(X_valid)
# test_preds = meta_clf.predict(X_test)

# # # 6. Evaluate
# from sklearn.metrics import accuracy_score, f1_score
# print("Meta-classifier validation accuracy:", accuracy_score(y_valid, valid_preds))
# print("Meta-classifier validation F1:", f1_score(y_valid, valid_preds, average="macro"))
# print("Meta-classifier test accuracy:", accuracy_score(y_test, test_preds))
# print("Meta-classifier test F1:", f1_score(y_test, test_preds, average="macro"))


# from sklearn.model_selection import StratifiedKFold, KFold
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import numpy as np

# le = LabelEncoder()
# y = prediction_dataframe["nli"]
# y_encoded = le.fit_transform(y)

# # Prepare features: one-hot predictions + probabilities
# X_preds = prediction_dataframe[model_cols].apply(le.fit_transform)
# ohe = OneHotEncoder(sparse_output=False)
# X_preds_onehot = ohe.fit_transform(X_preds)
# X_probs = probability_dataframe[[col for col in probability_dataframe.columns if any(m in col for m in model_cols)]]
# # meta_features = np.concatenate([X_preds_onehot, X_probs.values], axis=1)
# meta_features = X_probs.values

# from sklearn.model_selection import cross_val_score, KFold
# from sklearn.linear_model import LogisticRegression

# # Use meta_features and y_encoded as defined above
# cv = KFold(n_splits=5, shuffle=True, random_state=42)  # No stratification

# clf = LogisticRegression(solver="newton-cg", max_iter=1000)
# acc_scores = cross_val_score(clf, meta_features, y_encoded, cv=cv, scoring='accuracy')
# f1_scores = cross_val_score(clf, meta_features, y_encoded, cv=cv, scoring='f1_macro')

# print(f"Logistic Regression CV accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
# print(f"Logistic Regression CV macro F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")


# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier

# # Try different meta-classifiers
# meta_clfs = [
#     RandomForestClassifier(n_estimators=100, random_state=42),
#     GradientBoostingClassifier(n_estimators=100, random_state=42),
#     SVC(probability=True, kernel='rbf', random_state=42)
# ]

# for clf in meta_clfs:
#     acc = cross_val_score(clf, meta_features, y_encoded, cv=cv, scoring='accuracy').mean()
#     f1 = cross_val_score(clf, meta_features, y_encoded, cv=cv, scoring='f1_macro').mean()
#     print(f"{clf.__class__.__name__}: CV accuracy={acc:.4f}, CV macro F1={f1:.4f}")


# from sklearn.ensemble import StackingClassifier, RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

# estimators = [
#     ('rf', RandomForestClassifier()),
#     ('svc', SVC(probability=True)),
#     ('gbc', GradientBoostingClassifier()),
#     #('lr', LogisticRegression())
# ]
# stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# stack.fit(X_train, y_train)  # Fit first!
# y_pred = stack.predict(X_test)

# acc = cross_val_score(stack, meta_features, y_encoded, cv=cv, scoring='accuracy').mean()
# f1 = cross_val_score(stack, meta_features, y_encoded, cv=cv, scoring='f1_macro').mean()
# print(f"StackingClassifier: CV accuracy={acc:.4f}, CV macro F1={f1:.4f}")
# ...existing code...