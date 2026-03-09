import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score


CORPUS_FILE = "/home/oenni/Dokumente/NLI-Argumentation-project/corpus/full_corpus_SupportAttack.tsv"
PREDICTION_DIR = "/home/oenni/Dokumente/NLI-Argumentation-project/NLI_labeling/full_NLI_preds"
EVAL_DIR = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_finetuned"
MODELS = ["sileod_deberta-v3-large-tasksource-nli", "FacebookAI_roberta-large-mnli", "facebook_bart-large-mnli", "cross-encoder_nli-deberta-v3-large"]
ANNOT_OUT = "/home/oenni/Dokumente/NLI-Argumentation-project/corpus"

def read_in_test_file(file):
    df = pd.read_csv(file, sep="\t")
    return df

def read_in_prediction_files(pred_dir, model_names, eval=True):
    pred_dfs = []
    prob_dfs = []
    for model in model_names:
        if eval:
            pred_file = os.path.join(pred_dir, f"results_{model}.tsv")
        else:
            pred_file = os.path.join(pred_dir, f"full_NLI_preds_{model}.tsv")
        df = pd.read_csv(pred_file, sep="\t")
        if model == model_names[0]:  # only read in gold once
            gold = df['nli']
            pred_dfs.append(gold)  # assuming gold is same across files
        if eval:
            preds = df[f"{model}_ft"]
            prob_cols = df.columns[df.columns.str.startswith(f"{model}_ft_prob_")]
        else:
            preds = df[f"{model}_pred"]
            prob_cols = df.columns[df.columns.str.startswith(f"{model}_prob_")]
        probs = df[prob_cols]
        pred_dfs.append(preds)
        prob_dfs.append(probs)
    pred_df = pd.concat(pred_dfs, axis=1)
    prob_df = pd.concat(prob_dfs, axis=1)
    return pred_df, prob_df


pred_df, prob_df = read_in_prediction_files(EVAL_DIR, MODELS)
full_pred_df, full_prob_df = read_in_prediction_files(PREDICTION_DIR, MODELS, eval=False)

def thresholded_mixture_of_experts_hard(df, prob_df, model_cols, metric=f1_score, gold_col="nli", threshold=0.5, eval=True):
    """
    model_prob_cols: dict mapping model name to dict of label->column name, e.g.
        {
            "model1": {"entailment": "model1_entailment_prob", ...},
            ...
        }
    """
    labels = ["entailment", "contradiction", "neutral"]
    if eval:
        label_experts = {}
        for label in labels:
            #print(label)
            best_score = 0
            best_model = None
            mask = df[gold_col] == label
            for col in model_cols:
                col = col + "_ft"
                if mask.sum() == 0:
                    continue
                # Accuracy for this label as gold
                score = metric(df[gold_col], df[col], labels=[label], average="macro", zero_division=0)
                # print(col, "accuracy for label", label, ":", acc)
                #print(col, best_score, score)
                if score > best_score:
                    best_score = score
                    best_model = col
                    #print(best_model)
            label_experts[label] = best_model
        print("Experts per label (by mean prob):", label_experts)
    else:
        label_experts = {'entailment': 'sileod_deberta-v3-large-tasksource-nli', 
                         'contradiction': 'FacebookAI_roberta-large-mnli',
                         'neutral': 'sileod_deberta-v3-large-tasksource-nli'}
    preds = []
    for idx, row in df.iterrows():
        if eval:
            expert_preds = {label: row[f"{label_experts[label]}"] for label in labels}
        else:
            expert_preds = {label: row[f"{label_experts[label]}_pred"] for label in labels}
        expert_probs = {label: prob_df.loc[idx, f"{label_experts[label]}_prob_{expert_preds[label]}"] for label in labels}
        candidates = []
        for label, pred in expert_preds.items():
            if pred == label and prob_df.loc[idx, f"{label_experts[label]}_prob_{label}"] > threshold:
                candidates.append(pred)
        # print(candidates)
        if len(candidates) == 0:
            # Fallback: majority vote among all model predictions for this row
            # votes = [row[col+"_ft"] for col in model_cols]
            # preds.append(pd.Series(votes).mode()[0])
            preds.append(None)
        elif len(candidates) == 1:
            preds.append(candidates[0])
        else:
            # If multiple candidates, choose the one with highest probability
            highest_prob_label = max(candidates, key=lambda l: expert_probs[l])
            preds.append(highest_prob_label)

    coverage = sum(p is not None for p in preds) / len(preds)

    if eval:
        # evaluation only on rows with valid predictions
        eval_mask = pd.Series(preds).notna()
        preds_eval = [p for p, keep in zip(preds, eval_mask) if keep]
        gold_eval = df.loc[eval_mask, gold_col]

        if len(preds_eval) > 0:
            f1 = f1_score(gold_eval, preds_eval, average="macro")
            print(f"Thresholded models {metric} for threshold {threshold}: {f1:.4f}, Coverage: {coverage:.2%}")
        else:
            f1 = None
            print("No predictions above threshold – cannot compute F1.")

        return preds, coverage, f1
    else:
        corpus_df = read_in_test_file(CORPUS_FILE)
        corpus_df["nli_preds"] = preds
        corpus_df.dropna(subset=["nli_preds"], inplace=True)

        corpus_df.to_csv(f"{ANNOT_OUT}/nli_preds_threshold_hard_{threshold}.tsv", sep="\t", index=False)

        print(f"Coverage: {coverage:.2%} for threshold {threshold}")
        print(f"Saved annotations to: {ANNOT_OUT}_nli_preds_threshold_hard_{threshold}.tsv")

def thresholded_mixture_of_experts_soft(df, prob_df, model_cols, metric=f1_score, gold_col="nli", threshold=0.5, eval=True):
    """
    model_prob_cols: dict mapping model name to dict of label->column name, e.g.
        {
            "model1": {"entailment": "model1_entailment_prob", ...},
            ...
        }
    """
    labels = ["entailment", "contradiction", "neutral"]
    if eval:
        label_experts = {}
        for label in labels:
            #print(label)
            best_score = 0
            best_model = None
            mask = df[gold_col] == label
            for col in model_cols:
                col = col + "_ft"
                if mask.sum() == 0:
                    continue
                # Accuracy for this label as gold
                score = metric(df[gold_col], df[col], labels=[label], average="macro", zero_division=0)
                # print(col, "accuracy for label", label, ":", acc)
                #print(col, best_score, score)
                if score > best_score:
                    best_score = score
                    best_model = col
                    #print(best_model)
            label_experts[label] = best_model
        print("Experts per label (by mean prob):", label_experts)
    else:
        label_experts = {'entailment': 'sileod_deberta-v3-large-tasksource-nli', 
                         'contradiction': 'FacebookAI_roberta-large-mnli',
                         'neutral': 'sileod_deberta-v3-large-tasksource-nli'}
    preds = []
    for idx, row in df.iterrows():
        if eval:
            expert_preds = {label: row[f"{label_experts[label]}"] for label in labels}
        else:
            expert_preds = {label: row[f"{label_experts[label]}_pred"] for label in labels}
        expert_probs = {label: prob_df.loc[idx, f"{label_experts[label]}_prob_{expert_preds[label]}"] for label in labels}
        candidates = []
        for label, pred in expert_preds.items():
            if pred == label and prob_df.loc[idx, f"{label_experts[label]}_prob_{label}"] > threshold:
                candidates.append(pred)
        # print(candidates)
        if len(candidates) == 0:
            #Fallback: majority vote among all model predictions for this row
            if eval:
                votes = [row[col+"_ft"] for col in model_cols]
            else:
                votes = [row[col+"_pred"] for col in model_cols]
            preds.append(pd.Series(votes).mode()[0])
            # preds.append(None)
        elif len(candidates) == 1:
            preds.append(candidates[0])
        else:
            # If multiple candidates, choose the one with highest probability
            highest_prob_label = max(candidates, key=lambda l: expert_probs[l])
            preds.append(highest_prob_label)

    coverage = sum(p is not None for p in preds) / len(preds)

    if eval:
        # evaluation only on rows with valid predictions
        eval_mask = pd.Series(preds).notna()
        preds_eval = [p for p, keep in zip(preds, eval_mask) if keep]
        gold_eval = df.loc[eval_mask, gold_col]

        if len(preds_eval) > 0:
            f1 = f1_score(gold_eval, preds_eval, average="macro")
            print(f"Thresholded models F1 for threshold {threshold}: {f1:.4f}, Coverage: {coverage:.2%}")
        else:
            f1 = None
            print("No predictions above threshold – cannot compute F1.")

        return preds, coverage, f1
    else:
        corpus_df = read_in_test_file(CORPUS_FILE)
        corpus_df["nli_preds"] = preds
        corpus_df.dropna(subset=["nli_preds"], inplace=True)

        corpus_df.to_csv(f"{ANNOT_OUT}_nli_preds_threshold_soft_{threshold}.tsv", sep="\t", index=False)

        print(f"Coverage: {coverage:.2%} for threshold {threshold}")
        print(f"Saved annotations to: {ANNOT_OUT}/nli_preds_threshold_hard_{threshold}.tsv")

def find_best_threshold(df, prob_df, model_cols, 
                        metric=f1_score, gold_col="nli", 
                        start = 0.99, stop=1, step=0.0001):
    """
    Searches for the threshold that maximizes F1 score.
    """
    thresholds = np.arange(start, stop, step)
    best_threshold = None
    best_f1 = -1
    best_preds = None
    best_coverage = None

    for t in thresholds:
        preds, coverage, f1 = thresholded_mixture_of_experts_hard(
            df, prob_df, model_cols, metric=metric, gold_col=gold_col, threshold=t
        )
        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_preds = preds
            best_coverage = coverage

    print(f"\nBest threshold: {best_threshold:.3f}, F1: {best_f1:.4f}, Coverage: {best_coverage:.2%}")
    return best_threshold, best_preds, best_f1, best_coverage

#best_t, preds, f1, coverage = find_best_threshold(pred_df, prob_df, MODELS, metric=precision_score)

preds, coverage, f1 = thresholded_mixture_of_experts_soft(pred_df, prob_df, MODELS, metric=precision_score, gold_col="nli", threshold=0.999)

thresholded_mixture_of_experts_hard(full_pred_df, full_prob_df, MODELS, metric=precision_score, threshold=0.999, eval=False)
thresholded_mixture_of_experts_soft(full_pred_df, full_prob_df, MODELS, metric=precision_score, threshold=0.999, eval=False)
# mixture_of_experts_probabilities(pred_df, prob_df, MODELS, precision_score, threshold=0.9991)
# mixture_of_experts_probabilities(pred_df, prob_df, MODELS, f1_score, threshold=0.9991)
# preds, coverage, f1 = thresholded_predictions(df=pred_df, prob_df=prob_df, model_cols=MODELS, threshold=0.9)
# preds, coverage, f1 = thresholded_predictions(df=pred_df, prob_df=prob_df, model_cols=MODELS, threshold=0.99)
# preds, coverage, f1 = thresholded_predictions(df=pred_df, prob_df=prob_df, model_cols=MODELS, threshold=0.999)
# preds, coverage, f1 = thresholded_predictions(df=pred_df, prob_df=prob_df, model_cols=MODELS, threshold=0.99968)