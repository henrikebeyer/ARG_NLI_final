import pandas as pd
from sklearn.metrics import classification_report
import re

OUTPUT_FILE = "/home/henrike/ARG-NLI_project/case-study/Arg_preds_myData_train_DeepseekR1_32b.csv"

deepseek_df = pd.read_csv(OUTPUT_FILE)
extracted_explanations = []
for idx, row in deepseek_df.iterrows():
    raw_content = row["deepseek_output"]
    clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).replace("**","").strip().lower()
    explanation_candidate = clean_content.split("explanation:")[1].strip()
    if "label:" in explanation_candidate:
        explanation_candidate = explanation_candidate.split("label:")[0].replace("answer:","").replace('the correct label is "supports" and the explanation is as follows:',"").strip()
    extracted_explanations.append(explanation_candidate)
    
deepseek_df["llm_explanation"] = extracted_explanations

deepseek_df.to_csv(OUTPUT_FILE, index=False) 

def eval_preds(gold, preds, model, mode="3-way"):
    if mode == "binary":
        gold = ["related" if label in ["support", "attack"] else "unrelated" for label in gold]
        preds = ["related" if label in ["support", "attack"] else "unrelated" for label in preds]
        
        print(pd.Series(preds).value_counts())
        print(pd.Series(gold).value_counts())
        
        report_dict = classification_report(gold,preds, output_dict=True)

        # eval_dict = {"Model": model,
        #             "Related Recall": report_dict.get("related", {"recall":0.0})["recall"],
        #             "Related Precision": report_dict.get("related", {"precision":0.0})["precision"],
        #             "Realted F1": report_dict.get("related", {"f1-score":0.0})["f1-score"],
        #             "Unrelated Recall": report_dict.get("unrelated", {"recall":0.0})["recall"],
        #             "Unrelated Precision": report_dict.get("unrelated", {"precision":0.0})["precision"],
        #             "Unrelated F1": report_dict.get("unrelated", {"f1-score":0.0})["f1-score"],
        #             "Macro Recall": report_dict["macro avg"]["recall"],
        #             "Macro Precision": report_dict["macro avg"]["precision"],
        #             "Macro F1": report_dict["macro avg"]["f1-score"],
        #             "Weighted Recall": report_dict["weighted avg"]["recall"],
        #             "Weighted Precision": report_dict["weighted avg"]["precision"],
        #             "Weighted F1": report_dict["weighted avg"]["f1-score"]
        # }
    else:
        print(pd.Series(gold).value_counts())
        print(pd.Series(preds).value_counts())
        report_dict = classification_report(gold,preds, output_dict=True)
        # eval_dict = {"Model": model,
        #             "Support Recall": report_dict.get("supports", {"recall":0.0})["recall"],
        #             "Support Precision": report_dict.get("supports", {"precision":0.0})["precision"],
        #             "Support F1": report_dict.get("supports", {"f1-score":0.0})["f1-score"],
        #             "Attack Recall": report_dict.get("attacks", {"recall":0.0})["recall"],
        #             "Attack Precision": report_dict.get("attacks", {"precision":0.0})["precision"],
        #             "Attack F1": report_dict.get("attacks", {"f1-score":0.0})["f1-score"],
        #             "Unrelated Recall": report_dict.get("None", {"recall":0.0})["recall"],
        #             "Unrelated Precision": report_dict.get("None", {"precision":0.0})["precision"],
        #             "Unrelated F1": report_dict.get("None", {"f1-score":0.0})["f1-score"],
        #             "Macro Recall": report_dict["macro avg"]["recall"],
        #             "Macro Precision": report_dict["macro avg"]["precision"],
        #             "Macro F1": report_dict["macro avg"]["f1-score"],
        #             "Weighted Recall": report_dict["weighted avg"]["recall"],
        #             "Weighted Precision": report_dict["weighted avg"]["precision"],
        #             "Weighted F1": report_dict["weighted avg"]["f1-score"]
        # }
    eval_df = pd.DataFrame().from_dict(report_dict)
    
    out_path = f"results_IAMCaseStudy_{model}_{mode}_baseline.csv"
    print(f"Evaluating {model} in {mode} mode:")
    print(eval_df)
    eval_df.to_csv(out_path, index=False)
    
# pred_df = pd.read_csv(OUTPUT_FILE)

# old = pred_df["relation_type"]
# preds = pred_df["pred"]

# eval_preds(gold, preds, "deepseekR1_32b", mode="binary")
# eval_preds(gold, preds, "deepseekR1_32b", mode="3-way")
