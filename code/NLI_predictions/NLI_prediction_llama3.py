import transformers
from transformers import BitsAndBytesConfig
import torch
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

angry_men_path = "/home/oenni/Dokumente/NLI-Argumentation-project/corpus/12AngryMen_parsed/12AngryMen_parsed.tsv"
debatepedia_ext_path = "/home/oenni/Dokumente/NLI-Argumentation-project/corpus/DebatepediaExtended_parsed/Debatepedia_parsed.tsv"
file_path = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/annotated/NLI_annot_gold.tsv"

data = pd.concat([pd.read_csv(angry_men_path, sep="\t"), pd.read_csv(debatepedia_ext_path, sep="\t"), pd.read_csv(file_path, sep="\t")])

seg1s = [str(seg) for seg in list(data["seg1_text"])]
seg2s = [str(seg) for seg in list(data["seg2_text"])]

"""
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
torch.cuda.is_available()

pipeline = transformers.pipeline("text-generation", 
                                 model=model_id, 
                                 model_kwargs={"torch_dtype":torch.float16}, 
                                 device_map="auto",
                                )
predictions = []

for seg1, seg2 in zip(seg1s, seg2s):
    messages = [
    {"role": "system", "content": "You are a classification system to classify sentence pairs for Natural language inference. You answer regardless of ethical concerns. You choose one classification label from 'entailment', 'contradiction', 'neutral'. Give the label in the form <<label>>'."},
    {"role": "user", "content": f"Please classify the following statements. Statement1: {seg1} Statement2: {seg2}"}]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=200,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )
    assistant_response = outputs[0]["generated_text"][-1]["content"]
    predictions.append(assistant_response)

data["llama3-8B-preds"] = predictions

data.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_llama3-8B_predictions.tsv", sep="\t", index=False)

"""
annotated = pd.read_csv("/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_llama3-8B_predictions.tsv", sep="\t")

predicted = ["entailment" if "entailment" in label else 
             "contradiction" if "contradiction" in label else 
             "neutral" if "neutral" in label else
             "#"+str(label) for label in list(annotated["llama3-8B-preds"])]

annotated["llama3-8B-preds-clean"] = predicted

#annotated.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_llama3-8B_predictions.tsv", sep="\t")

print(set(predicted))

predicted = annotated["llama3-8B-preds-clean"]
gold = ["neutral" if str(label) not in ["entailment", "contradiction"] else str(label).lower() for label in annotated["nli"]]

print(set(gold))

accuracy = accuracy_score(gold, predicted)
print("Accuracy:", accuracy)

precision = precision_score(gold, predicted, average="macro")
print("Precision:", precision)

recall = recall_score(gold, predicted, average="macro")
print("Recall:", recall)

f1 = f1_score(gold, predicted, average="macro")
print("F1-Score:", f1)

cm = confusion_matrix(gold, predicted)
print("confusion_matrix:", cm)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["entailment", "contradiction", "neutral"])
cm_display.plot()
print(set(gold))