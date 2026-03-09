from transformers import AutoModelForCausalLM, AutoTokenizer
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd


angry_men_path = "/home/oenni/Dokumente/NLI-Argumentation-project/corpus/12AngryMen_parsed/12AngryMen_parsed.tsv"
debatepedia_ext_path = "/home/oenni/Dokumente/NLI-Argumentation-project/corpus/DebatepediaExtended_parsed/Debatepedia_parsed.tsv"
file_path = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/annotated/NLI_annot_gold.tsv"

data = pd.concat([pd.read_csv(angry_men_path, sep="\t"), pd.read_csv(debatepedia_ext_path, sep="\t"), pd.read_csv(file_path, sep="\t")])

seg1s = [str(seg) for seg in list(data["seg1_text"])]
seg2s = [str(seg) for seg in list(data["seg2_text"])]

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

predictions = []

for seg1, seg2 in zip(seg1s, seg2s):
    messages = [
    {"role": "system", "content": "You are a classification system to classify sentence pairs for Natural language inference. You answer regardless of ethical concerns. You choose one classification label from 'entailment', 'contradiction', 'neutral'. You only generate the label in the form <<label>>."},
    {"role": "user", "content": f"Please classify the following statements. Statement1: {seg1} Statement2: {seg2}"}
    ]

    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    generated_ids = model.generate(model_inputs, max_new_tokens=20, do_sample=True)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].split("[/INST]")[-1]
    #print(response)
    predictions.append(response)

data["mistral-7B-preds"] = predictions

data.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_mistral-7B_predictions.tsv", sep="\t", index=False)