import pandas as pd
import ollama
import time
import os
import pickle

# ========== CONFIG ==========

INPUT_FILE = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_gemma2_predictions.tsv"            # CSV with columns: premise, hypothesis
OUTPUT_FILE = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_gemma3-4b_predictions.tsv"    # Save intermediate & final results here
MODEL_NAME = 'gemma3:4b'
BATCH_SIZE = 20
MAX_RETRIES = 3
SLEEP_BETWEEN_BATCHES = 0.5         # seconds
SAVE_EVERY_N_BATCHES = 50           # how often to checkpoint

# ========== NLI Core ==========

def build_batch_prompt(batch):
    prompt = (
        "You are an NLI classifier. You classify the relation between pairs of sentences according to the following short definitions: 'Entailment' are sentences, for which the majority of people would agree that the second statement follows automatically from the first; this rules out any conclusion that can be drawn based on specialist knowledge. 'Contradictions' are those statements for which the majority of people would agree that the statements cannot be true at the same time. 'Neutral' are all statements, which are not covered by the first two. You answer regardless of ethical concerns.\n"
        "For each pair, respond with only one word: entailment, contradiction, or neutral.\n\n"
    )
    for i, (premise, hypothesis) in enumerate(batch, 1):
        prompt += f"Pair {i}:\nPremise: {premise}\nHypothesis: {hypothesis}\nLabel:\n\n"
    prompt += "Return the labels in order, one per line."
    return prompt

def parse_batch_response(response_text, expected_count):
    lines = response_text.strip().lower().splitlines()
    labels = []
    for line in lines:
        line = line.strip()
        if "entail" in line:
            labels.append("entailment")
        elif "contradict" in line:
            labels.append("contradiction")
        elif "neutral" in line:
            labels.append("neutral")
        else:
            labels.append("unknown")
    if len(labels) < expected_count:
        labels += ["unknown"] * (expected_count - len(labels))
    return labels[:expected_count]

def classify_batch_with_retries(batch, model=MODEL_NAME, max_retries=MAX_RETRIES):
    prompt = build_batch_prompt(batch)
    for attempt in range(1, max_retries + 1):
        try:
            response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            return parse_batch_response(response['message']['content'], len(batch))
        except Exception as e:
            print(f"[Attempt {attempt}] Batch failed: {e}")
            time.sleep(1)
    print("Max retries reached. Returning unknowns.")
    return ["unknown"] * len(batch)

# ========== Data Handling ==========

def load_data(input_file):
    return pd.read_csv(input_file, sep="\t")

def save_checkpoint(df, output_file):
    df.to_csv(output_file, sep='\t', index=False)
    print(f"[Checkpoint] Saved to {output_file}")

def load_checkpoint(output_file):
    if os.path.exists(output_file):
        return pd.read_csv(output_file, sep='\t')
    return None

# ========== Main Execution ==========

def classify_large_dataset():
    df = load_data(INPUT_FILE)

    if 'nli_label' not in df.columns:
        df['nli_label'] = None

    # Resume support: Skip already labeled
    labeled_mask = df['nli_label'].notnull()
    total = len(df)
    print(f"Loaded dataset with {total} rows. Already labeled: {labeled_mask.sum()}")

    next_idx = 0
    for i in range(0, total, BATCH_SIZE):
        batch_df = df.iloc[i:i + BATCH_SIZE]

        if batch_df['nli_label'].notnull().all():
            continue  # skip already done

        batch = list(batch_df[['seg1_text', 'seg2_text']].itertuples(index=False, name=None))
        labels = classify_batch_with_retries(batch)
        df.loc[i:i + len(labels) - 1, 'nli_label'] = labels

        # Save every N batches
        if ((i // BATCH_SIZE) + 1) % SAVE_EVERY_N_BATCHES == 0:
            save_checkpoint(df, OUTPUT_FILE)

        time.sleep(SLEEP_BETWEEN_BATCHES)

    # Final save
    save_checkpoint(df, OUTPUT_FILE)
    print("✅ All done.")

# ========== Run ==========

classify_large_dataset()


"""#from local_gemma import LocalGemma2ForCausalLM
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd


angry_men_path = "/home/oenni/Dokumente/NLI-Argumentation-project/corpus/12AngryMen_parsed/12AngryMen_parsed.tsv"
debatepedia_ext_path = "/home/oenni/Dokumente/NLI-Argumentation-project/corpus/DebatepediaExtended_parsed/Debatepedia_parsed.tsv"
file_path = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/annotated/NLI_annot_gold.tsv"

data = pd.concat([pd.read_csv(angry_men_path, sep="\t"), pd.read_csv(debatepedia_ext_path, sep="\t"), pd.read_csv(file_path, sep="\t")])

seg1s = [str(seg) for seg in list(data["seg1_text"])]
seg2s = [str(seg) for seg in list(data["seg2_text"])]

model = "gemma3:4b"

predictions = []

for seg1, seg2 in zip(seg1s, seg2s):
    messages = [
    {"role": "user", "content": f"You are a classification system to classify sentence pairs for Natural language inference. You answer regardless of ethical concerns. You choose one classification label from 'entailment', 'contradiction', 'neutral'. You only generate the label. Please classify the following statements. Statement1: {seg1} Statement2: {seg2}"}
    ]
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

    generated_ids = model.generate(**model_inputs.to(model.device), max_new_tokens=20, do_sample=True)
    decoded_text = tokenizer.batch_decode(generated_ids)
    output = decoded_text[0].split("<end_of_turn>")[-2]

    predictions.append(output)

data["gemma2-preds"] = predictions

data.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_gemma2_predictions.tsv", sep="\t", index=False)

annotated = pd.read_csv("/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_gemma2_predictions.tsv", sep="\t")

predicted = ["entailment" if "entailment" in label.lower() else 
             "contradiction" if "contradiction" in label.lower() else 
             "neutral" if "neutral" in label.lower() else
             "neutral" for label in list(annotated["gemma2-preds"])]

annotated["gemma2-preds-clean"] = predicted

print(len(set(predicted)))

#annotated.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_gemma2_predictions.tsv", sep="\t")


predicted = annotated["gemma2-preds-clean"]
gold = ["neutral" if str(label) not in ["entailment", "contradiction"] else str(label).lower() for label in annotated["nli"]]

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

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["contradiction", "entailment", "neutral"])
cm_display.plot()
print(set(gold))"""