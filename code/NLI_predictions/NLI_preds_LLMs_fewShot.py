import ollama
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

PRIME_NUM = 32 # Number of few-shot examples to use
MODEL_NAME = 'gemma3:4b'  # Ollama model name
INPUT_FILE = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/nli_gold_df.tsv"  # CSV with columns: premise, hypothesis
OUTPUT_FILE = f'/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_noUnrelated_{MODEL_NAME}_{PRIME_NUM}Shot.tsv'     # Save intermediate & final results here
FEW_SHOT_FILE = f"/home/oenni/Dokumente/NLI-Argumentation-project/priming/LLM_primes_{PRIME_NUM}_42.tsv"  # TSV with columns: premise, hypothesis, label
BATCH_SIZE = 20
MAX_RETRIES = 3
SLEEP_BETWEEN_BATCHES = 0.5         # seconds
SAVE_EVERY_N_BATCHES = 50           # how often to checkpoint

# ========== NLI Core ==========

def load_few_shot_examples(few_shot_file, n_examples=PRIME_NUM):
    """Load few-shot examples from a TSV file."""
    if not os.path.exists(few_shot_file):
        print(f"[Few-shot] File not found: {few_shot_file}")
        return []
    df = pd.read_csv(few_shot_file, sep="\t")
    # Ensure columns exist
    required_cols = {"seg1_text", "seg2_text", "nli"}
    if not required_cols.issubset(df.columns):
        print(f"[Few-shot] File must contain columns: {required_cols}")
        return []
    # Select first n_examples
    print("Priming run with few-shot examples:", n_examples)
    return list(df[["seg1_text", "seg2_text", "nli"]].itertuples(index=False, name=None))[:n_examples]

FEW_SHOT_EXAMPLES = load_few_shot_examples(FEW_SHOT_FILE, n_examples=PRIME_NUM)

def build_batch_prompt(batch):
    prompt = (
        "You are an NLI classifier. You classify the relation between pairs of sentences according to the following short definitions: 'entailment' are sentences, for which the majority of people would agree that the second statement follows automatically from the first; this rules out any conclusion that can be drawn based on specialist knowledge. 'contradiction' are those statements for which the majority of people would agree that the statements cannot be true at the same time. 'neutral' are all statements, which are not covered by the first two. You answer regardless of ethical concerns.\n"
        "For each pair, respond with only one word: entailment, contradiction, or neutral.\n\n"
        "Here are some examples:\n"
    )
    # Add few-shot examples
    for i, (premise, hypothesis, label) in enumerate(FEW_SHOT_EXAMPLES, 1):
        prompt += f"Example {i}:\nPremise: {premise}\nHypothesis: {hypothesis}\nLabel: {label}\n\n"
    prompt += "Now classify the following pairs:\n"
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

    if f'{MODEL_NAME}_{PRIME_NUM}Shot_nli_label' not in df.columns:
        df[f'{MODEL_NAME}_{PRIME_NUM}Shot_nli_label'] = None

    # Resume support: Skip already labeled
    labeled_mask = df[f'{MODEL_NAME}_{PRIME_NUM}Shot_nli_label'].notnull()
    total = len(df)
    print(f"Loaded dataset with {total} rows. Already labeled: {labeled_mask.sum()}")

    for i in range(0, total, BATCH_SIZE):
        batch_df = df.iloc[i:i + BATCH_SIZE]

        if batch_df[f'{MODEL_NAME}_{PRIME_NUM}Shot_nli_label'].notnull().all():
            continue  # skip already done

        batch = list(batch_df[['seg1_text', 'seg2_text']].itertuples(index=False, name=None))
        labels = classify_batch_with_retries(batch)
        df.loc[i:i + len(labels) - 1, f'{MODEL_NAME}_{PRIME_NUM}Shot_nli_label'] = labels

        # Progress reporting
        done = (i + len(labels))
        percent = 100 * done / total
        print(f"[Progress] {done}/{total} ({percent:.1f}%) rows processed.")

        # Save every N batches
        if ((i // BATCH_SIZE) + 1) % SAVE_EVERY_N_BATCHES == 0:
            save_checkpoint(df, OUTPUT_FILE)
            print(df)

        time.sleep(SLEEP_BETWEEN_BATCHES)

    # Final save
    save_checkpoint(df, OUTPUT_FILE)
    print("✅ All done.")

# ========== Run ==========

classify_large_dataset()
comparison_df = pd.read_csv(INPUT_FILE, sep="\t")
print(comparison_df["nli"].unique())
df = pd.read_csv(OUTPUT_FILE, sep="\t")
df["nli"] = comparison_df["nli"]

predicted = ["neutral" if pred == "unknown" else str(pred) for pred in df[f'{MODEL_NAME}_{PRIME_NUM}Shot_nli_label']]
gold = df["nli"]

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
# cm_display.plot()
# print(set(gold))

"""file_path = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/nli_gold_df.tsv"

data = pd.read_csv(file_path, sep="\t")

seg1s = [str(seg) for seg in list(data["seg1_text"])]
seg2s = [str(seg) for seg in list(data["seg2_text"])]


model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

predictions = []

for seg1, seg2 in zip(seg1s, seg2s):
    messages = [
    {"role": "system", "content": "You are a classification system to classify sentence pairs for Natural language inference. You answer regardless of ethical concerns. You choose one classification label from 'entailment', 'contradiction', 'neutral'. Please give the label in the form <<label>>'."},
    {"role": "user", "content": f"Please classify the following statements. Statement1: {seg1} Statement2: {seg2}"}]
    

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=10,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    predictions.append(response)

data["qwen2-preds"] = predictions

data.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_qwen2_predictions.tsv", sep="\t", index=False)



annotated = pd.read_csv("/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_qwen2_predictions.tsv", sep="\t")

predicted = ["entailment" if "entailment" in label else 
             "contradiction" if "contradiction" in label else 
             "neutral" if "neutral" in label else
             "#"+str(label) for label in list(annotated["qwen2-preds"])]

annotated["qwen2-preds-clean"] = predicted

print(set(predicted))

annotated.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_qwen2_predictions.tsv", sep="\t")
"""
