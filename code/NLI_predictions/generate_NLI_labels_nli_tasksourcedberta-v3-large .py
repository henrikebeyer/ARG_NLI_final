import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

# ========== CONFIG ==========
MODEL_NAME = "sileod/deberta-v3-large-tasksource-nli"
INPUT_FILE = '/home/oenni/Dokumente/NLI-Argumentation-project/NLI_labeling/full_corpus_qwen2_predictions.tsv'            # CSV with columns: premise, hypothesis
OUTPUT_FILE = '/home/oenni/Dokumente/NLI-Argumentation-project/NLI_labeling/full_corpus_deberta_large_v3_nli_tasksource_predictions.tsv'  
BATCH_SIZE = 62
SAVE_EVERY = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_MAPPING = ['entailment', 'neutral', 'contradiction']

print(DEVICE)

# ========== Dataset ==========
class NLIDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(
            row['seg1_text'],
            row['' \
            'seg2_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {key: val.squeeze(0) for key, val in inputs.items()}

# ========== Load Model ==========
print(f"Loading model [{MODEL_NAME}] on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# ========== Dataset ==========
class NLIDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(
            row['seg1_text'],
            row['seg2_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            **{key: val.squeeze(0) for key, val in inputs.items()},
            "row_index": row["index"] if "index" in row else idx
        }

# ========== Load or Resume ==========
if os.path.exists(OUTPUT_FILE):
    print(f"Resuming from checkpoint: {OUTPUT_FILE}")
    df_full = pd.read_csv(OUTPUT_FILE, sep='\t')
    df_full["seg1_text"] = [str(text) for text in df_full["seg1_text"]]
    df_full["seg2_text"] = [str(text) for text in df_full["seg2_text"]]

else:
    print(f"Starting new classification run from: {INPUT_FILE}")
    df_full = pd.read_csv(INPUT_FILE, sep="\t")
    df_full['nli_label_deberta_large'] = pd.NA

# df_full = pd.DataFrame(data={"premise":["A man is walking.", "A cat sleeps on the sofa.", "The sky is clear."],
#                                 "hypothesis": ["A man is moving.", "An animal is napping", "It is raining."],
#                                 "nli_label": [None, None, None]})
df_pending = df_full[df_full['nli_label_deberta_large'].isna()].reset_index()
if df_pending.empty:
    print("✅ All rows are already labeled. Nothing to do.")
    exit()

print(f"Remaining examples to classify: {len(df_pending)}")

# ========== Run Inference with Periodic Saving ==========
dataset = NLIDataset(df_pending, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

completed = 0
batch_num = 0

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Classifying"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        row_indices = batch['row_index']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        labels = [LABEL_MAPPING[p] for p in preds]
        #print(labels)
        

        df_full.loc[row_indices, 'nli_label_deberta_large'] = labels
        #print(df_full.loc[row_indices]["nli_label_deberta_large"])

        completed += len(labels)
        batch_num += 1

        if batch_num % SAVE_EVERY == 0:
            df_full.to_csv(OUTPUT_FILE, sep='\t', index=False)
            print(f"[Checkpoint] Saved after {completed} examples.")

# ========== Final Save ==========
print(df_full.value_counts("nli_label_deberta_large"))
df_full.to_csv(OUTPUT_FILE, sep='\t', index=False)
print(f"✅ Final save complete. Total labeled: {completed}")


"""import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


angry_men_path = "/home/oenni/Dokumente/NLI-Argumentation-project/corpus/parsed_corpora/12AngryMen_parsed/12AngryMen_parsed.tsv"
debatepedia_ext_path = "/home/oenni/Dokumente/NLI-Argumentation-project/corpus/parsed_corpora/DebatepediaExtended_parsed/Debatepedia_parsed.tsv"
file_path = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/annotated/NLI_annot_gold.tsv"

data = pd.concat([pd.read_csv(angry_men_path, sep="\t"), pd.read_csv(debatepedia_ext_path, sep="\t"), pd.read_csv(file_path, sep="\t")])

seg1s = [str(seg) for seg in list(data["seg1_text"])]
seg2s = [str(seg) for seg in list(data["seg2_text"])]


tokenizer = AutoTokenizer.from_pretrained("sileod/deberta-v3-large-tasksource-nli")
model = AutoModelForSequenceClassification.from_pretrained("sileod/deberta-v3-large-tasksource-nli")

predictions = []

model.eval()
for seg1, seg2 in zip(seg1s, seg2s):
    features = tokenizer([seg1], [seg2],  padding=True, truncation=True, return_tensors="pt")
    #print(features)
    with torch.no_grad():
        scores = model(**features).logits
        label_mapping = ['entailment', 'neutral', 'contradiction']
        labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
        predictions.append(labels[0])
        #print(labels)

#print(predictions)

data["deberta-v3-large-tasksource-nli-preds"] = predictions
print(data)

data.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_deberta-v3-large-tasksource-nli_predictions.tsv", sep="\t", index=False)

annotated = pd.read_csv("/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/all_deberta-v3-large-tasksource-nli_predictions.tsv", sep="\t")

gold = ["neutral" if str(label) not in ["entailment", "contradiction"] else str(label).lower() for label in annotated["nli"]]
predicted = annotated["deberta-v3-large-tasksource-nli-preds"]

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
cm_display.plot()"""