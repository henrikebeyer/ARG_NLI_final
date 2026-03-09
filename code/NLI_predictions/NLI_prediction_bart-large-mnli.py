import pandas as pd
import os
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F

# ========== CONFIG ==========
MODEL_NAME = "facebook/bart-large-mnli"
model_name = MODEL_NAME.split("/")[-1]
INPUT_FILE = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/nli_gold_df.tsv"
OUTPUT_FILE = f'/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/noUnrelated/preds/all_noUnrelated_{model_name}_ft.tsv'
OUTPUT_FILE2 = f'/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/noUnrelated_decoderOnly/probs/all_noUnrelated_{model_name}_ft.tsv'
MODEL_OUTPUT_DIR = f'/media/oenni/OenniDrive/Dokumente/NLI-Argumentation-project/NLI_models/other/{model_name}_ft'
BATCH_SIZE = 32
SAVE_EVERY = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_MAPPING = ['contradiction', 'neutral', 'entailment']
SEED = 42

try:
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
except Exception as e:
    pass

try:
    os.makedirs(os.path.dirname(OUTPUT_FILE2), exist_ok=True)
except Exception as e:
    pass

print(DEVICE)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    precision = precision_score(labels, preds, average="macro")
    return {"precision": precision}

import os
import shutil

def cleanup_checkpoints(output_dir, best_model_dir):
    """
    Removes all checkpoints except the best model.

    Args:
        output_dir: str, directory where checkpoints are stored
        best_model_dir: str, path to the best model to keep
    """
    for sub in os.listdir(output_dir):
        path = os.path.join(output_dir, sub)
        if os.path.isdir(path) and path != best_model_dir:
            print(f"Removing {path}")
            shutil.rmtree(path)

# ========== Load Data & Split ==========
df = pd.read_csv(INPUT_FILE, sep="\t")
np.random.seed(SEED)
indices = np.random.permutation(len(df))
train_idx, test_idx = indices[:100], indices#[100:]
df_train = df.iloc[train_idx].reset_index(drop=True)
df_test = df#.iloc[test_idx].reset_index(drop=True)
print(df_test)

# ========== Dataset ==========
class NLIDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256, label_col="nli"):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {l: i for i, l in enumerate(LABEL_MAPPING)}
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(
            str(row['seg1_text']),
            str(row['seg2_text']),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.label_col in row:
            item['labels'] = torch.tensor(self.label2id[str(row[self.label_col]).lower()])
        return item

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)

# ========== Fine-tune ==========
train_dataset = NLIDataset(df_train, tokenizer)
# val_dataset = NLIDataset(df_val, tokenizer)
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=10,
    seed=SEED,
    evaluation_strategy="no",
    report_to="none"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    #eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # <-- add this
)
trainer.train()
best_model_dir = "/media/oenni/OenniDrive/Dokumente/NLI-Argumentation-project/NLI_models/other/bart-large-mnli_ft/checkpoint-50" #trainer.state.best_model_checkpoint
print("Best model checkpoint:", trainer.state.best_model_checkpoint)
cleanup_checkpoints(MODEL_OUTPUT_DIR, best_model_dir)
model.eval()

# ========== Inference with Progress and Periodic Saving ==========
df_test = df_test.copy()
df_test[f'{model_name}_ft'] = pd.NA

# Add columns for probabilities
for label in LABEL_MAPPING:
    df_test[f'{model_name}_ft_prob_{label}'] = pd.NA

dataset = NLIDataset(df_test, tokenizer, label_col=None)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

completed = 0
batch_num = 0

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Classifying"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).cpu().numpy()  # shape: (batch_size, num_labels)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        labels = [LABEL_MAPPING[p] for p in preds]
        start = batch_num * BATCH_SIZE
        end = start + len(labels)
        df_test.iloc[start:end, df_test.columns.get_loc(f'{model_name}_ft')] = labels

        # Save probabilities for each label
        for i, label in enumerate(LABEL_MAPPING):
            df_test.iloc[start:end, df_test.columns.get_loc(f'{model_name}_ft_prob_{label}')] = probs[:, i]

        completed += len(labels)
        batch_num += 1

        if batch_num % SAVE_EVERY == 0:
            df_test.to_csv(OUTPUT_FILE, sep='\t', index=False)
            # Save probabilities to a separate file
            prob_cols = [f'{model_name}_ft_prob_{label}' for label in LABEL_MAPPING]
            df_test[prob_cols].to_csv(OUTPUT_FILE.replace('.tsv', '_probs.tsv'), sep='\t', index=False)
            print(f"[Checkpoint] Saved after {completed} examples.")

# ========== Final Save ==========
df_test.to_csv(OUTPUT_FILE2, sep='\t', index=False)
# Save probabilities to a separate file
print(f"✅ Final save complete. Total labeled: {completed}")

# ========== Evaluation ==========
predicted = df_test[f'{model_name}_ft'].astype(str).tolist()
gold = df_test['nli'].astype(str).tolist()

accuracy = accuracy_score(gold, predicted)
print("Accuracy:", accuracy)
precision = precision_score(gold, predicted, average="macro")
print("Precision:", precision)
recall = recall_score(gold, predicted, average="macro")
print("Recall:", recall)
f1 = f1_score(gold, predicted, average="macro")
print("F1-Score:", f1)
cm = confusion_matrix(gold, predicted, labels=LABEL_MAPPING)
print("confusion_matrix:", cm)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_MAPPING).plot()