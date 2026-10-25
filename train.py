import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models" / "distilbert_sarcasm"
VIS_DIR = MODEL_DIR / "visualizations"
RESULTS_DIR = MODEL_DIR / "checkpoints"
LOG_DIR = MODEL_DIR / "logs"

# 1. LOAD DATASET
print("Loading dataset...")
data_path = ROOT_DIR / "train-balanced-sarcasm.csv"

# Use a smaller sample for local runs so the script completes in a reasonable time.
# Increase this if you want a longer training job on a stronger machine.
df = pd.read_csv(data_path, nrows=2000, usecols=["comment", "label"])

# The dataset columns typically include 'comment' (text) and 'label' (0 or 1)
df = df[["comment", "label"]].dropna().copy()
df["label"] = df["label"].astype(int)

# 2. DATA VISUALIZATION
print("Generating visualizations...")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Plot 1: Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="label", data=df, hue="label", palette="Set2", legend=False)
plt.title("Distribution of Sarcastic vs Non-Sarcastic Comments")
plt.xlabel("Label (0 = Normal, 1 = Sarcastic)")
plt.ylabel("Count")
plt.savefig(VIS_DIR / "class_distribution.png")
plt.close()

# Plot 2: Comment Length Distribution
df["comment_length"] = df["comment"].astype(str).apply(len)
plt.figure(figsize=(8, 4))
sns.histplot(df["comment_length"], bins=50, kde=True, color="purple")
plt.title("Distribution of Comment Lengths")
plt.xlabel("Character Count")
plt.ylabel("Frequency")
plt.savefig(VIS_DIR / "comment_lengths.png")
plt.close()

# Plot 3: Label balance summary
label_counts = df["label"].value_counts().sort_index()
plt.figure(figsize=(5, 5))
plt.pie(label_counts.values, labels=["Normal", "Sarcastic"], autopct="%1.1f%%", startangle=90)
plt.title("Class Balance")
plt.savefig(VIS_DIR / "class_balance.png")
plt.close()
print(
    "Visualizations saved successfully in the 'visualizations/' directory!"
)

# 3. PREPARE FOR MODELING (Using DistilBERT for efficiency)
# DistilBERT is lighter and faster than BERT/RoBERTa while retaining ~97% of their performance.
MODEL_NAME = "distilbert-base-uncased"  # You can swap to 'bert-base-uncased' or 'roberta-base'

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["comment"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class SarcasmDataset(torch.utils.data.Dataset):

  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx] for key, val in self.encodings.items()}
    item["labels"] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=128
)
val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, max_length=128
)

train_dataset = SarcasmDataset(train_encodings, train_labels)
val_dataset = SarcasmDataset(val_encodings, val_labels)

# 4. MODEL TRAINING
print(f"Initializing model: {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

training_args = TrainingArguments(
    output_dir=str(RESULTS_DIR),
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("Starting training...")
train_result = trainer.train()

# Save model and training metrics for later inspection.
final_model_dir = MODEL_DIR / "final_model"
trainer.save_model(str(final_model_dir))
trainer.save_state()

history = trainer.state.log_history
metric_rows = []
for entry in history:
    if "loss" in entry:
        metric_rows.append(
            {
                "step": entry.get("step"),
                "loss": entry.get("loss"),
                "learning_rate": entry.get("learning_rate"),
                "epoch": entry.get("epoch"),
            }
        )

if metric_rows:
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULTS_DIR / "training_metrics.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df["step"], metrics_df["loss"], marker="o")
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_loss.png")
    plt.close()

print(f"Training complete! DistilBERT artifacts saved in {MODEL_DIR}")