import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

import torch
import json

# Load dataset
df = pd.read_csv("data/complaints.csv")

# Encode labels
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

# Save label mapping
with open("model/labels.json", "w") as f:
    json.dump(list(le.classes_), f)

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding=True,
        truncation=True
    )

dataset = dataset.map(tokenize, batched=True)

# Rename labels for trainer
dataset = dataset.rename_column("label_encoded", "labels")

# Remove unnecessary columns
dataset = dataset.remove_columns([
    "text",
    "label",
    "language",
    "priority"
])

# Set tensor format
dataset.set_format("torch")

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)

# GPU support
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Training config
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train
trainer.train()

# Save model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Training completed successfully")
