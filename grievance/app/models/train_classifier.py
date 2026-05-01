import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Load data
df = pd.read_csv("data/complaints.csv")

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

# Save label mapping
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label Map:", label_map)

dataset = Dataset.from_pandas(df)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize)

dataset = dataset.train_test_split(test_size=0.2)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)

# Training
training_args = TrainingArguments(
    output_dir="./model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()

# Save model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")