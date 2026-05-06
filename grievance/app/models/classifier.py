from transformers import pipeline
import torch
import json
import math

device = 0 if torch.cuda.is_available() else -1

print("Loading trained model...")

classifier = pipeline(
    "text-classification",
    model="./model",
    device=device
)

with open("model/labels.json", "r") as f:
    LABELS = json.load(f)

print("Model loaded")

def classify(text: str):

    result = classifier(text)[0]

    label_index = int(result["label"].split("_")[-1])

    confidence = float(result["score"])

    # Fix invalid float values
    if math.isnan(confidence) or math.isinf(confidence):
        confidence = 0.0

    return {
        "department": LABELS[label_index],
        "confidence": confidence
    }
