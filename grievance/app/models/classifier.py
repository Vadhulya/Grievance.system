from transformers import pipeline
import torch
from app.core.config import LABELS

device = 0 if torch.cuda.is_available() else -1

print("Loading trained model...")

classifier = pipeline(
    "text-classification",
    model="./model",
    device=device
)

print("Model loaded")

LABELS = [
    "Electricity",
    "Water",
    "Sanitation",
    "Roads",
    "Public Services"
]

def classify(text: str):
    result = classifier(text)[0]

    return {
        "department": LABELS[int(result["label"].split("_")[-1])],
        "confidence": float(result["score"])
    }