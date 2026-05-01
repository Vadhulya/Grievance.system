from app.models.classifier import classify
from app.models.priority import get_priority

def process_complaint(text: str):
    classification = classify(text)
    priority = get_priority(text)

    return {
        "text": text,
        "classification": classification,
        "priority": priority
    }