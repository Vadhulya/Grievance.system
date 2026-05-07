from transformers import pipeline
import torch
import re

device = 0 if torch.cuda.is_available() else -1

# AI sentiment/context model
priority_classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

# Emergency keywords
CRITICAL_KEYWORDS = [
    "explosion",
    "fire",
    "blast",
    "electrocution",
    "collapsed",
    "danger",
    "short circuit",
    "transformer exploded",
    "accident",
    "shock"
]

HIGH_KEYWORDS = [
    "6 hours",
    "8 hours",
    "whole day",
    "water leakage",
    "overflow",
    "sewage",
    "power outage",
    "pipeline burst"
]

MEDIUM_KEYWORDS = [
    "delay",
    "not working",
    "garbage",
    "damaged road",
    "street light"
]

def extract_duration_hours(text):

    match = re.search(r"(\\d+)\\s*hours?", text.lower())

    if match:
        return int(match.group(1))

    return 0

def get_priority(text: str):

    text_lower = text.lower()

    # Critical override

    for word in CRITICAL_KEYWORDS:
        if word in text_lower:
            return {
                "priority": "Critical",
                "score": 95
            }


    # Duration analysis

    hours = extract_duration_hours(text_lower)

    if hours >= 6:
        return {
            "priority": "High",
            "score": 85
        }


    # High keyword analysis

    for word in HIGH_KEYWORDS:
        if word in text_lower:
            return {
                "priority": "High",
                "score": 80
            }


    # Medium keyword analysis

    for word in MEDIUM_KEYWORDS:
        if word in text_lower:
            return {
                "priority": "Medium",
                "score": 60
            }

    # AI semantic analysis
    
    ai_result = priority_classifier(text)[0]

    ai_score = float(ai_result["score"])

    if ai_score > 0.90:
        return {
            "priority": "Medium",
            "score": round(ai_score * 100, 2)
        }

    return {
        "priority": "Low",
        "score": round(ai_score * 100, 2)
    }
