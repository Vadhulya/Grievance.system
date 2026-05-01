def get_priority(text: str):
    text = text.lower()

    # High priority keywords
    if any(word in text for word in ["fire", "shock", "blast", "accident", "leak"]):
        return {
            "priority": "High",
            "score": 90
        }

    # Medium priority keywords
    if any(word in text for word in ["not working", "delay", "problem", "issue"]):
        return {
            "priority": "Medium",
            "score": 60
        }

    # Default
    return {
        "priority": "Low",
        "score": 30
    }