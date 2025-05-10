# src/hallu_detector/evaluate.py

def compute_metrics(labels):
    """
    Given a list of labels like ['correct', 'hallucinated', 'correct'],
    return a dictionary with hallucination rate.
    """
    total = len(labels)
    if total == 0:
        return {"hallucination_rate": None}
    
    hallu_count = sum(1 for label in labels if label == "hallucinated")
    return {"hallucination_rate": hallu_count / total}
