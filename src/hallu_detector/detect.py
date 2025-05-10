def is_hallucinated(answer, correct_answer):
    """
    Check if the model's answer matches the correct answer (roughly).
    Ignores case and extra punctuation.
    """
    if not answer and correct_answer:
        return True  # empty answer = hallucination

    if not correct_answer:
        return False  # no ground truth = can't call it hallucinated

    answer = answer.strip().lower()
    correct_answer = correct_answer.strip().lower()

    return correct_answer not in answer
