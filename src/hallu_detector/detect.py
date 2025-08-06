import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import string

"""
Hallucination detection using semantic similarity and NLI.
"""

# Load models
st_model = SentenceTransformer('all-MiniLM-L6-v2')
nli = pipeline(
    'text-classification',
    model='roberta-large-mnli',
    device=0 if torch.cuda.is_available() else -1
)

def normalize_text(text: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace.
    """
    if text is None:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())


def semantic_similarity(a: str, b: str) -> float:
    """Return cosine similarity between embeddings of a and b."""
    emb1 = st_model.encode(a, convert_to_tensor=True)
    emb2 = st_model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


def nli_entailment_probability(premise: str, hypothesis: str) -> float:
    """
    Returns probability that premise entails hypothesis using MNLI model.
    """
    res = nli(f"{premise} </s></s> {hypothesis}")
    for r in res:
        if r['label'].upper() == 'ENTAILMENT':
            return r['score']
    return 0.0


def is_hallucinated(answer: str, correct_answer: str,
                    sem_thresh: float = 0.75,
                    nli_thresh: float = 0.80) -> bool:
    """
    Return True if answer is hallucinated relative to correct_answer.
    """
    ans = normalize_text(answer)
    corr = normalize_text(correct_answer)

    if not corr:
        return False
    if not ans:
        return True
    if corr in ans or ans in corr:
        return False

    sim = semantic_similarity(ans, corr)
    if sim < sem_thresh:
        return True

    entail_p = nli_entailment_probability(correct_answer, answer)
    return entail_p < nli_thresh
