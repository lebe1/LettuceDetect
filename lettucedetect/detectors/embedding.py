from lettucedetect.detectors.base import BaseDetector
import numpy as np
import re
from typing import List
from numpy.linalg import norm

# Use your actual model2vec-compatible import here
import model2vec

EMBEDDING_SIMILARITY_THRESHOLD = 0.75

class StaticEmbeddingDetector(BaseDetector):
    def __init__(self, model: str = "minishlab/potion-base-8M",):
        """
        Initializes the sentence embedding model internally.
        """
        print("Loading static embedding model: ", model)
        self.model = model2vec.StaticModel.from_pretrained(model)  

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if norm(v1) == 0 or norm(v2) == 0:
            return 0.0
        return np.dot(v1, v2) / (norm(v1) * norm(v2))

    def _predict(self, context: List[str], answer: str, output_format: str = "spans") -> List[dict]:
        if output_format != "spans":
            raise ValueError("This detector only supports 'spans' output format with sentence-level embeddings.")

        context_text = " ".join(context).strip().lower()
        context_embedding = self.model.encode(context_text)

        hallucinated_spans = []

        for match in re.finditer(r'[^.?!]+[.?!]', answer):
            sentence = match.group().strip()
            if not sentence:
                continue

            sentence_embedding = self.model.encode(sentence)
            sim = self._cosine_similarity(sentence_embedding, context_embedding)

            if sim < EMBEDDING_SIMILARITY_THRESHOLD:
                hallucinated_spans.append({
                    "text": sentence,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 1 - sim
                })

        return hallucinated_spans

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "spans") -> list:
        return self._predict([prompt], answer, output_format)

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "spans",
    ) -> list:
        print("Entered predict method in StaticEmbeddingDetector class")
        return self._predict(context, answer, output_format)

    def predict_prompt_batch(self, prompts: List[str], answers: List[str], output_format: str = "spans") -> list:
        return [self._predict([p], a, output_format) for p, a in zip(prompts, answers)]
