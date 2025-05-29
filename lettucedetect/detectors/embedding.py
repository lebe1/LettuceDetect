from lettucedetect.detectors.base import BaseDetector
import numpy as np
import re
from typing import List
from numpy.linalg import norm
import gensim.downloader as api

EMBEDDING_SIMILARITY_THRESHOLD = 0.75

class StaticEmbeddingDetector(BaseDetector):
    def __init__(self, embedding_model_name="glove-wiki-gigaword-100"):
        # Load pre-trained static word embeddings
        print(f"Loading embedding model: {embedding_model_name}")
        self.model = api.load(embedding_model_name)
        self.vector_size = self.model.vector_size

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if norm(v1) == 0 or norm(v2) == 0:
            return 0.0
        return np.dot(v1, v2) / (norm(v1) * norm(v2))

    def _get_embedding(self, word: str) -> np.ndarray:
        return self.model[word] if word in self.model else np.zeros(self.vector_size)

    def _predict(self, context: List[str], answer: str, output_format: str = "tokens") -> List:
        context_text = " ".join(context).lower()
        context_tokens = re.findall(r'\b\w+\b', context_text)
        context_embeddings = [self._get_embedding(tok) for tok in context_tokens]

        if output_format == "tokens":
            token_outputs = []
            for match in re.finditer(r'\b\w+\b', answer):
                token = match.group().lower()
                token_embedding = self._get_embedding(token)

                if norm(token_embedding) == 0:
                    # Token is out-of-vocabulary
                    token_outputs.append({
                        "token": token,
                        "pred": 1,
                        "prob": 1
                    })
                    continue

                # Compute maximum similarity with context
                similarities = [
                    self._cosine_similarity(token_embedding, ctx_emb)
                    for ctx_emb in context_embeddings
                ]
                max_sim = max(similarities) if similarities else 0.0
                is_hallucinated = max_sim < EMBEDDING_SIMILARITY_THRESHOLD

                token_outputs.append({
                    "token": token,
                    "pred": int(is_hallucinated),
                    "prob": 1 - max_sim if is_hallucinated else 0.01
                })
            return token_outputs

        elif output_format == "spans":
            hallucinated_spans = []
            for match in re.finditer(r'[^.?!]+[.?!]', answer):
                sentence = match.group()
                tokens = re.findall(r'\b\w+\b', sentence)
                if not tokens:
                    continue

                hallucinated = 0
                for token in tokens:
                    emb = self._get_embedding(token.lower())
                    if norm(emb) == 0:
                        hallucinated += 1
                        continue
                    max_sim = max([self._cosine_similarity(emb, ctx_emb) for ctx_emb in context_embeddings], default=0.0)
                    if max_sim < EMBEDDING_SIMILARITY_THRESHOLD:
                        hallucinated += 1

                hallucination_ratio = hallucinated / len(tokens)
                if hallucination_ratio > 0.5:
                    hallucinated_spans.append({
                        "text": sentence,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": hallucination_ratio
                    })
            return hallucinated_spans

        else:
            raise ValueError("Invalid output_format. Use 'tokens' or 'spans'.")

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        return self._predict([prompt], answer, output_format)

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
    ) -> list:
        print("Entered predict method in StaticEmbeddingDetector class")
        return self._predict(context, answer, output_format)

    def predict_prompt_batch(self, prompts, answers, output_format="tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompts and answers.

        :param prompts: List of prompt strings.
        :param answers: List of answer strings.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        return [self._predict(p, a, output_format) for p, a in zip(prompts, answers)]