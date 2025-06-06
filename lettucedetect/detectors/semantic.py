from lettucedetect.detectors.base import BaseDetector
from sentence_transformers import SentenceTransformer, util
import re

SEMANTIC_SIMILARITY_THRESHOLD = 0.75

class SemanticBasedDetector(BaseDetector):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the SemanticBasedDetector with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)

    def _predict(self, context: list[str], answer: str, output_format: str = "spans") -> list:
        context_str = " ".join(context)
        
        # Split into sentences
        context_sentences = [
            s.strip() for ctx in context 
            for s in re.split(r'(?<=[.?!])\s+', ctx) 
            if s.strip()
        ]
        answer_sentences = re.findall(r'[^.?!]+[.?!]', answer)

        # Compute embeddings
        context_embeddings = self.model.encode(context_sentences, convert_to_tensor=True)
        hallucinations = []

        if output_format == "spans":
            for match in re.finditer(r'[^.?!]+[.?!]', answer):
                sentence = match.group().strip()
                sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)
                
                # Compute max semantic similarity
                cosine_scores = util.cos_sim(sentence_embedding, context_embeddings)
                max_score = float(cosine_scores.max())

                if max_score < SEMANTIC_SIMILARITY_THRESHOLD:
                    hallucinations.append({
                        "text": sentence,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 1 - max_score
                    })
            return hallucinations

        elif output_format == "tokens":
            token_outputs = []
            context_embedding = self.model.encode(context_str, convert_to_tensor=True)
            for match in re.finditer(r'\b\w+\b', answer):
                token = match.group()
                token_embedding = self.model.encode(token, convert_to_tensor=True)
                similarity = float(util.cos_sim(token_embedding, context_embedding)[0][0])
                is_hallucinated = similarity < SEMANTIC_SIMILARITY_THRESHOLD

                token_outputs.append({
                    "token": token,
                    "pred": int(is_hallucinated),
                    "prob": 1 - similarity if is_hallucinated else 0.01
                })
            return token_outputs

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
        print("Entered predict method in Semantic-based class")
        return self._predict(context, answer, output_format)
    
    def predict_prompt_batch(self, prompts, answers, output_format="tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompts and answers.

        :param prompts: List of prompt strings.
        :param answers: List of answer strings.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        return [self._predict(p, a, output_format) for p, a in zip(prompts, answers)]
    
