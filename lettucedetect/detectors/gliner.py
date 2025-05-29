from lettucedetect.detectors.base import BaseDetector
from gliner import GLiNER
import re
from difflib import SequenceMatcher

ENTITY_SIMILARITY_THRESHOLD = 0.85  # for fuzzy entity comparison

class GLiNERBasedDetector(BaseDetector):
    def __init__(self, model_name: str = "urchade/gliner_multi"):
        self.model = GLiNER.from_pretrained(model_name)

    def _fuzzy_match(self, s1: str, s2: str) -> float:
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def _get_entities(self, text: str) -> set:
        entities = self.model.predict_entities(text)
        return {e["text"] for e in entities}

    def _predict(self, context: list[str], answer: str, output_format: str = "spans") -> list:
        context_text = " ".join(context)
        context_entities = self._get_entities(context_text)
        answer_entities = self._get_entities(answer)

        hallucinated_entities = set()

        for ae in answer_entities:
            max_score = max([self._fuzzy_match(ae, ce) for ce in context_entities], default=0.0)
            if max_score < ENTITY_SIMILARITY_THRESHOLD:
                hallucinated_entities.add(ae)

        if output_format == "spans":
            hallucinations = []
            for match in re.finditer(r'[^.?!]+[.?!]', answer):
                sentence = match.group().strip()
                if any(ent in sentence for ent in hallucinated_entities):
                    hallucinations.append({
                        "text": sentence,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 1
                    })
            return hallucinations

        elif output_format == "tokens":
            token_outputs = []
            for match in re.finditer(r'\b\w+\b', answer):
                token = match.group()
                is_hallucinated = any(
                    self._fuzzy_match(token, he) > ENTITY_SIMILARITY_THRESHOLD for he in hallucinated_entities
                )
                token_outputs.append({
                    "token": token,
                    "pred": int(is_hallucinated),
                    "prob": 1 if is_hallucinated else 0.01
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
        print("Entered predict method in GLiNER-based class")
        return self._predict(context, answer, output_format)
    
    def predict_prompt_batch(self, prompts, answers, output_format="tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompts and answers.

        :param prompts: List of prompt strings.
        :param answers: List of answer strings.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        return [self._predict(p, a, output_format) for p, a in zip(prompts, answers)]

