from lettucedetect.detectors.base import BaseDetector
from difflib import SequenceMatcher
import re

SEQUENCE_MATCH_THRESHOLD = 0.8

# ==== Fuzzy-based detector ====
class FuzzyBasedDetector(BaseDetector):
    def __init__(self):
        """Initialize the FuzzyBasedDetector."""


    def _predict(self, context: list[str], answer: str, output_format: str = "spans") -> list:
        """
        Perform fuzzy-based hallucination detection on the answer, comparing it to the provided context.

        :param context: A list of context strings to compare against.
        :param answer: The generated answer string to evaluate.
        :param output_format: Output format - either 'spans' (sentence-level) or 'tokens' (word-level).
        :return: A list of hallucination spans or tokens, each with confidence scores.
        """
        # Tokenize context for better comparison.
        context_str = " ".join(context).lower()
        number_hallucinations = self._detect_number_hallucinations(context_str, answer)
        context_sentences = [
            s.strip().lower()
            for ctx in context
            for s in re.split(r'(?<=[.?!])\s+', ctx)
            if s.strip()
        ]

        if output_format == "spans":
            spans = []
            # Iterate over all sentence-like segments in the answer.
            for match in re.finditer(r'[^.?!]+[.?!]', answer):
                sentence = match.group().strip()
                sentence_lower = sentence.lower()
                sentence_numbers = self._extract_numbers(sentence)

                fuzzy_match_score = self._fuzzy_sequence_matcher(sentence_lower, context_str)

                is_hallucinated = sentence_lower not in context_str and fuzzy_match_score < SEQUENCE_MATCH_THRESHOLD


                # Compare with each context sentence and store highest similarity
                # max_similarity = max(
                #     self._fuzzy_sequence_matcher(sentence_lower, ctx_sent)
                #     for ctx_sent in context_sentences
                # ) if context_sentences else 0.0

                has_number_hallucination = any(n in number_hallucinations for n in sentence_numbers)
                # is_hallucinated = max_similarity < SEQUENCE_MATCH_THRESHOLD


                if has_number_hallucination:
                    spans.append({
                        "text": sentence,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 1
                    })
                elif is_hallucinated:
                    spans.append({
                        "text": sentence,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 1 - fuzzy_match_score
                        # "confidence": 1 - max_similarity
                    })
            return spans

        elif output_format == "tokens":
            token_outputs = []
            # Check each word-level token in the answer.
            for match in re.finditer(r'\b\w+\b', answer):
                token = match.group()
                token_lower = token.lower()
                fuzzy_match_score = self._fuzzy_sequence_matcher(token_lower, context_str)

                is_hallucinated = token_lower not in context_str and fuzzy_match_score < SEQUENCE_MATCH_THRESHOLD 
                has_number_hallucination = token_lower in number_hallucinations

                # Check for number hallucination first, since variable is_hallucinated might be true as well in hallucinated number case
                if has_number_hallucination: 
                    token_outputs.append({
                        "token": token,
                        "pred": 1,
                        "prob": 1
                    })
                else:
                    token_outputs.append({
                        "token": token,
                        "pred": int(is_hallucinated),
                        "prob": 1 - fuzzy_match_score if is_hallucinated else 0.01
                    })

                
            return token_outputs

        else:
            raise ValueError("Invalid output_format. Use 'tokens' or 'spans'.")

    def _fuzzy_sequence_matcher(self, s1: str, s2: str) -> float:
        """
        Compute a fuzzy similarity ratio between two strings using SequenceMatcher.

        :param s1: First string (usually the sentence or token).
        :param s2: Second string (typically the full context).
        :return: A float similarity ratio between 0.0 and 1.0.
        """
        return SequenceMatcher(None, s1, s2).ratio()
    
    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: "spans" to return grouped spans.
        """
        return self._predict([prompt], answer, output_format)
    
    def predict_prompt_batch(self, prompts, answers, output_format="tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompts and answers.

        :param prompts: List of prompt strings.
        :param answers: List of answer strings.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        return [self._predict(p, a, output_format) for p, a in zip(prompts, answers)]
    
    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
    ) -> list:
        """Predict hallucination tokens or spans from the provided context, answer, and question.
        This is a useful interface when we don't want to predict a specific prompt, but rather we have a list of contexts, answers, and questions. Useful to interface with RAG systems.

        :param context: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """

        print("Entered predict method in Fuzzy-based class")
        return self._predict(context, answer, output_format)
    
    def _extract_numbers(self, text: str) -> set:
        # This regex pattern checks first for numbers like 1,222.59; second for 0.12; third for 4,12; forth for common digits
        number_pattern = r'\d+\,\d+\.\d+|\d+\.\d+|\d+\,\d+|\d+'
        matches = re.findall(number_pattern, text)
        numbers = set()
        for m in matches:
            numbers.add(m)
        return numbers

    def _detect_number_hallucinations(self, context: str, answer: str) -> set:
        context_numbers = self._extract_numbers(context)
        answer_numbers = self._extract_numbers(answer)

        hallucinated = set()
        for num in answer_numbers:
           if not any((num == ctx_num) for ctx_num in context_numbers):
                hallucinated.add(num)

        return hallucinated