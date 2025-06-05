from lettucedetect.detectors.base import BaseDetector
import re


class NumberDetector(BaseDetector):
    """Detect hallucinations only if a number appears in the answer not mentioned in the context."""

    def predict(
        self,
        context: list[str],
        answer: str,
        question = None,
        output_format: str = "tokens",
    ) -> list:
        """
        Detect number hallucinations by comparing numbers in the answer with the context.

        :param context: A list of context strings.
        :param answer: The answer string.
        :param output_format: 'tokens' for word-level or 'spans' for sentence-level results.
        :return: List of hallucinated spans or tokens.
        """
        context_text = " ".join(context).lower()
        hallucinated_numbers = self._detect_number_hallucinations(context_text, answer)

        if output_format == "tokens":
            return self._get_token_level_predictions(answer, hallucinated_numbers)
        elif output_format == "spans":
            return self._get_span_level_predictions(answer, hallucinated_numbers)
        else:
            raise ValueError("Invalid output_format. Use 'tokens' or 'spans'.")

    def _extract_numbers(self, text: str) -> set:
        number_pattern = r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+'
        return set(re.findall(number_pattern, text))

    def _detect_number_hallucinations(self, context: str, answer: str) -> set:
        context_numbers = self._extract_numbers(context)
        answer_numbers = self._extract_numbers(answer)
        return {num for num in answer_numbers if num not in context_numbers}

    def _get_token_level_predictions(self, answer: str, hallucinated_numbers: set) -> list:
        results = []
        for match in re.finditer(r'\b\w+\b', answer):
            token = match.group()
            is_hallucinated = token in hallucinated_numbers
            results.append({
                "token": token,
                "pred": int(is_hallucinated),
                "prob": 1.0 if is_hallucinated else 0.0,
            })
        return results

    def _get_span_level_predictions(self, answer: str, hallucinated_numbers: set) -> list:
        spans = []
        for match in re.finditer(r'[^.?!]+[.?!]', answer):
            sentence = match.group().strip()
            sentence_numbers = self._extract_numbers(sentence)
            if any(num in hallucinated_numbers for num in sentence_numbers):
                spans.append({
                    "text": sentence,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 1.0,
                })
        return spans

    def predict_prompt(self, prompt, answer, output_format="tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        return self._predict(prompt, answer, output_format)

    def predict_prompt_batch(self, prompts, answers, output_format="tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompts and answers.

        :param prompts: List of prompt strings.
        :param answers: List of answer strings.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        return [self._predict(p, a, output_format) for p, a in zip(prompts, answers)]