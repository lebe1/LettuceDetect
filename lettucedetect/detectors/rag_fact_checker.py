"""Simple RAGFactChecker detector wrapper for lettuceDetect factory pattern."""

from typing import Any, Dict, List

from lettucedetect.detectors.base import BaseDetector


class RAGFactCheckerDetector(BaseDetector):
    """Simple wrapper around RAGFactChecker for lettuceDetect's factory pattern.

    This provides a minimal adapter between lettuceDetect's detector interface
    and our clean RAGFactChecker wrapper.
    """

    def __init__(
        self,
        openai_api_key: str = None,
        model: str = "gpt-4o",
        base_url: str = None,
        temperature: float = 0.0,
        **kwargs,
    ):
        """Initialize the RAGFactChecker detector.

        :param openai_api_key: OpenAI API key
        :param model: OpenAI model to use (default: "gpt-4o")
        :param base_url: Optional base URL for API (e.g., "http://localhost:1234/v1" for local servers)
        :param temperature: Temperature for model sampling (default: 0.0 for deterministic outputs)
        :param kwargs: Additional arguments (ignored for simplicity)
        :return: RAGFactChecker instance
        """
        from lettucedetect.ragfactchecker import RAGFactChecker

        # Use our simple, clean wrapper internally
        self.rag = RAGFactChecker(
            openai_api_key=openai_api_key, model=model, base_url=base_url, temperature=temperature
        )

    def predict(
        self,
        context: List[str],
        answer: str,
        question: str = None,
        output_format: str = "tokens",
        **kwargs,
    ) -> List[Dict[str, Any]] | Dict[str, Any]:
        """Predict hallucinations using RAGFactChecker.

        :param context: List of context documents
        :param answer: Answer text to check for hallucinations
        :param question: Question (optional)
        :param output_format: "tokens", "spans", or "detailed"
        :param kwargs: Additional arguments

        :return: List of predictions in lettuceDetect format, or dict for detailed format
        """
        if output_format not in ["tokens", "spans", "detailed"]:
            raise ValueError(
                f"Invalid output format '{output_format}'. "
                "RAGFactChecker supports 'tokens', 'spans', or 'detailed'"
            )

        # Use our simple wrapper's detection method
        result = self.rag.detect_hallucinations(context, answer, question)

        # Convert to lettuceDetect's expected format
        if output_format == "detailed":
            return {
                "spans": self._convert_to_spans(answer, result),
                "triplets": {
                    "answer": result.get("answer_triplets", []),
                    "context": result.get("context_triplets", []),
                    "hallucinated": result.get("hallucinated_triplets", []),
                },
                "fact_check_results": result.get("fact_check_results", {}),
            }
        elif output_format == "spans":
            return self._convert_to_spans(answer, result)
        else:  # tokens
            return self._convert_to_tokens(answer, result)

    def predict_prompt(
        self, prompt: str, answer: str, output_format: str = "tokens"
    ) -> List[Dict[str, Any]]:
        """Predict using a single prompt string as context."""
        return self.predict([prompt], answer, output_format=output_format)

    def predict_prompt_batch(
        self, prompts: List[str], answers: List[str], output_format: str = "tokens"
    ) -> List[List[Dict[str, Any]]]:
        """Batch prediction using RAGFactChecker's batch processing."""
        if len(prompts) != len(answers):
            raise ValueError("Number of prompts must match number of answers")

        contexts = [[prompt] for prompt in prompts]  # Convert prompts to context lists
        rag_results = self.rag.detect_hallucinations_batch(contexts, answers)

        # Convert each result to lettuceDetect format
        converted_results = []
        for i, (answer, rag_result) in enumerate(zip(answers, rag_results)):
            if output_format == "tokens":
                converted = self._convert_to_tokens(answer, rag_result)
            elif output_format == "spans":
                converted = self._convert_to_spans(answer, rag_result)
            else:
                raise ValueError(f"Unknown output format: {output_format}")
            converted_results.append(converted)

        return converted_results

    def _convert_to_tokens(self, answer: str, rag_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert RAGFactChecker result to token format."""
        tokens = answer.split()
        hallucinated_triplets = rag_result.get("hallucinated_triplets", [])

        token_predictions = []
        for i, token in enumerate(tokens):
            # Simple check if token appears in any hallucinated triplet
            is_hallucinated = any(
                token.lower() in " ".join(triplet).lower() for triplet in hallucinated_triplets
            )

            token_predictions.append(
                {
                    "token": token,
                    "pred": 1 if is_hallucinated else 0,
                    "prob": 0.9 if is_hallucinated else 0.1,
                }
            )

        return token_predictions

    def _convert_to_spans(self, answer: str, rag_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert RAGFactChecker result to span format with improved triplet matching."""
        spans = []
        hallucinated_triplets = rag_result.get("hallucinated_triplets", [])

        for triplet in hallucinated_triplets:
            if len(triplet) < 3:
                continue

            # Try different patterns to find triplet elements in text
            patterns = [
                f"{triplet[0]} {triplet[1]} {triplet[2]}",  # Full triplet phrase
                f"{triplet[0]} {triplet[2]}",  # Subject + object
                triplet[2],  # Object (often contains the hallucination)
                triplet[0],  # Subject
                triplet[1],  # Predicate
            ]

            found_span = False
            for pattern in patterns:
                if not pattern or not pattern.strip():
                    continue

                # Try exact match first, then case-insensitive
                start = answer.find(pattern)
                if start == -1:
                    start = answer.lower().find(pattern.lower())
                    if start != -1:
                        # Get the actual text from the answer with correct case
                        pattern = answer[start : start + len(pattern)]

                if start != -1:
                    spans.append(
                        {
                            "start": start,
                            "end": start + len(pattern),
                            "text": pattern,
                            "confidence": 0.9,
                            "triplet": triplet,  # Include source triplet for transparency
                        }
                    )
                    found_span = True
                    break

            # If no pattern matched, try individual words from the triplet
            if not found_span:
                for element in triplet:
                    if element and element.strip() and len(element) > 3:  # Skip short words
                        start = answer.lower().find(element.lower())
                        if start != -1:
                            actual_text = answer[start : start + len(element)]
                            spans.append(
                                {
                                    "start": start,
                                    "end": start + len(element),
                                    "text": actual_text,
                                    "confidence": 0.7,  # Lower confidence for partial matches
                                    "triplet": triplet,
                                }
                            )
                            break

        # Merge overlapping spans
        return self._merge_overlapping_spans(spans)

    def _merge_overlapping_spans(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlapping spans to avoid duplicates."""
        if not spans:
            return spans

        # Sort spans by start position
        sorted_spans = sorted(spans, key=lambda x: x["start"])
        merged = [sorted_spans[0]]

        for current in sorted_spans[1:]:
            last = merged[-1]

            # Check if spans overlap
            if current["start"] <= last["end"]:
                # Merge spans - extend the end and combine triplets
                merged[-1] = {
                    "start": last["start"],
                    "end": max(last["end"], current["end"]),
                    "text": last["text"],  # Keep original text
                    "confidence": max(last["confidence"], current["confidence"]),
                    "triplet": last.get("triplet", current.get("triplet")),
                }
            else:
                merged.append(current)

        return merged
