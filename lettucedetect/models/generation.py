"""Simple hallucination generation using RAGFactChecker."""

from typing import Any, Dict, List

from lettucedetect.ragfactchecker import RAGFactChecker


class HallucinationGenerator:
    """Simple hallucination generator using RAGFactChecker.

    This provides the same interface as before but uses our clean RAGFactChecker wrapper.
    """

    def __init__(
        self,
        method: str = "rag_fact_checker",
        openai_api_key: str = None,
        model: str = "gpt-4o",
        base_url: str = None,
        **kwargs,
    ):
        """Initialize hallucination generator.

        :param method: Method name (kept for compatibility, only "rag_fact_checker" exists)
        :param openai_api_key: OpenAI API key
        :param model: OpenAI model to use (default: "gpt-4o")
        :param base_url: Optional base URL for API (e.g., "http://localhost:1234/v1" for local servers)
        :param kwargs: Additional arguments (ignored)

        """
        self.rag = RAGFactChecker(openai_api_key=openai_api_key, model=model, base_url=base_url)

    def generate(
        self, context: List[str], question: str, answer: str = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate hallucinated content.

        :param context: List of context documents
        :param question: Question to generate answer for
        :param answer: Original answer (optional, for answer-based generation)
        :param kwargs: Additional parameters

        :return: Generation results

        """
        if answer:
            # Answer-based generation
            return self.rag.generate_hallucination_from_answer(answer, question)
        else:
            # Context-based generation
            return self.rag.generate_hallucination_from_context(context, question)

    def generate_batch(
        self, contexts: List[List[str]], questions: List[str], answers: List[str] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate hallucinated content for multiple inputs.

        :param contexts: List of context lists
        :param questions: List of questions
        :param answers: List of answers (optional)
        :param kwargs: Additional parameters

        :return: List of generation results

        """
        results = []
        for i, (context, question) in enumerate(zip(contexts, questions)):
            answer = answers[i] if answers and i < len(answers) else None
            result = self.generate(context, question, answer, **kwargs)
            results.append(result)
        return results
