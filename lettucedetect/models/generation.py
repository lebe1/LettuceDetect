"""Simple hallucination generation using RAGFactChecker."""

from typing import Any, Dict, List, Optional

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
        temperature: float = 0.0,
        **kwargs,
    ):
        """Initialize hallucination generator.

        :param method: Method name (kept for compatibility, only "rag_fact_checker" exists)
        :param openai_api_key: OpenAI API key
        :param model: OpenAI model to use (default: "gpt-4o")
        :param base_url: Optional base URL for API (e.g., "http://localhost:1234/v1" for local servers)
        :param temperature: Temperature for model sampling (default: 0.0 for deterministic outputs)
        :param kwargs: Additional arguments (ignored)

        """
        self.rag = RAGFactChecker(
            openai_api_key=openai_api_key, model=model, base_url=base_url, temperature=temperature
        )

    def generate(
        self,
        context: List[str],
        question: str,
        answer: str = None,
        error_types: Optional[List[str]] = None,
        intensity: float = 0.3,
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
            return self.rag.generate_hallucination_from_answer(
                answer, question, error_types, intensity
            )
        else:
            # Context-based generation
            return self.rag.generate_hallucination_from_context(
                context, question, error_types, intensity
            )

    def generate_batch(
        self,
        contexts: List[List[str]],
        questions: List[str],
        answers: List[str] = None,
        error_types: Optional[List[str]] = None,
        intensity: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Generate hallucinated content for multiple inputs.

        :param contexts: List of context lists
        :param questions: List of questions
        :param answers: List of answers (optional)
        :param kwargs: Additional parameters

        :return: List of generation results
        """
        if error_types:
            error_types = [error_types] * len(contexts)
        if intensity:
            intensity = [intensity] * len(contexts)

        if answers:
            return self.rag.generate_hallucination_from_answer_batch(
                answers, questions, error_types, intensity
            )
        else:
            return self.rag.generate_hallucination_from_context_batch(
                contexts, questions, error_types, intensity
            )

    async def generate_batch_async(
        self,
        contexts: List[List[str]],
        questions: List[str],
        answers: List[str] = None,
        error_types: Optional[List[str]] = None,
        intensity: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Generate hallucinated content for multiple inputs.

        :param contexts: List of context lists
        :param questions: List of questions
        :param answers: List of answers (optional)
        :param kwargs: Additional parameters

        :return: List of generation results

        """
        if error_types:
            error_types = [error_types] * len(contexts)
        if intensity:
            intensity = [intensity] * len(contexts)

        if answers:
            return await self.rag.generate_hallucination_from_answer_batch_async(
                answers, questions, error_types, intensity
            )
        else:
            return await self.rag.generate_hallucination_from_context_batch_async(
                contexts, questions, error_types, intensity
            )
