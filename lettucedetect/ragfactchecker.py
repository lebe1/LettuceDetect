"""Simple, clean RAGFactChecker wrapper for lettuceDetect."""

import logging
import os
from typing import Any, Dict, List, Optional


class RAGFactChecker:
    """Simple wrapper around RAGFactChecker with a clean, unified API.

    This provides all RAGFactChecker functionality through one interface:
    - Triplet generation and comparison
    - Hallucination detection
    - Hallucination generation
    - Batch processing
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """Initialize RAGFactChecker.

        :param openai_api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        :param model: OpenAI model to use (default: "gpt-4o"). Options: "gpt-4o", "gpt-4", "gpt-3.5-turbo", etc.
        :param base_url: Optional base URL for API (e.g., "http://localhost:1234/v1" for local servers).
        :param temperature: Temperature for model sampling (default: 0.0 for deterministic outputs).

        :return: RAGFactChecker instance
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass explicitly."
            )

        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _setup_components(self):
        """Initialize RAGFactChecker components."""
        try:
            from rag_fact_checker.data import Config
            from rag_fact_checker.model.fact_checker import LLMFactChecker
            from rag_fact_checker.model.hallucination_data_generator import (
                AnswerBasedHallucinationDataGenerator,
                LLMHallucinationDataGenerator,
            )
            from rag_fact_checker.model.triplet_generator import LLMTripletGenerator

            # Create config with defaults and API key
            self.config = Config()
            self.config.model.llm.api_key = self.openai_api_key
            self.config.model.llm.generator_model = self.model
            self.config.model.llm.temperature = self.temperature
            if self.base_url:
                self.config.model.llm.base_url = self.base_url

            # Initialize components
            self.triplet_generator = LLMTripletGenerator(self.config, self.logger)
            self.fact_checker = LLMFactChecker(self.config, self.logger)
            self.reference_generator = LLMHallucinationDataGenerator(self.config, self.logger)
            self.answer_generator = AnswerBasedHallucinationDataGenerator(self.config, self.logger)

        except ImportError as e:
            raise ImportError(
                "RAGFactChecker not available. Install with: pip install rag-fact-checker"
            ) from e

    # ============ TRIPLET OPERATIONS ============

    def generate_triplets(self, text: str) -> List[List[str]]:
        """Generate triplets from text.

        :param text: Input text

        :return: List of triplets [subject, predicate, object]
            List of triplets [subject, predicate, object]

        """
        result = self.triplet_generator.forward(text)
        return result.triplets

    def compare_triplets(
        self, answer_triplets: List[List[str]], reference_triplets: List[List[str]]
    ) -> Dict[str, Any]:
        """Compare answer triplets against reference triplets.

        :param answer_triplets: Triplets from answer to check
        :param reference_triplets: Reference triplets to compare against

        :return: Dict with fact check results

        """
        result = self.fact_checker.forward(
            answer_triplets=answer_triplets, reference_triplets=[reference_triplets]
        )
        return {"fact_check_results": result.fact_check_prediction_binary, "raw_output": result}

    def analyze_text_pair(self, answer_text: str, reference_text: str) -> Dict[str, Any]:
        """Generate and compare triplets for two texts.

        :param answer_text: Text to analyze
        :param reference_text: Reference text to compare against

        :return: Complete analysis with triplets and comparison results

        """
        answer_triplets = self.generate_triplets(answer_text)
        reference_triplets = self.generate_triplets(reference_text)
        comparison = self.compare_triplets(answer_triplets, reference_triplets)

        return {
            "answer_triplets": answer_triplets,
            "reference_triplets": reference_triplets,
            "comparison": comparison,
        }

    # ============ HALLUCINATION DETECTION ============

    def detect_hallucinations(
        self, context: List[str], answer: str, question: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect hallucinations in answer given context.

        :param context: List of context documents
        :param answer: Answer to check
        :param question: Optional question for context

        :return: Detection results with triplets and fact checking

        """
        # Generate triplets
        answer_triplets = self.generate_triplets(answer)
        context_text = "\n".join(context)
        context_triplets = self.generate_triplets(context_text)

        # Fact check
        comparison = self.compare_triplets(answer_triplets, context_triplets)

        return {
            "answer_triplets": answer_triplets,
            "context_triplets": context_triplets,
            "fact_check_results": comparison["fact_check_results"],
            "hallucinated_triplets": [
                answer_triplets[i]
                for i, fact_is_true in comparison["fact_check_results"].items()
                if not fact_is_true and i < len(answer_triplets)
            ],
        }

    # ============ HALLUCINATION GENERATION ============

    def generate_hallucination_from_context(
        self, context: List[str], question: str
    ) -> Dict[str, Any]:
        """Generate hallucinated content from context and question.

        :param context: List of context documents
        :param question: Question to answer

        :return: Generated hallucinated and non-hallucinated answers

        """
        context_text = "\n".join(context)
        result = self.reference_generator.generate_hlcntn_data(context_text, question)

        return {
            "hallucinated_answer": result.generated_hlcntn_answer,
            "non_hallucinated_answer": result.generated_non_hlcntn_answer,
            "hallucinated_parts": result.hlcntn_part,
        }

    def generate_hallucination_from_answer(
        self,
        correct_answer: str,
        question: str,
        error_types: Optional[List[str]] = None,
        intensity: float = 0.3,
    ) -> Dict[str, Any]:
        """Generate hallucinated version of a correct answer.

        :param correct_answer: The correct answer to modify
        :param question: Original question for context
        :param error_types: Types of errors to inject (factual, temporal, numerical, etc.)
        :param intensity: Error intensity 0.1-1.0

        :return: Generated hallucinated version with error details

        """
        # Convert string error types to ErrorType enums if provided
        error_type_enums = None
        if error_types:
            from rag_fact_checker.model.hallucination_data_generator.answer_based_hallucination_data_generator import (
                ErrorType,
            )

            error_type_enums = []
            for error_type in error_types:
                if hasattr(ErrorType, error_type.upper()):
                    error_type_enums.append(getattr(ErrorType, error_type.upper()))

        result = self.answer_generator.generate_answer_based_hallucination(
            correct_answer=correct_answer,
            question=question,
            error_types=error_type_enums,
            intensity=intensity,
        )

        return {
            "original_answer": result.generated_non_hlcntn_answer,
            "hallucinated_answer": result.generated_hlcntn_answer,
            "hallucinated_parts": result.hlcntn_part,
        }

    # ============ BATCH OPERATIONS ============

    async def generate_hallucination_from_answer_batch_async(
        self,
        correct_answers: List[str],
        questions: List[str],
        error_types: Optional[List[List[str]]] = None,
        intensities: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate hallucinated version of multiple correct answers."""
        error_type_enums_list = None
        if error_types:
            from rag_fact_checker.model.hallucination_data_generator.answer_based_hallucination_data_generator import (
                ErrorType,
            )

            error_type_enums_list = []
            for error_type in error_types:
                error_type_enums = []
                for error_type in error_type:
                    if hasattr(ErrorType, error_type.upper()):
                        error_type_enums.append(getattr(ErrorType, error_type.upper()))
                error_type_enums_list.append(error_type_enums)

        result = await self.answer_generator.generate_answer_based_hallucination_batch_async(
            correct_answers=correct_answers,
            questions=questions,
            error_types_list=error_type_enums_list,
            intensities=intensities,
        )
        return result

    async def generate_hallucination_from_context_batch_async(
        self,
        contexts: List[List[str]],
        questions: List[str],
    ) -> List[Dict[str, Any]]:
        """Generate hallucinated version of multiple correct answers."""
        result = await self.reference_generator.generate_hlcntn_data_batch_async(
            contexts, questions
        )
        return result

    def generate_hallucination_from_answer_batch(
        self,
        correct_answers: List[str],
        questions: List[str],
        error_types: Optional[List[List[str]]] = None,
        intensities: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate hallucinated version of multiple correct answers.

        :param correct_answers: List of correct answers to modify
        :param questions: List of original questions for context
        :param error_types: List of lists of types of errors to inject (factual, temporal, numerical, etc.)
        :param intensities: List of error intensities 0.1-1.0

        :return: List of generated hallucinated versions with error details

        """
        error_type_enums_list = None
        if error_types:
            from rag_fact_checker.model.hallucination_data_generator.answer_based_hallucination_data_generator import (
                ErrorType,
            )

            error_type_enums_list = []
            for error_type in error_types:
                error_type_enums = []
                for error_type in error_type:
                    if hasattr(ErrorType, error_type.upper()):
                        error_type_enums.append(getattr(ErrorType, error_type.upper()))
                error_type_enums_list.append(error_type_enums)

        result = self.answer_generator.generate_answer_based_hallucination_batch(
            correct_answers=correct_answers,
            questions=questions,
            error_types_list=error_type_enums_list,
            intensities=intensities,
        )
        return result

    def generate_hallucination_from_context_batch(
        self,
        contexts: List[List[str]],
        questions: List[str],
    ) -> List[Dict[str, Any]]:
        """Generate hallucinated version of multiple correct answers.

        :param contexts: List of context document lists
        :param questions: List of original questions for context

        :return: List of generated hallucinated versions with error details

        """
        result = self.reference_generator.generate_hlcntn_data_batch(contexts, questions)
        return result

    def generate_triplets_batch(self, texts: List[str]) -> List[List[List[str]]]:
        """Generate triplets for multiple texts.

        :param texts: List of input texts

        :return: List of triplet lists for each text

        """
        batch_result = self.triplet_generator.forward_batch(texts)

        # Create results list with empty lists for failed items
        results = [[] for _ in texts]  # Initialize with empty lists

        # Fill in successful results
        result_index = 0
        for i in range(len(texts)):
            if i not in batch_result.failed_indices:
                if result_index < len(batch_result.results):
                    results[i] = batch_result.results[result_index].triplets
                    result_index += 1

        return results

    def detect_hallucinations_batch(
        self, contexts: List[List[str]], answers: List[str], questions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Detect hallucinations for multiple context-answer pairs.

        :param contexts: List of context document lists
        :param answers: List of answers to check
        :param questions: Optional list of questions

        :return: List of detection results

        """
        results = []
        for i, (context, answer) in enumerate(zip(contexts, answers)):
            question = questions[i] if questions and i < len(questions) else None
            result = self.detect_hallucinations(context, answer, question)
            results.append(result)
        return results
