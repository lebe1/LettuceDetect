"""Factory function for creating detector instances."""

from __future__ import annotations

from lettucedetect.detectors.base import BaseDetector

__all__ = ["make_detector"]


def make_detector(method: str, **kwargs) -> BaseDetector:
    """Create a detector of the requested type with the given parameters.

    :param method: One of "transformer", "llm", or "rag_fact_checker".
    :param kwargs: Passed to the concrete detector constructor.
    :return: A concrete detector instance.
    :raises ValueError: If method is not supported.
    """
    if method == "transformer":
        from lettucedetect.detectors.transformer import TransformerDetector

        return TransformerDetector(**kwargs)
    elif method == "llm":
        from lettucedetect.detectors.llm import LLMDetector

        return LLMDetector(**kwargs)
    elif method == "rag_fact_checker":
        from lettucedetect.detectors.rag_fact_checker import RAGFactCheckerDetector

        return RAGFactCheckerDetector(**kwargs)
    else:
        raise ValueError(
            f"Unknown detector method: {method}. Use one of: transformer, llm, rag_fact_checker"
        )
