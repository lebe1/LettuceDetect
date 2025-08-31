"""LettuceDetect integration for Elysia.

This integration provides hallucination detection tools that can be used
directly in Elysia decision trees for automatic quality control of AI responses.
"""

from .tools import detect_hallucinations

__all__ = ["detect_hallucinations"]
