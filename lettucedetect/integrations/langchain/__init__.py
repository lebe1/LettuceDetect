"""LangChain integration for LettuceDetect hallucination detection.

This module provides a clean, minimal callback for integrating LettuceDetect
with LangChain applications. The callback automatically detects hallucinations
in LLM responses when used with retrieval chains.

Example usage:

    from integrations.langchain import LettuceDetectCallback, detect_in_chain
    from langchain.chains import RetrievalQA

    # Basic usage
    callback = LettuceDetectCallback(verbose=True)
    result = chain.run("Your question", callbacks=[callback])

    if callback.has_issues():
        print("Potential hallucinations detected")

    # Or use convenience function
    result = detect_in_chain(chain, "Your question")
    print(f"Answer: {result['answer']}")
    print(f"Issues: {result['has_issues']}")
"""

from .callbacks import (
    LettuceDetectCallback,
    LettuceStreamingCallback,
    detect_in_chain,
    stream_with_detection,
)

__all__ = [
    "LettuceDetectCallback",
    "LettuceStreamingCallback",
    "detect_in_chain",
    "stream_with_detection",
]

__version__ = "1.0.0"
