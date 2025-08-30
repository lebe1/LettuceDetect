"""LettuceDetect integration tools for Elysia."""

from typing import List, Optional

from elysia import tool

from lettucedetect import HallucinationDetector


@tool
async def detect_hallucinations(
    context: List[str],
    answer: str,
    question: Optional[str] = None,
):
    """Verify AI-generated answers by comparing them against source context by using detecting hallucinations.

    This tool analyzes whether statements in an answer are supported by the provided context,
    identifying specific spans of text that may be hallucinated or unsupported. It uses
    advanced NLP models to perform token-level analysis and provides detailed feedback
    about problematic content.

    Args:
        context: List of source documents or passages that should support the answer.
                Each string represents a separate context document or paragraph.
                These are the "ground truth" sources the answer should be based on.
        answer: The AI-generated response to analyze for potential hallucinations.
                This is the text that will be checked against the context.
        question: Optional original question that was asked. Providing this improves
                 detection accuracy by understanding what information was requested.

    This tool performs the following analysis:
    1. Tokenizes the answer and compares each segment against the context
    2. Identifies spans that are not supported by any context document
    3. Assigns confidence scores to problematic spans
    4. Returns structured results with exact character positions

    The tool will identify various types of hallucinations:
    - Factual errors (wrong dates, names, numbers)
    - Unsupported claims not present in context
    - Contradictions to the provided information
    - Invented details not mentioned in sources

    Always use this tool when you need to:
        - Verify AI responses against source documents in RAG systems
        - Implement quality control for generated content
        - Build fact-checking pipelines
        - Ensure accuracy in knowledge-based applications
        - Validate information before presenting to users

    Example scenario:
        Context: ["Python was created in 1991 by Guido van Rossum", "It's known for readable syntax"]
        Answer: "Python was created in 1985 by James Gosling and is known for complex syntax"

        This tool would identify:
        - "1985" as hallucinated (should be 1991)
        - "James Gosling" as hallucinated (should be Guido van Rossum)
        - "complex syntax" as hallucinated (context says readable syntax)

    """
    try:
        # Initialize detector with transformer method
        detector = HallucinationDetector(
            method="transformer", model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
        )

        # Perform hallucination detection
        spans = detector.predict(
            context=context, answer=answer, question=question, output_format="spans"
        )

        # Calculate overall metrics
        has_issues = len(spans) > 0
        max_confidence = max([span.get("confidence", 0) for span in spans], default=0)

        # Create structured result
        result = {
            "has_issues": has_issues,
            "confidence": max_confidence,
            "issue_count": len(spans),
            "spans": spans,
        }

        # Yield structured data for the AI agent
        yield result

        # Create human-readable summary
        if has_issues:
            issue_details = []
            for span in spans[:5]:  # Show up to 5 examples
                text = span.get("text", "unknown")
                conf = span.get("confidence", 0)
                start = span.get("start", 0)
                end = span.get("end", 0)
                issue_details.append(f"'{text}' at position {start}-{end} (confidence: {conf:.2f})")

            summary = f"Detected {len(spans)} potential hallucination(s) in the answer. "
            summary += f"Most problematic spans: {', '.join(issue_details)}. "
            summary += (
                "The AI should revise these unsupported claims or provide additional context."
            )
        else:
            summary = "No hallucinations detected. The answer appears to be well-supported by the provided context."

        yield summary

    except Exception as e:
        error_msg = f"Hallucination detection failed: {e!s}"
        yield {"error": True, "message": str(e)}
        yield error_msg
