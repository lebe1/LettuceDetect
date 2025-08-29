"""Clean, minimal LangChain callbacks for LettuceDetect integration."""

from typing import Any, Callable, Dict, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.schema.document import Document

from lettucedetect import HallucinationDetector


class LettuceDetectCallback(BaseCallbackHandler):
    """Simple callback for post-generation hallucination detection.

    Automatically detects hallucinations in LLM responses when used with
    retrieval chains or when context is provided manually.
    """

    def __init__(
        self,
        method: str = "rag_fact_checker",
        model_path: Optional[str] = None,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
        verbose: bool = False,
    ):
        """Initialize the callback.

        Args:
            method: Detection method ("transformer", "llm", "rag_fact_checker")
            model_path: Path to model (for transformer method)
            on_result: Optional function to handle detection results
            verbose: Whether to print results

        """
        super().__init__()
        self.detector = HallucinationDetector(method=method, model_path=model_path)
        self.on_result = on_result
        self.verbose = verbose

        # State
        self.context: List[str] = []
        self.question: Optional[str] = None
        self.results: List[Dict[str, Any]] = []

    def set_context(self, context: List[str]) -> None:
        """Manually set context documents."""
        self.context = context

    def set_question(self, question: str) -> None:
        """Manually set the question."""
        self.question = question

    def on_retriever_end(self, documents: List[Document], **kwargs: Any) -> None:
        """Store retrieved context."""
        self.context = [doc.page_content for doc in documents]

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Extract question from chain inputs."""
        for key in ["question", "query", "input"]:
            if key in inputs:
                self.question = inputs[key]
                break

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run hallucination detection on LLM response."""
        if not self.context or not response.generations:
            return

        for generation in response.generations:
            if not generation:
                continue

            text = generation[0].text
            if not text.strip():
                continue

            try:
                spans = self.detector.predict(
                    context=self.context, answer=text, question=self.question, output_format="spans"
                )

                result = {
                    "text": text,
                    "question": self.question,
                    "context": self.context.copy(),
                    "has_issues": len(spans) > 0,
                    "confidence": max([s.get("confidence", 0) for s in spans], default=0),
                    "spans": spans,
                    "issue_count": len(spans),
                }

                self.results.append(result)

                if self.verbose:
                    status = "ISSUES DETECTED" if result["has_issues"] else "CLEAN"
                    print(f"LettuceDetect: {status} (confidence: {result['confidence']:.3f})")

                if self.on_result:
                    self.on_result(result)

            except Exception as e:
                if self.verbose:
                    print(f"LettuceDetect: Detection error: {e}")

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all detection results."""
        return self.results.copy()

    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """Get the most recent detection result."""
        return self.results[-1] if self.results else None

    def has_issues(self) -> bool:
        """Check if any results had issues."""
        return any(r["has_issues"] for r in self.results)

    def reset(self) -> None:
        """Reset callback state."""
        self.context = []
        self.question = None
        self.results = []


class LettuceStreamingCallback(BaseCallbackHandler):
    """Real-time hallucination detection during streaming generation.

    Runs detection periodically during token streaming, enabling real-time
    feedback about potential hallucinations as they're being generated.
    """

    def __init__(
        self,
        method: str = "transformer",
        model_path: Optional[str] = None,
        context: Optional[List[str]] = None,
        question: Optional[str] = None,
        check_every: int = 10,
        on_detection: Optional[Callable[[Dict[str, Any]], None]] = None,
        verbose: bool = False,
    ):
        """Initialize streaming callback.

        Args:
            method: Detection method
            model_path: Path to model (for transformer method)
            context: Context documents for detection
            question: Question being answered
            check_every: Run detection every N tokens
            on_detection: Function called when detection runs
            verbose: Whether to print detection results

        """
        super().__init__()
        self.detector = HallucinationDetector(method=method, model_path=model_path)
        self.context = context or []
        self.question = question
        self.check_every = check_every
        self.on_detection = on_detection
        self.verbose = verbose

        # Streaming state
        self.accumulated_text = ""
        self.token_count = 0
        self.last_checked_length = 0
        self.detection_results = []

    def set_context(self, context: List[str]) -> None:
        """Set context documents."""
        self.context = context

    def set_question(self, question: str) -> None:
        """Set the question being answered."""
        self.question = question

    def on_llm_start(self, *args, **kwargs):
        """Reset state when streaming starts."""
        self.accumulated_text = ""
        self.token_count = 0
        self.last_checked_length = 0
        self.detection_results = []

    def on_chat_model_start(self, *args, **kwargs):
        """Handle chat model start for newer LangChain versions."""
        self.on_llm_start(*args, **kwargs)

    def on_llm_new_token(self, token: str, **kwargs):
        """Process new token and run detection periodically."""
        self.accumulated_text += token
        self.token_count += 1

        # Run detection every N tokens
        if (
            self.token_count >= self.check_every
            and len(self.accumulated_text.strip()) > 20
            and self.context
        ):
            try:
                # Run detection on accumulated text
                spans = self.detector.predict(
                    context=self.context,
                    answer=self.accumulated_text,
                    question=self.question,
                    output_format="spans",
                )

                # Create detection result
                result = {
                    "text": self.accumulated_text,
                    "has_issues": len(spans) > 0,
                    "spans": spans,
                    "confidence": max([s.get("confidence", 0) for s in spans], default=0),
                    "issue_count": len(spans),
                    "token_count": len(self.accumulated_text.split()),
                    "new_text": self.accumulated_text[self.last_checked_length :],
                    "is_incremental": True,
                }

                self.detection_results.append(result)
                self.last_checked_length = len(self.accumulated_text)

                # Call user handler
                if self.on_detection:
                    self.on_detection(result)

                # Verbose output
                if self.verbose and result["has_issues"]:
                    print(f"Real-time detection: {result['issue_count']} issues found")

                # Reset token counter
                self.token_count = 0

            except Exception as e:
                if self.verbose:
                    print(f"Streaming detection error: {e}")

    def on_llm_end(self, response, **kwargs):
        """Run final detection on complete response."""
        if self.accumulated_text and self.context:
            try:
                spans = self.detector.predict(
                    context=self.context,
                    answer=self.accumulated_text,
                    question=self.question,
                    output_format="spans",
                )

                final_result = {
                    "text": self.accumulated_text,
                    "has_issues": len(spans) > 0,
                    "spans": spans,
                    "confidence": max([s.get("confidence", 0) for s in spans], default=0),
                    "issue_count": len(spans),
                    "token_count": len(self.accumulated_text.split()),
                    "is_final": True,
                    "total_checks": len(
                        [r for r in self.detection_results if not r.get("is_final", False)]
                    ),
                }

                self.detection_results.append(final_result)

                if self.on_detection:
                    self.on_detection(final_result)

                if self.verbose:
                    status = "Issues found" if final_result["has_issues"] else "Clean"
                    print(
                        f"Final detection: {status} ({final_result['total_checks']} incremental checks)"
                    )

            except Exception as e:
                if self.verbose:
                    print(f"Final detection error: {e}")

    def on_chat_model_end(self, response, **kwargs):
        """Handle chat model end for newer LangChain versions."""
        self.on_llm_end(response, **kwargs)

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all detection results."""
        return self.detection_results.copy()

    def get_final_result(self) -> Optional[Dict[str, Any]]:
        """Get the final detection result."""
        final_results = [r for r in self.detection_results if r.get("is_final", False)]
        return final_results[-1] if final_results else None

    def has_issues(self) -> bool:
        """Check if any detection found issues."""
        return any(r["has_issues"] for r in self.detection_results)


def detect_in_chain(
    chain, query: str, context: Optional[List[str]] = None, **kwargs
) -> Dict[str, Any]:
    """Convenience function to run a chain with automatic hallucination detection.

    Args:
        chain: LangChain chain to execute
        query: Query/question to ask
        context: Optional context documents (if not using retrieval)
        **kwargs: Additional arguments passed to chain

    Returns:
        Dictionary with chain result and detection info

    """
    callback = LettuceDetectCallback(**kwargs)

    if context:
        callback.set_context(context)
    callback.set_question(query)

    # Run chain with callback
    chain_result = chain.invoke({"query": query}, config={"callbacks": [callback]})
    result = chain_result.get("result", "")

    detection_result = callback.get_last_result()

    return {
        "answer": result,
        "detection": detection_result,
        "has_issues": detection_result["has_issues"] if detection_result else False,
    }
