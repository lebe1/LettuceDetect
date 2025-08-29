#!/usr/bin/env python3
"""Clean Streamlit demo for LettuceDetect + LangChain real-time detection.

Run with: streamlit run lettucedetect/integrations/langchain/examples/streamlit_app.py

Requirements:
- pip install streamlit langchain langchain-openai lettucedetect
- export OPENAI_API_KEY=your_key
"""

import os
import time

import streamlit as st
import streamlit.components.v1 as components

# LangChain imports with compatibility handling
try:
    from langchain_openai import ChatOpenAI

    try:
        ChatOpenAI.model_rebuild()
    except Exception:
        pass
except ImportError:
    ChatOpenAI = None

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage

from lettucedetect import HallucinationDetector

# LettuceDetect integration
from lettucedetect.integrations.langchain.callbacks import LettuceStreamingCallback


def create_interactive_text(text: str, spans: list[dict]) -> str:
    """Create clean interactive HTML with highlighting (matching original demo style)."""
    html_text = text

    # Apply highlighting (reverse order to preserve indices)
    for span in sorted(spans, key=lambda x: x.get("start", 0), reverse=True):
        start = span.get("start", 0)
        end = span.get("end", 0)
        confidence = span.get("confidence", 0)

        if 0 <= start < end <= len(text):
            span_text = text[start:end]
            highlighted_span = f'<span class="hallucination" title="Confidence: {confidence:.3f}">{span_text}</span>'
            html_text = html_text[:start] + highlighted_span + html_text[end:]

    return f"""
    <style>
        .container {{
            font-family: Arial, sans-serif;
            font-size: 16px;
            line-height: 1.6;
            padding: 20px;
        }}
        .hallucination {{
            background-color: rgba(255, 99, 71, 0.3);
            padding: 2px;
            border-radius: 3px;
            cursor: help;
        }}
        .hallucination:hover {{
            background-color: rgba(255, 99, 71, 0.5);
        }}
    </style>
    <div class="container">{html_text}</div>
    """


class StreamlitRealtimeHandler(BaseCallbackHandler):
    """Simple handler for real-time streaming with HTML display."""

    def __init__(self, html_placeholder):
        super().__init__()
        self.html_placeholder = html_placeholder
        self.text = ""
        self.spans = []

    def on_llm_start(self, *args, **kwargs):
        self.text = ""
        self.spans = []
        self._update_display()

    def on_chat_model_start(self, *args, **kwargs):
        self.on_llm_start(*args, **kwargs)

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        # Update display with current text and any spans
        self._update_display()
        # sleep for 0.1 seconds
        time.sleep(0.1)

    def update_with_detection(self, spans):
        """Update display with detection results."""
        self.spans = spans
        self._update_display()

    def _update_display(self):
        """Update the HTML display with current text and spans."""
        if not self.text.strip():
            html_content = (
                "<div style='font-style: italic; color: gray;'>Generating response...</div>"
            )
        else:
            html_content = create_interactive_text(self.text, self.spans)

        with self.html_placeholder:
            components.html(html_content, height=max(200, len(self.text) // 4))


def create_prompt(question: str, context: str) -> str:
    """Create prompt from context and question."""
    return f"""Based on the following context, answer the question:

Context: {context}

Question: {question}

Answer based only on the provided context:"""


def main():
    """Main Streamlit application - clean and simple like the original demo."""
    st.set_page_config(page_title="LettuceDetect Real-time Demo")

    # Show lettuce detective image like original
    st.image(
        "https://github.com/KRLabsOrg/LettuceDetect/blob/main/assets/lettuce_detective.png?raw=true",
        width=600,
    )

    st.title("Real-time Hallucination Detection")

    # Check requirements
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY environment variable required")
        st.stop()

    if ChatOpenAI is None:
        st.error("langchain-openai not installed")
        st.stop()

    # Simple form like original demo
    context = st.text_area(
        "Context",
        "Python is a high-level programming language created by Guido van Rossum in 1991. "
        "It is known for its simple, readable syntax and extensive standard library.",
        height=100,
    )

    question = st.text_area(
        "Question",
        "What is Python and who created it?",
        height=100,
    )

    # Initialize components
    @st.cache_resource
    def get_llm():
        return ChatOpenAI(model="gpt-4o-mini", streaming=True)

    @st.cache_resource
    def get_detector():
        model_path = "output/hallucination_detection_ettin_17m"
        if os.path.exists(model_path):
            return HallucinationDetector(method="transformer", model_path=model_path)
        else:
            return HallucinationDetector(method="rag_fact_checker")

    llm = get_llm()
    detector = get_detector()

    # Single response area for HTML display
    html_placeholder = st.empty()

    # Simple detect button like original
    if st.button("Generate with Real-time Detection"):
        if not context.strip() or not question.strip():
            st.warning("Please provide both context and question")
            return

        # State for real-time detection
        final_spans = []

        def handle_detection(result):
            """Handle detection results by passing to output handler."""
            nonlocal final_spans
            spans = result.get("spans", [])

            # Pass detection results to the output handler
            output_handler.update_with_detection(spans)

            if result.get("is_final", False):
                final_spans = spans

        # Create callbacks
        detection_callback = LettuceStreamingCallback(
            method="transformer"
            if os.path.exists("output/hallucination_detection_ettin_17m")
            else "rag_fact_checker",
            model_path="output/hallucination_detection_ettin_17m"
            if os.path.exists("output/hallucination_detection_ettin_17m")
            else None,
            context=[context],
            question=question,
            check_every=10,
            on_detection=handle_detection,
            verbose=False,
        )

        output_handler = StreamlitRealtimeHandler(html_placeholder)
        callbacks = [detection_callback, output_handler]

        # Generate response
        try:
            messages = [HumanMessage(content=create_prompt(question, context))]

            with st.spinner("Generating..."):
                llm.invoke(messages, config={"callbacks": callbacks})

            # Show final status message
            issue_count = len(final_spans)
            if issue_count > 0:
                st.warning(
                    f"⚠️ {issue_count} potential issue{'s' if issue_count > 1 else ''} detected"
                )
            else:
                st.success("✅ Response appears clean")

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
