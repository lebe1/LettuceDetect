#!/usr/bin/env python3
"""Professional LettuceDetect + LangChain RAG example.

Demonstrates automatic hallucination detection in a retrieval-augmented
generation pipeline using clean, production-ready code.

Requirements:
- pip install -r lettucedetect/integrations/langchain/requirements.txt
- export OPENAI_API_KEY=your_key
"""

import os

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage

# LangChain imports
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings

# LettuceDetect integration
from lettucedetect.integrations.langchain.callbacks import (
    LettuceDetectCallback,
    LettuceStreamingCallback,
    detect_in_chain,
)

# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    "The Pacific Ocean is the largest ocean on Earth, covering about 46% of the water surface.",
    "Python was created by Guido van Rossum and first released in 1991.",
    "Machine learning is a subset of artificial intelligence focused on data-driven predictions.",
    "The human brain contains approximately 86 billion neurons.",
    "Photosynthesis converts light energy into chemical energy in plants.",
]


def create_rag_chain():
    """Create a simple RAG chain with vector retrieval."""
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()

    # Split documents and create vector store
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = text_splitter.create_documents(SAMPLE_DOCUMENTS)
    vectorstore = Chroma.from_documents(docs, embeddings)

    # Create retrieval chain
    llm = OpenAI(model="gpt-4o-mini")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=False,
    )

    return chain


def example_basic_usage():
    """Basic usage with automatic detection."""
    print("Basic RAG + Detection Example")
    print("-" * 40)

    chain = create_rag_chain()

    # Questions to test
    questions = [
        "What is the Pacific Ocean?",  # Should be clean
        "Who created Python and when was it invented?",  # Should be clean
        "How does Python relate to ocean exploration?",  # Likely hallucinated
    ]

    for question in questions:
        print(f"Q: {question}")

        # Use convenience function
        result = detect_in_chain(chain, question, verbose=True)

        print(f"A: {result['answer']}")

        if result["has_issues"]:
            detection = result["detection"]
            print(f"ALERT: {detection['issue_count']} issues detected")
            print(f"Confidence: {detection['confidence']:.3f}")
        else:
            print("Status: Clean response")

        print()


def example_streaming_detection():
    """Real streaming example with token-by-token detection - using proven Streamlit pattern."""
    print("Real-time Streaming Detection Example")
    print("-" * 40)
    print("Watch tokens appear one-by-one with real-time detection...")
    print()

    # Manual retrieval setup
    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = text_splitter.create_documents(SAMPLE_DOCUMENTS)
    vectorstore = Chroma.from_documents(docs, embeddings)

    # Create streaming LLM - same as Streamlit demo
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

    question = "How does Python relate to ocean exploration and marine biology?"
    print(f"Q: {question}")

    # Retrieve relevant context
    retrieved_docs = vectorstore.similarity_search(question, k=2)
    context = [doc.page_content for doc in retrieved_docs]

    # Create prompt message
    context_str = "\n".join(context)
    prompt_text = f"""Based on the following context, answer the question:

Context: {context_str}

Question: {question}

Answer based only on the provided context:"""

    # Track detection results and tokens
    detection_events = []
    tokens_received = []

    class ConsoleStreamingHandler(BaseCallbackHandler):
        """Print tokens to console as they arrive."""

        def on_llm_start(self, *args, **kwargs):
            print("A: ", end="", flush=True)

        def on_chat_model_start(self, *args, **kwargs):
            print("A: ", end="", flush=True)

        def on_llm_new_token(self, token: str, **kwargs):
            tokens_received.append(token)
            print(token, end="", flush=True)

    def handle_realtime_detection(result):
        """Handle real-time detection results."""
        if result.get("is_final", False):
            print(f"\n🎯 Final: {result['issue_count']} total issues detected")
        elif result.get("has_issues", False):
            detection_events.append(result)
            print(f"\n🔍 Alert: {result['issue_count']} issues at {result['token_count']} tokens")
            print(
                "A: " + "".join(tokens_received), end="", flush=True
            )  # Continue from where we left off

    # Create callbacks - same pattern as Streamlit demo
    streaming_callback = LettuceStreamingCallback(
        method="transformer"
        if os.path.exists("output/hallucination_detection_ettin_17m")
        else "rag_fact_checker",
        model_path="output/hallucination_detection_ettin_17m"
        if os.path.exists("output/hallucination_detection_ettin_17m")
        else None,
        context=context,
        question=question,
        check_every=8,  # Check every 8 tokens
        on_detection=handle_realtime_detection,
        verbose=False,
    )

    console_handler = ConsoleStreamingHandler()
    callbacks = [streaming_callback, console_handler]

    # Use the exact same pattern as Streamlit demo
    try:
        messages = [HumanMessage(content=prompt_text)]
        llm.invoke(messages, config={"callbacks": callbacks})

        print(f"\n\nSummary: {len(detection_events)} detection events during streaming")

    except Exception as e:
        print(f"Error: {e}")


def example_with_manual_context():
    """Example providing context manually (without retrieval)."""
    print("Manual Context Example")
    print("-" * 40)

    # Simple LLM without retrieval
    llm = OpenAI(model="gpt-4o-mini")

    # Manual context
    context = [
        "Python is a programming language created by Guido van Rossum in 1991.",
        "It is known for its simple syntax and readability.",
    ]

    callback = LettuceDetectCallback(verbose=True)
    callback.set_context(context)
    callback.set_question("What is Python?")

    # Direct LLM call
    response = llm.generate(["What is Python?"], callbacks=[callback])
    answer = response.generations[0][0].text

    print(f"A: {answer}")

    result = callback.get_last_result()
    if result:
        print(f"Detection: {'Issues found' if result['has_issues'] else 'Clean'}")


def main():
    """Run all examples."""
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable required")
        return

    try:
        example_basic_usage()
        print("=" * 60)
        example_streaming_detection()  # Real streaming with token-by-token detection!
        print("=" * 60)
        example_with_manual_context()

    except Exception as e:
        print(f"Error: {e}")
        print(
            "Make sure you have: pip install -r lettucedetect/integrations/langchain/requirements.txt"
        )


if __name__ == "__main__":
    main()
