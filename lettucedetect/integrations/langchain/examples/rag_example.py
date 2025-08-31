#!/usr/bin/env python3
"""Professional LettuceDetect + LangChain RAG example.

Demonstrates automatic hallucination detection in a retrieval-augmented
generation pipeline using clean, production-ready code.

Requirements:
- pip install -r lettucedetect/integrations/langchain/requirements.txt
- export OPENAI_API_KEY=your_key
"""

import os

from langchain.chains import RetrievalQA

# LangChain imports
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings

# LettuceDetect integration
from lettucedetect.integrations.langchain.callbacks import (
    LettuceDetectCallback,
    detect_in_chain,
    stream_with_detection,
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


def example_basic_rag_detection():
    """Basic RAG with post-generation hallucination detection."""
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

        # Use convenience function for simple post-generation detection
        result = detect_in_chain(chain, question, verbose=True)

        print(f"A: {result['answer']}")

        if result["has_issues"]:
            detection = result["detection"]
            print(f"🚨 Issues detected: {detection['issue_count']} spans")
            print(f"Max confidence: {detection['confidence']:.3f}")
        else:
            print("✅ No issues detected")

        print()


def example_rag_streaming_detection():
    """RAG with real-time streaming detection - simplified to show working approach."""
    print("RAG + Real-time Streaming Detection Example")
    print("-" * 40)
    print("Shows structured JSON events during streaming")
    print()

    # Setup RAG chain
    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = text_splitter.create_documents(SAMPLE_DOCUMENTS)
    vectorstore = Chroma.from_documents(docs, embeddings)

    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
    )

    question = "How does Python relate to ocean exploration and marine biology?"
    context = [doc.page_content for doc in vectorstore.similarity_search(question, k=2)]

    print(f"Q: {question}")
    print(f"Context: {context[0][:50]}...")
    print()
    print("Streaming Events:")
    print("-" * 18)

    # Use the working streaming approach
    event_count = 0
    for event in stream_with_detection(chain, {"query": question}, context, check_every=8):
        event_count += 1
        if event["type"] == "token":
            print(event["content"], end="", flush=True)
        elif event["type"] == "detection" and event["has_issues"]:
            print(
                f"\n[Detection {event_count}: {event['issue_count']} issues, confidence: {event['confidence']:.3f}]",
                end="",
                flush=True,
            )

    print("\n")
    print(f"Total events processed: {event_count}")


def example_simple_json_streaming():
    """Simple example showing TRUE JSON streaming - perfect for API developers."""
    print("Simple JSON Streaming Example")
    print("-" * 35)
    print("Shows real-time JSON events - exactly what API developers need!")
    print()

    # Setup simple RAG chain
    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = text_splitter.create_documents(SAMPLE_DOCUMENTS)
    vectorstore = Chroma.from_documents(docs, embeddings)

    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
    )

    question = "How does Python relate to ocean exploration?"
    context = [doc.page_content for doc in vectorstore.similarity_search(question, k=2)]

    print(f"Q: {question}")
    print(f"Context: {context[0][:50]}...")
    print()
    print("JSON Events Stream:")
    print("-" * 18)

    # THIS IS THE MAGIC - Stream JSON events in real-time!
    for event in stream_with_detection(chain, {"query": question}, context, check_every=5):
        # Each event is a JSON-serializable dict
        import json

        print(json.dumps(event))

        # In your API:
        # if event["type"] == "token":
        #     await websocket.send_json(event)
        # elif event["type"] == "detection" and event["has_issues"]:
        #     await websocket.send_json({"alert": "hallucination_detected", "spans": event["spans"]})

    print()
    print("Perfect for:")
    print("  - FastAPI streaming responses")
    print("  - WebSocket real-time chat")
    print("  - Server-sent events (SSE)")
    print("  - Any API that needs live updates")


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
        example_basic_rag_detection()
        print("=" * 60)
        example_simple_json_streaming()  # TRUE JSON streaming!
        print("=" * 60)
        example_rag_streaming_detection()  # Detailed streaming analysis
        print("=" * 60)
        example_with_manual_context()

    except Exception as e:
        print(f"Error: {e}")
        print(
            "Make sure you have: pip install -r lettucedetect/integrations/langchain/requirements.txt"
        )


if __name__ == "__main__":
    main()
