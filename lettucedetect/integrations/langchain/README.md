# LettuceDetect + LangChain Integration

Clean, professional hallucination detection for LangChain applications.

## Installation

```bash
pip install lettucedetect
pip install -r lettucedetect/integrations/langchain/requirements.txt
export OPENAI_API_KEY=your_key
```

## Quick Start

```python
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from lettucedetect.integrations.langchain import LettuceDetectCallback, detect_in_chain

# Method 1: Use convenience function
chain = RetrievalQA.from_llm(llm, retriever)
result = detect_in_chain(chain, "Your question")

print(f"Answer: {result['answer']}")
if result['has_issues']:
    print("Potential hallucinations detected")

# Method 2: Use callback directly  
callback = LettuceDetectCallback(verbose=True)
answer = chain.run("Your question", callbacks=[callback])

if callback.has_issues():
    print("Issues found in response")
```

## Real-time Detection Demo

Interactive Streamlit demo showcasing real-time hallucination detection:

```bash
streamlit run lettucedetect/integrations/langchain/examples/streamlit_app.py
```

Features:
- Real-time token-level detection during streaming
- Visual highlighting of potential hallucinations
- Clean, professional interface
- Uses local transformer models for fast inference

## API Reference

### LettuceDetectCallback

Main callback class for automatic hallucination detection.

**Parameters:**
- `method` (str): Detection method ("rag_fact_checker", "transformer", "llm")
- `model_path` (str, optional): Path to model for transformer method
- `on_result` (callable, optional): Function to handle detection results
- `verbose` (bool): Whether to print results

**Methods:**
- `get_results()` - Get all detection results
- `get_last_result()` - Get most recent result  
- `has_issues()` - Check if any issues were detected
- `set_context(context)` - Manually set context documents
- `reset()` - Reset callback state

### LettuceStreamingCallback

Real-time hallucination detection during streaming generation.

**Parameters:**
- `method` (str): Detection method
- `model_path` (str, optional): Path to model for transformer method
- `context` (list): Context documents for detection
- `question` (str): Question being answered
- `check_every` (int): Run detection every N tokens
- `on_detection` (callable): Function called when detection runs
- `verbose` (bool): Whether to print detection results

### detect_in_chain()

Convenience function to run a chain with detection.

**Parameters:**
- `chain` - LangChain chain to execute
- `query` (str) - Question to ask
- `context` (list, optional) - Manual context documents
- `**kwargs` - Additional arguments for LettuceDetectCallback

**Returns:**
Dictionary with `answer`, `detection`, and `has_issues` keys.

## Detection Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `transformer` | Fine-tuned encoder models | High accuracy, local inference |
| `rag_fact_checker` | Triplet-based detection | General purpose, no local models |
| `llm` | LLM-based detection | Flexible, API-based |

## Examples

### Basic RAG Pipeline

```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from lettucedetect.integrations.langchain import LettuceDetectCallback

# Setup RAG chain
embeddings = OpenAIEmbeddings() 
vectorstore = Chroma.from_texts(documents, embeddings)
chain = RetrievalQA.from_llm(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever()
)

# Add detection
callback = LettuceDetectCallback(verbose=True)
result = chain.run("Your question", callbacks=[callback])

# Check results
if callback.has_issues():
    last_result = callback.get_last_result()
    print(f"Found {last_result['issue_count']} issues")
```

### Manual Context

```python
from langchain_openai import OpenAI
from lettucedetect.integrations.langchain import LettuceDetectCallback

llm = OpenAI()
callback = LettuceDetectCallback()

# Set context manually
callback.set_context([
    "Python was created by Guido van Rossum in 1991.",
    "It is known for readable syntax."
])
callback.set_question("What is Python?")

# Generate with detection
response = llm.generate(["What is Python?"], callbacks=[callback])
```

## Production Usage

For production applications:

1. Use the `transformer` method with local models for fastest inference
2. Set `verbose=False` to avoid console output
3. Use `on_result` callback for custom logging/alerts
4. Monitor detection results for system health

```python
import logging
from lettucedetect.integrations.langchain import LettuceDetectCallback

def log_detection(result):
    if result['has_issues']:
        logging.warning(f"Hallucination detected: {result['issue_count']} issues")

callback = LettuceDetectCallback(
    method="transformer",
    model_path="output/hallucination_detection_ettin_17m",
    on_result=log_detection,
    verbose=False
)
```

## Requirements

- Python 3.8+
- LangChain
- LettuceDetect
- OpenAI API key (for LLM-based detection)

For local transformer models:
```bash
pip install transformers torch
```