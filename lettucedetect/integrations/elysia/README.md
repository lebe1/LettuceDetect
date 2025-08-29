# LettuceDetect + Elysia Integration

Automatic hallucination detection for Elysia AI decision trees.

## Installation

```bash
pip install lettucedetect elysia-ai
```

## Usage

```python
from elysia import Tree
from lettucedetect.integrations.elysia import detect_hallucinations

# Create tree with hallucination detection
tree = Tree()
tree.add_tool(detect_hallucinations)

# The AI can now automatically validate responses
response = tree("""
Context: Python was created by Guido van Rossum in 1991.
Question: When was Python created?
Please answer and verify your response for accuracy.
""")
```

## What It Does

The `detect_hallucinations` tool automatically:
- ✅ Analyzes AI responses against provided context
- ✅ Identifies unsupported claims and factual errors  
- ✅ Provides confidence scores and exact text spans
- ✅ Guides the AI to self-correct when needed

## Tool Details

**detect_hallucinations**: Main hallucination detection tool
- Compares generated answers against source context
- Returns structured data about problematic spans
- Supports multiple detection methods (transformer, LLM, fact-checker)