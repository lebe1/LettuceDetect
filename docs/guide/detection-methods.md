# Detection Methods

LettuceDetect supports three ways to detect hallucinations, each with different trade-offs.

## Transformer (Recommended)

Fine-tuned encoder models that classify each token in the answer as supported or hallucinated. Best balance of speed and accuracy.

```python
detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedetect-large-modernbert-en-v1"
)
```

**How it works:** The model reads the context, question, and answer together. It labels each answer token, then merges consecutive hallucinated tokens into character spans. A single forward pass — fast enough for production use (30-60 samples/sec on GPU).

**When to use:** Production systems, latency-sensitive applications, or when you need precise span locations.

## LLM-based

Uses OpenAI-compatible APIs (GPT-4, Claude, etc.) for hallucination detection. No fine-tuning needed.

```python
detector = HallucinationDetector(method="llm", model_path="gpt-4o-mini")
```

**How it works:** Sends context + question + answer to the LLM with a prompt requesting hallucination spans in a structured format.

**When to use:** Quick prototyping, or when you want the LLM to explain *why* something is hallucinated.

## RAG Fact Checker

Triplet-based fact checking that breaks the answer into structured claims and verifies each one.

```python
detector = HallucinationDetector(method="rag_fact_checker", model_path="gpt-4o-mini")
```

**How it works:** Extracts (subject, predicate, object) claims from the answer, then checks each claim against the context.

**When to use:** When you want claim-level granularity and structured verification results.
