# Benchmarks

## RAGTruth (English)

Evaluated on the [RAGTruth](https://aclanthology.org/2024.acl-long.585/) test set. This benchmark measures how well models detect hallucinations in LLM-generated text across QA, summarization, and data-to-text tasks.

### Example-Level Detection

Binary classification: does the answer contain any hallucination?

| Model | Type | Overall F1 |
|-------|------|-----------|
| GPT-4 | LLM (zero-shot) | 63.4% |
| Luna | Encoder | 65.4% |
| **lettucedetect-base-v1** | **Encoder (149M)** | **76.8%** |
| Llama-2-13B (fine-tuned) | LLM | 78.7% |
| **lettucedetect-large-v1** | **Encoder (395M)** | **79.2%** |
| RAG-HAT (Llama-3-8B) | LLM | 83.9% |

### Span-Level Detection

LettuceDetect achieves state-of-the-art span-level results among models that report this metric, outperforming fine-tuned Llama-2-13B. Span-level evaluation measures how precisely the model can locate the exact hallucinated text within an answer.

## What These Numbers Mean

- **Example F1** — Can the model tell if an answer has *any* hallucination? Higher is better.
- **Span F1** — Can the model point to *exactly which parts* are hallucinated? This is the harder task and where LettuceDetect excels relative to its size.

LettuceDetect models are 50-500x smaller than LLM-based detectors while achieving competitive or better accuracy.

## Citation

```bibtex
@misc{Kovacs:2025,
      title={LettuceDetect: A Hallucination Detection Framework for RAG Applications},
      author={Ádám Kovács and Gábor Recski},
      year={2025},
      eprint={2502.17125},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.17125},
}
```
