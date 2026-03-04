# Architecture Research: Detection Models for Code Hallucination

Research notes on model architectures for training on the code hallucination dataset. We compare four approaches ranging from fast encoder-based classifiers to generative span detectors.

## Approach A: Token Classification (Encoder)

**Architecture:** ModernBERT/EuroBERT + linear classification head

The current LettuceDetect approach. Each answer token gets a binary label (0=supported, 1=hallucinated). Consecutive hallucinated tokens are merged into spans at inference.

```
Input:  [CLS] context [SEP] question [SEP] answer [SEP]
Output: [-100, -100, ..., 0, 0, 1, 1, 1, 0, 0, ...]
                              ^^^^^^^^^ hallucinated span
```

| Property | Value |
|----------|-------|
| **Models** | ModernBERT-base (149M), ModernBERT-large (395M), EuroBERT (210M-2.1B) |
| **Context** | 8K tokens |
| **Inference** | Single forward pass, 30-60 samples/sec on A100 |
| **Training** | Standard token classification, CrossEntropyLoss |
| **Validated by** | LettuceDetect (79.2% F1), HaluGate (vLLM), PsiloQA (EMNLP 2025) |

**Strengths:** Fast, simple, production-ready. Handles long contiguous spans well.
**Weaknesses:** No code-specific pretraining. Cannot explain *why* something is hallucinated.

---

## Approach B: Token Classification (Decoder LLM)

**Architecture:** Qwen3.5-2B + bidirectional attention (LLM2Vec) + linear head

Use a decoder LLM pretrained on massive code corpora, convert to bidirectional encoder via [LLM2Vec](https://arxiv.org/abs/2404.05961), then add a token classification head.

```
Step 1: Load Qwen3.5-2B base (2B params, code-heavy pretraining)
Step 2: Enable bidirectional attention (remove causal mask)
Step 3: Short MNTP adaptation (masked next token prediction with LoRA)
Step 4: Add linear head (hidden_dim=2048 → 2 classes)
Step 5: Fine-tune on code hallucination dataset with LoRA
```

| Property | Value |
|----------|-------|
| **Model** | Qwen3.5-2B (2B params) |
| **Context** | 262K native (practically limited by GPU memory) |
| **Inference** | Single forward pass, ~5-15 samples/sec |
| **VRAM** | ~5-8GB in bf16 |
| **Reference** | [Looking Right is Sometimes Right (ACL 2024)](https://arxiv.org/abs/2401.14556) — 0.947 F1 on NER with mask removal |

**Strengths:** Deep code understanding from pretraining. Bidirectional attention after conversion.
**Weaknesses:** 5x larger than ModernBERT. Requires LLM2Vec conversion step. Novel (unvalidated for hallucination detection).

**Key insight:** The [ACL 2024 paper](https://arxiv.org/abs/2401.14556) showed decoder LLMs with causal mask removal reach 0.947 F1 on NER, significantly above RoBERTa-large (0.900). The gains come from combining rich pretrained representations with bidirectional context.

---

## Approach C: Chunk Verification (Reranker-style)

**Architecture:** Qwen3.5-2B or Qwen3-0.6B, reranker-style yes/no scoring

Inspired by [Qwen3-Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B). Split the answer into chunks (lines, statements), then ask the model for each chunk: "Is this code correct given the context?"

```
Input:  "Given this source code, is this line correct? yes/no"
Output: P(yes) = 0.12  →  hallucinated
        P(yes) = 0.95  →  supported
```

No architectural modifications. Uses the LLM's native next-token prediction to classify.

| Property | Value |
|----------|-------|
| **Models** | Qwen3-0.6B (tiny, fast) or Qwen3.5-2B |
| **Inference** | N forward passes per sample (one per chunk) |
| **Training** | Standard SFT with yes/no labels |
| **Reference** | [MiniCheck (EMNLP 2024)](https://arxiv.org/abs/2404.10774) — GPT-4-level at 400x lower cost |

**Strengths:** No architecture changes. Uses LLM code reasoning directly. Can work with tiny models.
**Weaknesses:** Slowest inference (N passes per sample). Chunk boundary sensitivity. No sub-chunk granularity.

---

## Approach D: Generative Span Detection

**Architecture:** Qwen3.5-2B, standard SFT, generates JSON with hallucinated spans

The model directly outputs which spans are hallucinated and why. This is the reverse of the hallucination injection process.

```
Input:  "Given the source code and answer, identify hallucinated spans."
Output: {
  "hallucinated_spans": [
    {"text": "response.json_decode()", "explanation": "method is json(), not json_decode()"}
  ]
}
```

| Property | Value |
|----------|-------|
| **Models** | Qwen3.5-2B or larger |
| **Inference** | Single generation (autoregressive, slower than forward pass) |
| **Training** | Standard SFT with LoRA |
| **SOTA** | [RL4HS (Oct 2025)](https://arxiv.org/abs/2510.02173) — 58.3 F1 on RAGTruth, beats GPT-5 (42.2) and o3 (51.2) |

**Strengths:**

- No architecture changes — pure text generation
- Free explanations alongside span detection
- Naturally handles variable span counts
- Can leverage the LLM's code knowledge ("this API doesn't exist")
- Training data format already matches (reverse of injection pipeline)
- Current SOTA approach (RL4HS)

**Weaknesses:** Autoregressive generation is slower. Risk of hallucinating in the detector itself. String matching needed to map spans back to character offsets.

**RL enhancement:** [RL4HS](https://arxiv.org/abs/2510.02173) shows that adding reinforcement learning (GRPO with span-level rewards) on top of SFT dramatically improves performance. SFT alone is a strong baseline; RL pushes it to SOTA.

---

## Comparison

| | A. Encoder token | B. LLM token | C. Chunk verifier | D. Generative span |
|---|---|---|---|---|
| **Base model** | ModernBERT-large | Qwen3.5-2B | Qwen3-0.6B | Qwen3.5-2B |
| **Parameters** | 395M | 2B | 0.6B | 2B |
| **Architecture mods** | None | Mask removal | None | None |
| **Inference speed** | Fastest | Medium | Slowest | Medium-slow |
| **Explainable** | No | No | No | Yes |
| **Code understanding** | Limited | Deep | Deep | Deep |
| **Training complexity** | Simple | LLM2Vec + LoRA | Simple SFT | Simple SFT |
| **SOTA reference** | LettuceDetect, HaluGate | ACL 2024 paper | MiniCheck | RL4HS |

## Recommended Experiments

1. **A vs D** — Token classification (ModernBERT) vs generative span detection (Qwen3.5-2B). The core comparison: fast encoder vs reasoning LLM, both trained on the same dataset.

2. **A vs B** — Does code pretraining help token classification? Same task, different backbone.

3. **D with RL** — If SFT results are promising, add GRPO with span-overlap rewards (following RL4HS).

## Key References

- [LettuceDetect (arXiv:2502.17125)](https://arxiv.org/abs/2502.17125) — Encoder token classification baseline
- [HaluGate (vLLM, Dec 2025)](https://blog.vllm.ai/2025/12/14/halugate.html) — Production ModernBERT + NLI pipeline
- [RL4HS (arXiv:2510.02173)](https://arxiv.org/abs/2510.02173) — SOTA generative span detection with RL
- [FAVA (COLM 2024)](https://arxiv.org/abs/2401.06855) — Generative hallucination editing
- [PsiloQA (EMNLP 2025)](https://arxiv.org/abs/2510.04849) — Multilingual encoder-based span detection
- [Looking Right is Sometimes Right (ACL 2024)](https://arxiv.org/abs/2401.14556) — Decoder LLMs for token classification
- [LLM2Vec (2024)](https://arxiv.org/abs/2404.05961) — Converting decoders to bidirectional encoders
- [MiniCheck (EMNLP 2024)](https://arxiv.org/abs/2404.10774) — Sentence-level fact checking
- [Qwen3-Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) — LLM-based yes/no classification
- [CodeMirage (2024)](https://arxiv.org/abs/2408.08333) — Code hallucination taxonomy (snippet-level only)
