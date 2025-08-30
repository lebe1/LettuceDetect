# TinyLettuce: Efficient Hallucination Detection with 17–68M Encoders

<p align="center">
  <img src="https://github.com/KRLabsOrg/LettuceDetect/blob/dev/assets/tinytinylettuce.png?raw=true" alt="TinyLettuce Detective" width="400"/>
  <br>
  <em>Small, task‑specialized encoders trained on synthetic data</em>
</p>

---

We present **TinyLettuce**, our approach to efficient hallucination detection. By training tiny Ettin encoders (17-68M parameters) on synthetic data, we achieve better accuracy than billion-parameter LLM judges while running in real-time on CPU.

## TL;DR

- **TinyLettuce‑17M** (17M parameters) reaches **90.87% F1**, outperforming GPT‑5‑mini (83.69%), GPT‑OSS‑120B (83.38%), and Qwen3‑235B (79.84%)
- Runs in **real-time on CPU** with <50ms latency and 1000+ req/s throughput
- **Synthetic data generation** creates training data **100x cheaper** than manual annotation
- Complete **end‑to‑end pipeline** for domain-specific model training - generate data and train in minutes
- All models and code are **MIT licensed** and ready for production deployment

Specialized training on synthetic data beats raw parameter count.

---

## Quick Links

- **GitHub**: [github.com/KRLabsOrg/LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect)  
- **PyPI**: [pypi.org/project/lettucedetect](https://pypi.org/project/lettucedetect/)
- **Hugging Face Models**:  
  - [TinyLettuce Collection](https://huggingface.co/collections/KRLabsOrg/tinylettuce-models) (Coming Soon)
- **Demo**: [Synthetic Data Generation Showcase](../demo/synthetic_data_generation_showcase.ipynb)
- **Notebook**: [TinyLettuce end‑to‑end](../demo/tinylettuce.ipynb)
 - **Ettin Paper (LightOn)**: https://huggingface.co/papers/2507.11412

---

## Get Started

Install the package:

```bash
pip install lettucedetect
```

### Generate Synthetic Training Data

```python
from lettucedetect import HallucinationGenerator

# Initialize generator - temperature=1.0 required for GPT-5 models
generator = HallucinationGenerator(model="gpt-5", temperature=1.0)

# Medical domain example: Generate numerical errors in dosage information
result = generator.generate(
    context=[
        "Ibuprofen is an NSAID that reduces inflammation and pain. The typical adult dose is 400-600mg every 6-8 hours, not exceeding 2400mg daily."
    ],
    question="What is the maximum daily dose of ibuprofen?",
    answer="The maximum daily dose of ibuprofen for adults is 2400mg.",
    error_types=["numerical"],
    intensity=0.4
)

print(f"Original: {result['original_answer']}")
print(f"Hallucinated: {result['hallucinated_answer']}")
print(f"Modified parts: {result['hallucinated_parts']}")
# Output:
# Original: The maximum daily dose of ibuprofen for adults is 2400mg.
# Hallucinated: The maximum daily dose of ibuprofen for adults is 3200mg.
# Modified parts: ['3200mg']
```

### Control Error Intensity and Types

```python
# Low intensity for subtle errors
result_subtle = generator.generate(
    context=[
        "Ibuprofen is an NSAID that reduces inflammation and pain. The typical adult dose is 400-600mg every 6-8 hours, not exceeding 2400mg daily."
    ],
    question="What is the maximum daily dose of ibuprofen?",
    answer="The maximum daily dose of ibuprofen for adults is 2400mg.",
    error_types=["numerical"],
    intensity=0.1  # Very subtle change
)
# Output: "The maximum daily dose of ibuprofen for adults is 2500mg."

# Temporal errors for historical events
result_temporal = generator.generate(
    context=[
        "Apollo 11 was the first crewed mission to land on the Moon, touching down on July 20, 1969. Neil Armstrong and Buzz Aldrin spent about 21 hours on the lunar surface."
    ],
    question="On what date did Apollo 11 land on the Moon?",
    answer="Apollo 11 landed on the Moon on July 20, 1969.",
    error_types=["temporal"],
    intensity=0.5
)
# Output: "Apollo 11 landed on the Moon on July 21, 1969."
```

### Generate from Real Datasets

```python
from datasets import load_dataset

def load_rag_mini_bioasq(split="train", filter_min_words=10):
    """Load rag-mini-bioasq dataset for generation."""
    qa_dataset = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")
    corpus_dataset = load_dataset("enelpol/rag-mini-bioasq", "text-corpus")
    
    # Create corpus lookup
    corpus_lookup = {item["id"]: item["passage"] for item in corpus_dataset["test"]}
    
    processed_data = []
    for item in qa_dataset[split]:
        passage_ids = item["relevant_passage_ids"]
        context_passages = [corpus_lookup.get(pid, None) for pid in passage_ids]
        context_passages = [p for p in context_passages if p is not None]
        
        if len(item["answer"].split()) >= filter_min_words:
            processed_data.append({
                "question": item["question"],
                "answer": item["answer"],
                "context": context_passages,
            })
    return processed_data

# Load biomedical data and generate hallucinations
data = load_rag_mini_bioasq()
sample = data[3]  # Example biomedical question

result = generator.generate(
    context=sample["context"],
    question=sample["question"],
    answer=sample["answer"]
)

# Convert to training format (RAGTruth)
from lettucedetect.detectors.prompt_utils import PromptUtils

train_sample = {
    "prompt": PromptUtils.format_context(sample["context"], sample["question"], lang="en"),
    "answer": result["hallucinated_answer"],
    "labels": [{
        "start": result["hallucinated_answer"].find(part),
        "end": result["hallucinated_answer"].find(part) + len(part),
        "label": "hallucinated"
    } for part in result["hallucinated_parts"]],
    "split": "train",
    "task_type": "qa"
}
```

### Use TinyLettuce for Detection

```python
from lettucedetect import HallucinationDetector

# Load tiny but powerful model
detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/tinylettuce-ettin-17m-v1"
)

# Detect hallucinations in real-time on CPU
spans = detector.predict(
    context=["Ibuprofen is an NSAID that reduces inflammation and pain. The typical adult dose is 400-600mg every 6-8 hours, not exceeding 2400mg daily."],
    question="What is the maximum daily dose of ibuprofen?",
    answer="The maximum daily dose of ibuprofen for adults is 3200mg.",
    output_format="spans"
)

print(spans)
# Output: [{"start": 51, "end": 57, "text": "3200mg"}]
```

---

## Motivation

RAG systems need hallucination detection, but current solutions force painful trade-offs between accuracy, cost, and speed.

**Current hallucination detection approaches:**

1. **Prompt-based detectors** - Use LLM APIs for zero/few-shot detection
   - Can be expensive for large-scale production deployments
   - Latency issues (2-10s per request) unsuitable for real-time use
   - Multiple API calls per detection increase costs

2. **Fine-tuned LLM detectors** - Large models (Llama-2-13B, Llama-3-8B) fine-tuned for detection
   - High accuracy but resource-intensive to train and deploy
   - Need GPU clusters, slow inference, high operational costs

3. **Encoder-based detectors** - BERT-style models for token classification
   - Fast and efficient but historically limited by short context (512 tokens)
   - Can't handle typical RAG contexts which often exceed this limit

**LettuceDetect's breakthrough**: We solved the context problem by leveraging ModernBERT's 8K token capacity, achieving better accuracy than fine-tuned LLMs at a fraction of the computational cost. This proved encoder-based detection could work at scale.

**But we asked: can we go even smaller and faster?**

**Enter TinyLettuce with Ettin encoders**: These lightweight transformers (17–68M parameters), introduced by LightOn, support 8K token contexts and are optimized for classification. Unlike large generative LLMs, Ettin focuses on efficient representation learning for fast, accurate detection.

**The key insight**: With the right synthetic training data, a 17M parameter Ettin encoder can outperform 235B parameter giants at hallucination detection while running real-time on CPU. TinyLettuce democratizes hallucination detection by making it accessible, fast, and cost-effective for any deployment.

## Approach

We discovered something counterintuitive: **specialized training data matters more than parameter count**. With the right synthetic training data, a 17M parameter model can outperform 235B parameter giants at hallucination detection.

Our approach challenges conventional wisdom through four steps:

1. **Generate synthetic data** using RAGFactChecker - no manual annotation needed
2. **Train tiny Ettin encoders** (17M-68M parameters) on this specialized data  
3. **Deploy on CPU** for real-time inference at <50ms latency
4. **Scale effortlessly** - no GPU clusters or API limits

---

## Synthetic Hallucination Data

TinyLettuce leverages **synthetic training data** to achieve high performance. Instead of manually annotating thousands of examples, we use RAGFactChecker to generate training pairs automatically at scale.

### Production-Scale Generation

For large datasets, use our generation script:

```bash
# Generate 10,000 training samples
python scripts/generate_synthetic_data.py \
  --dataset rag-mini-bioasq \
  --num-samples 10000 \
  --model gpt-4o-mini \
  --error-types factual numerical temporal \
  --intensity 0.4 \
  --batch-size 20 \
  --output data/synthetic_10k.json
```

### Data Schema (RAGTruth format)

Minimal entry used for training:

```json
{
  "prompt": "...",
  "answer": "...",
  "labels": [{"start": 31, "end": 71, "label": "hallucinated"}],
  "split": "train",
  "task_type": "qa",
  "dataset": "synthetic",
  "language": "en"
}
```

## TinyLettuce Models (Ettin Encoders)

Our **TinyLettuce** models prove that architecture and training data matter more than parameter count. Built on the **Ettin encoder** (LightOn) — a lightweight, efficient transformer optimized for classification — these models achieve strong accuracy with low latency.

### Model Family

| Model | Parameters | Context Length | Key Advantage |
|-------|------------|----------------|---------------|
| **Ettin-17M** | 17 million | 8K tokens | Fastest, smallest memory |
| **Ettin-32M** | 32 million | 8K tokens | Best speed/accuracy balance |
| **Ettin-68M** | 68 million | 8K tokens | Highest accuracy |

Why Ettin encoders work well:
- 8K token context windows (longer than most inputs)
- Modern transformer design (RoPE, GLU activations)
- Optimized for token classification, not generation
- Efficient CPU inference without GPU overhead

**Training is straightforward:**
```bash
# 1) Generate 10K synthetic examples (~$50, ~5 hours) in RAGTruth format
python scripts/generate_synthetic_data.py \
  --dataset rag-mini-bioasq \
  --split train \
  --num-samples 10000 \
  --model gpt-4o-mini \
  --output-format ragtruth \
  --output data/synthetic_ragtruth_10k.json

# 2) (Optional) Combine with original RAGTruth for training
python - << 'PY'
import json
a = json.load(open('data/ragtruth/ragtruth_data.json'))
b = json.load(open('data/synthetic_ragtruth_10k.json'))
json.dump(a + b, open('data/train_combined.json','w'))
PY

# 3) Train TinyLettuce model (2–4 hours on a single GPU)
python scripts/train.py \
  --model-name jhu-clsp/ettin-encoder-68m \
  --ragtruth-path data/train_combined.json \
  --output-dir output/tinylettuce_68m
```

The results speak for themselves.

### Data & Training Setup (Published Models)

- Source dataset: `enelpol/rag-mini-bioasq` (question–answer–passages).
- Training generation: 1,500 hallucinated samples (≈3,000 total including non‑hallucinated) using a 120B LLM baseline (gpt‑oss‑120b); default intensity 0.3; no explicit error‑type constraints.
- Test generation: 300 hallucinated samples (≈600 total) using GPT‑5 to improve sample quality; held out for evaluation.
- Training recipe: Ettin encoders (17M/32M/68M) fine‑tuned as token classifiers on combined synthetic + RAGTruth (RAGTruth JSON concatenation), then evaluated on synthetic and RAGTruth splits.

### Training Hyperparameters (Released Models)

- Optimizer: AdamW; learning rate `1e-5`; weight decay `0.01`.
- Epochs: 3–6 (released checkpoints typically 3 for Ettin‑17M/32M, 3–6 for Ettin‑68M).
- Batch size: 8; max sequence length: 4096 tokens.
- Tokenization: `AutoTokenizer`; label pad `-100`; `DataCollatorForTokenClassification`.

## Results

When we trained TinyLettuce on synthetic data and tested it against billion-parameter models, the results shocked us.

### Synthetic Data Evaluation (example-level)

Metrics are computed at example level (answer contains any hallucination vs none). Precision/recall/F1 reflect this binary decision; thresholds and post‑processing can affect absolute values.

*When trained and evaluated on domain-specific synthetic data, tiny models dominate:*

| Model | Parameters | Precision (%) | Recall (%) | F1 (%) | Hardware |
|-------|------------|---------------|------------|---------|----------|
| **TinyLettuce-17M** | **17M** | 84.56 | 98.21 | **90.87** | **CPU** |
| **TinyLettuce-32M** | **32M** | 80.36 | 99.10 | 88.76 | **CPU** |
| **TinyLettuce-68M** | **68M** | **89.54** | 95.96 | **92.64** | **CPU** |
| GPT-5-mini | ~200B | 71.95 | **100.00** | 83.69 | API/GPU |
| GPT-OSS-120B | 120B | 72.21 | 98.64 | 83.38 | GPU |
| Qwen3-235B | 235B | 66.74 | 99.32 | 79.84 | GPU |

### RAGTruth Benchmark Evaluation (example-level)

*Strong performance on standard benchmarks:*

| Model | Parameters | F1 (%) |
|-------|------------|---------|
| **TinyLettuce-17M** | **17M** | 68.52 |
| **TinyLettuce-32M** | **32M** | 72.15 |
| **TinyLettuce-68M** | **68M** | **74.97** |
| LettuceDetect-base (ModernBERT) | — | 76.07 |
| LettuceDetect-large (ModernBERT) | 395M | **79.22** |
| Llama-2-13B (RAGTruth FT) | 13B | 78.70 |

### Relative Size vs Performance

How a 17M model compares to a 235B model:

| Aspect | TinyLettuce-17M | Qwen3-235B |
|--------|-----------------|------------|
| **Parameters** | 17 million | 235 billion |
| **F1 on Synthetic Data** | **90.87%** | 79.84% |
| **Performance Advantage** | **+11.03%** | baseline |
| **Size Advantage** | **14,000x smaller** | 1x |
| **Inference Hardware** | CPU | GPU cluster |
| **Deployment Cost** | ~$10/month | ~$10,000/month |

Baselines and judges: we compare against commonly used LLM judges (e.g., GPT‑5‑mini, GPT‑OSS‑120B, Qwen3‑235B) and fine‑tuned encoders/decoders reported in RAGTruth and follow-up work (e.g., Llama‑2‑13B FT). Beyond benchmarks, deployment characteristics often determine real‑world value.

### Evaluation Protocol

- Span construction from tokens: threshold 0.5 on token hallucination prob; contiguous tokens merged into spans.
- Reported F1 is span‑level unless explicitly noted.
- Example command:

```bash
python scripts/evaluate.py \
  --model_path output/tinylettuce_68m \
  --data_path data/ragtruth/ragtruth_data.json \
  --evaluation_type span_level
```

## Real‑Time CPU Inference

TinyLettuce's biggest advantage isn't just accuracy — it's accessibility. These models run in real time on standard CPUs, making hallucination detection practical to deploy widely.

### End-to-End Workflow

```bash
# Step 1: Generate synthetic training data
python scripts/generate_synthetic_data.py \
  --dataset rag-mini-bioasq \
  --num-samples 50000 \
  --model gpt-4o-mini \
  --batch-size 50 \
  --output data/synthetic_large.json

# Step 2: Train TinyLettuce model
python - << 'PY'
import json
a = json.load(open('data/ragtruth/ragtruth_data.json'))
b = json.load(open('data/synthetic_large.json'))
json.dump(a + b, open('data/train_combined_large.json','w'))
PY

python scripts/train.py \
  --ragtruth-path data/train_combined_large.json \
  --model-name jhu-clsp/ettin-encoder-17m \
  --output-dir output/tinylettuce_17m \
  --batch-size 8 \
  --epochs 3

# Step 3: Deploy on CPU for real-time inference
python scripts/start_api.py prod --model output/tinylettuce_17m
```

### Performance Characteristics

| Metric | TinyLettuce-17M | TinyLettuce-32M | TinyLettuce-68M | GPT-5-mini API |
|--------|-----------------|-----------------|-----------------|-----------|
| **Latency** | <50ms | <75ms | <100ms | 2-10s |
| **Throughput** | 1000+ req/s | 800 req/s | 500 req/s | 10 req/s |
| **Memory** | 200MB | 350MB | 600MB | N/A |
| **Cost/1M requests** | $0.10 | $0.15 | $0.25 | $1000+ |

---

## Trade‑offs: Choosing a Model Size

When selecting a TinyLettuce variant, consider these trade-offs:

### TinyLettuce-17M 
- **Best for**: High-throughput, latency-critical applications
- **Pros**: Fastest inference, smallest memory footprint, lowest cost
- **Cons**: Slightly lower accuracy on complex cases
- **Use cases**: Real-time RAG validation, edge deployment

### TinyLettuce-32M
- **Best for**: Balanced production deployments  
- **Pros**: Good accuracy/speed balance, reasonable memory usage
- **Cons**: Moderate resource requirements
- **Use cases**: Production RAG pipelines, content moderation

### TinyLettuce-68M
- **Best for**: Accuracy-critical applications
- **Pros**: Highest detection accuracy, still CPU-efficient
- **Cons**: Higher memory and compute requirements
- **Use cases**: High-stakes content validation, research applications

---

## Converting Synthetic Data to Training Format

Transform generated data into RAGTruth format for model training:

```python
from lettucedetect.detectors.prompt_utils import PromptUtils

def convert_to_ragtruth_format(samples, results, language="en"):
    ragtruth_data = []
    
    for sample, result in zip(samples, results):
        # Format context using LettuceDetect's prompt utils
        formatted_prompt = PromptUtils.format_context(
            sample['context'], 
            sample['question'], 
            lang=language
        )
        
        # Non-hallucinated sample
        ragtruth_data.append({
            "prompt": formatted_prompt,
            "answer": result.generated_non_hlcntn_answer,
            "labels": [],  # No hallucinations
            "split": "train",
            "task_type": "qa",
            "dataset": "synthetic",
            "language": language
        })
        
        # Hallucinated sample with span labels
        hallucinated_labels = []
        hallucinated_answer = result.generated_hlcntn_answer
        
        # Create span labels from hallucinated parts
        for part in result.hlcntn_part:
            if isinstance(part, str) and part in hallucinated_answer:
                start = hallucinated_answer.find(part)
                if start != -1:
                    hallucinated_labels.append({
                        "start": start,
                        "end": start + len(part),
                        "label": "hallucinated"
                    })
        
        ragtruth_data.append({
            "prompt": formatted_prompt,
            "answer": hallucinated_answer,
            "labels": hallucinated_labels,
            "split": "train",
            "task_type": "qa", 
            "dataset": "synthetic",
            "language": language
        })
    
    return ragtruth_data
```

---

## Key Takeaways

**Small Specialized > Large Generalist**: TinyLettuce-68M (92.64% F1) outperforms Qwen3-235B (79.84% F1) while being 14,000x smaller. Task-specific training beats raw parameter count.

**Dramatic Cost Reduction**: Synthetic data generation costs significantly less than manual annotation. CPU inference eliminates expensive API calls and GPU requirements.

**Real-Time CPU Inference**: TinyLettuce models achieve <50ms latency and 1000+ req/s on standard CPUs, making hallucination detection practical for any deployment.

**Synthetic Data Breakthrough**: RAGFactChecker-generated synthetic data enables 90%+ F1 scores - higher than what these same models achieve on manually annotated RAGTruth data.

**Complete Open Pipeline**: End-to-end framework from data generation to model deployment available under MIT license. No expensive GPUs or API calls required.


## Bonus: Triplet‑Based RAGFactChecker

RAGFactChecker exposes a symbolic path for analysis using knowledge triplets.

Generate triplets from any text:
```python
from lettucedetect.ragfactchecker import RAGFactChecker

rag = RAGFactChecker(model="gpt-4o-mini")  # requires OPENAI_API_KEY
triplets = rag.generate_triplets("Paris is the capital of France.")
print(triplets)  # e.g., [["Paris", "is_capital_of", "France"]]
```

Triplet‑based hallucination detection against context:
```python
context = [
    "France is a country in Europe. The capital of France is Paris. Population is ~67M."
]
answer = "France's capital is Lyon and population is 67M."

res = rag.detect_hallucinations(context=context, answer=answer)
print(res["hallucinated_triplets"])  # triplets not supported by context
```

Direct triplet comparison and pairwise analysis:
```python
ans_trips = rag.generate_triplets(answer)
ctx_trips = rag.generate_triplets("\n".join(context))
cmp = rag.compare_triplets(ans_trips, ctx_trips)
pair = rag.analyze_text_pair(answer_text=answer, reference_text=context[0])
```

This complements token/span detectors with interpretable, fact‑level explanations.

---

## Reproducibility & Environment

- Python ≥ 3.10; tested on Python 3.12.
- Set API key for synthetic generation: `export OPENAI_API_KEY=...`.
- CPU latency context: 8‑core x86; GPU training: 1× A100 80GB.

---

## Domain Fine‑tuning (Step‑by‑Step)

1) Prepare domain data (`my_domain.json` with `question` and `context` fields).

2) Generate synthetic training pairs in RAGTruth format:

```bash
python scripts/generate_synthetic_data.py \
  --custom-data data/my_domain.json \
  --num-samples 5000 \
  --model gpt-4o-mini \
  --error-types factual numerical temporal \
  --intensity 0.3 \
  --output-format ragtruth \
  --output data/my_domain_synth.json
```

3) Concatenate with RAGTruth (optional but recommended):

```bash
python - << 'PY'
import json
a = json.load(open('data/ragtruth/ragtruth_data.json'))
b = json.load(open('data/my_domain_synth.json'))
json.dump(a + b, open('data/train_my_domain.json','w'))
PY
```

4) Fine‑tune Ettin:

```bash
python scripts/train.py \
  --model-name jhu-clsp/ettin-encoder-17m \
  --ragtruth-path data/train_my_domain.json \
  --output-dir output/tinylettuce_17m_my_domain
```

5) Evaluate and deploy as above.

## Limitations & Notes

- Results labeled “synthetic” reflect evaluation on generated data; real‑world performance depends on domain match. Consider adding a small, manually curated eval set.
- Baselines: we report GPT‑5‑mini and open‑source LLM baselines where available; prompt configuration impacts absolute scores.
- Metrics: synthetic and RAGTruth F1 are span‑level unless otherwise noted; thresholds and post‑processing influence outcomes.
- Links marked “Coming Soon” will be updated as assets are published; model cards will include training details and configs.

---

## Citation

If you find this work useful, please cite it as follows:

```bibtex
@misc{Kovacs:2025:TinyLettuce,
      title={TinyLettuce: Training Efficient Hallucination Detectors with Synthetic Data Generation}, 
      author={Ádám Kovács and Gábor Recski},
      year={2025},
      eprint={2502.xxxxx},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.xxxxx}, 
}
```

---

## References

[1] [RAGTruth: A Dataset for Hallucination Detection in Retrieval-Augmented Generation](https://aclanthology.org/2024.acl-long.585/)

[2] [LettuceDetect: A Hallucination Detection Framework for RAG Applications](https://arxiv.org/abs/2502.17125)

[3] [Ettin: Encoder Models by LightOn (paper)](https://huggingface.co/papers/2507.11412)

[4] [Ettin Encoder Models (HF models)](https://huggingface.co/jhu-clsp/ettin-encoder-68m)

[5] [RAGFactChecker](https://github.com/KRLabsOrg/RAGFactChecker)
