# Training Your Own Model

LettuceDetect models are standard HuggingFace token classifiers. You can fine-tune them on your own data or on the provided RAGTruth dataset.

## How It Works

The model reads a concatenated input — context, question, and answer — and classifies each answer token as either **supported** (0) or **hallucinated** (1). Consecutive hallucinated tokens are merged into spans at inference time.

```
Input:  [CLS] context [SEP] question [SEP] answer [SEP]
Labels: [-100, -100, ...,  -100,  ...,  0, 0, 1, 1, 1, 0, ...]
                                         ^^^^^^^^^^^^^^^^^^^^^^^^
                                         only answer tokens are labeled
```

- `-100` = ignored by the loss function (context + question tokens)
- `0` = supported (answer token backed by context)
- `1` = hallucinated (answer token not supported by context)

The best checkpoint is saved based on validation F1 for the hallucinated class.

## Train on RAGTruth

[RAGTruth](https://aclanthology.org/2024.acl-long.585/) is a text hallucination dataset covering QA, summarization, and data-to-text tasks.

```bash
# Preprocess the raw data first
python lettucedetect/preprocess/preprocess_ragtruth.py \
    --input_dir data/ragtruth --output_dir data/ragtruth

# Train
python scripts/train.py \
    --ragtruth-path data/ragtruth/ragtruth_data.json \
    --model-name answerdotai/ModernBERT-base \
    --output-dir output/hallucination_detector \
    --batch-size 4 \
    --epochs 6 \
    --learning-rate 1e-5
```

## Train on Your Own Data

Your data must follow the `HallucinationSample` format — a JSON list where each item has:

```json
{
  "prompt": "Your context text here...",
  "answer": "The LLM-generated answer to check...",
  "labels": [{"start": 45, "end": 72, "label": "hallucination"}],
  "split": "train",
  "task_type": "qa",
  "dataset": "ragtruth",
  "language": "en"
}
```

- `labels` contains character-level spans within the `answer` marking hallucinated regions. Empty list `[]` for clean samples.
- `split` should be `"train"`, `"dev"`, or `"test"`.

Then train the same way:

```bash
python scripts/train.py \
    --ragtruth-path path/to/your_data.json \
    --model-name answerdotai/ModernBERT-base \
    --output-dir output/my_detector \
    --batch-size 4 \
    --epochs 6
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-name` | `answerdotai/ModernBERT-base` | HuggingFace model to fine-tune |
| `--batch-size` | 4 | Training batch size |
| `--epochs` | 6 | Number of training epochs |
| `--learning-rate` | 1e-5 | Learning rate |
| `--max-length` | 4096 | Maximum input length in tokens |
| `--seed` | 42 | Random seed for reproducibility |

## Recommended Base Models

| Model | Parameters | Max Tokens | Best For |
|-------|-----------|------------|----------|
| `answerdotai/ModernBERT-base` | 149M | 8K | Fast training, good baseline |
| `answerdotai/ModernBERT-large` | 395M | 8K | Best English performance |
| `EuroBERT/EuroBERT-210m` | 210M | 8K | Multilingual (7 languages) |
| `EuroBERT/EuroBERT-610m` | 610M | 8K | Best multilingual performance |

## Evaluate

```bash
# Example-level: does the answer contain any hallucination?
python scripts/evaluate.py \
    --model_path output/hallucination_detector \
    --data_path data/ragtruth/ragtruth_data.json \
    --evaluation_type example_level

# Span-level: how well do predicted spans match gold spans?
python scripts/evaluate.py \
    --model_path output/hallucination_detector \
    --data_path data/ragtruth/ragtruth_data.json \
    --evaluation_type span_level
```

## Code Hallucination

For training on code hallucination data (from SWE-bench), see the [Code Hallucination Dataset](../code-hallucination/index.md) section which covers generating the dataset and training both encoder and generative models.
