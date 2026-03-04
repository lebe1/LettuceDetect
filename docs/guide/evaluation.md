# Evaluation

LettuceDetect supports three levels of evaluation, from coarse to fine-grained.

## Example-Level

The simplest check: does the answer contain **any** hallucination? This is a binary yes/no classification per answer.

```bash
python scripts/evaluate.py \
    --model_path output/hallucination_detector \
    --data_path data/ragtruth/ragtruth_data.json \
    --evaluation_type example_level
```

## Span-Level (Character IoU)

Measures how well predicted hallucination spans overlap with the gold annotations at the character level. This is the most demanding metric — the model must identify not just *that* something is wrong, but *exactly where*.

```bash
python scripts/evaluate.py \
    --model_path output/hallucination_detector \
    --data_path data/ragtruth/ragtruth_data.json \
    --evaluation_type span_level
```

## Token-Level

Per-token precision, recall, and F1 for the hallucinated class. This is what the model is directly optimized for during training.

## Metrics Explained

- **Precision**: Of everything the model flagged as hallucinated, how much was actually hallucinated?
- **Recall**: Of all the actual hallucinations, how many did the model catch?
- **F1**: The balance between precision and recall (harmonic mean). This is the primary metric.
- **AUROC**: Area under the ROC curve — measures how well the model separates hallucinated from supported tokens across all confidence thresholds.

## LLM Baselines

You can also evaluate LLM-based detectors for comparison:

```bash
python scripts/evaluate_llm.py \
    --model "gpt-4o-mini" \
    --data_path data/ragtruth/ragtruth_data.json \
    --evaluation_type example_level
```
