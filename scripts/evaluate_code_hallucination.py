#!/usr/bin/env python3
"""Evaluate LLM baseline on the code hallucination dataset.

Uses LettuceDetect's existing LLMDetector and evaluator infrastructure.
Supports Groq API with any OpenAI-compatible model.

Usage:
    # With Groq + Kimi
    OPENAI_API_KEY=gsk_... OPENAI_API_BASE=https://api.groq.com/openai/v1 \
        python scripts/evaluate_code_hallucination.py \
        --model moonshotai/kimi-k2-instruct-0905 \
        --data_path data/code_hallucination_lettucedetect_v2.json \
        --evaluation_type example_level
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from string import Template

from openai import OpenAI
from sklearn.metrics import auc, classification_report, precision_recall_fscore_support, roc_curve
from tqdm import tqdm

from lettucedetect.datasets.hallucination_dataset import HallucinationSample

# Simpler prompt for code hallucination detection (no structured output dependency)
CODE_HALLUCINATION_PROMPT = Template("""<task>
You are an expert code reviewer who must identify hallucinated code spans.

A "code hallucination" is any part of the generated code that:
(a) Uses APIs, methods, functions, classes, or parameters that do NOT exist in the codebase context provided
(b) Contradicts the documentation or codebase shown in the source
(c) Does something semantically different from what the user requested

## Instructions
1. Read the generated code inside <answer>...</answer>.
2. Compare it against the source context in <source>...</source>.
3. Identify any code spans that are hallucinated.
4. Return a JSON object with this exact format (no markdown, no code blocks):
   {"hallucination_list": ["exact_span_1", "exact_span_2", ...]}
   If no hallucinations found, return: {"hallucination_list": []}

IMPORTANT: Each item in hallucination_list must be an EXACT substring from the answer.
</task>

<source>
${context}
</source>

<answer>
${answer}
</answer>""")


def predict_with_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    answer: str,
    temperature: float = 0.0,
) -> list[dict]:
    """Predict hallucination spans using an LLM."""
    llm_prompt = CODE_HALLUCINATION_PROMPT.substitute(context=prompt, answer=answer)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in detecting hallucinations in LLM-generated code.",
                },
                {"role": "user", "content": llm_prompt},
            ],
            temperature=temperature,
            max_tokens=1000,
        )
        raw = resp.choices[0].message.content.strip()

        # Parse JSON from response
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            payload = json.loads(json_match.group())
            spans = []
            for sub in payload.get("hallucination_list", []):
                if not sub:
                    continue
                match = re.search(re.escape(sub), answer)
                if match:
                    spans.append({"start": match.start(), "end": match.end(), "text": sub})
            return spans
        return []
    except Exception as e:
        print(f"  Error: {e}")
        return []


def evaluate_example_level(
    samples: list[HallucinationSample],
    client: OpenAI,
    model: str,
    temperature: float = 0.0,
) -> dict:
    """Evaluate at example level — binary: does this sample contain hallucinations?"""
    example_preds = []
    example_labels = []

    for sample in tqdm(samples, desc="Evaluating (example level)"):
        predicted_spans = predict_with_llm(client, model, sample.prompt, sample.answer, temperature)
        true_label = 1 if sample.labels else 0
        pred_label = 1 if predicted_spans else 0
        example_labels.append(true_label)
        example_preds.append(pred_label)
        time.sleep(0.5)  # Rate limit

    precision, recall, f1, _ = precision_recall_fscore_support(
        example_labels, example_preds, labels=[0, 1], average=None, zero_division=0
    )

    fpr, tpr, _ = roc_curve(example_labels, example_preds)
    auroc = auc(fpr, tpr)

    results = {
        "supported": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
        },
        "hallucinated": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
        },
        "auroc": auroc,
    }

    report = classification_report(
        example_labels,
        example_preds,
        target_names=["Supported", "Hallucinated"],
        digits=4,
        zero_division=0,
    )
    print("\nExample-Level Classification Report:")
    print(report)
    print(f"AUROC: {auroc:.4f}")

    return results


def evaluate_char_level(
    samples: list[HallucinationSample],
    client: OpenAI,
    model: str,
    temperature: float = 0.0,
) -> dict:
    """Evaluate at character level — overlap between predicted and gold spans."""
    total_overlap = 0
    total_predicted = 0
    total_gold = 0

    for sample in tqdm(samples, desc="Evaluating (char level)"):
        predicted_spans = predict_with_llm(client, model, sample.prompt, sample.answer, temperature)
        gold_spans = sample.labels

        total_predicted += sum(p["end"] - p["start"] for p in predicted_spans)
        total_gold += sum(g["end"] - g["start"] for g in gold_spans)

        for pred in predicted_spans:
            for gold in gold_spans:
                overlap_start = max(pred["start"], gold["start"])
                overlap_end = min(pred["end"], gold["end"])
                if overlap_end > overlap_start:
                    total_overlap += overlap_end - overlap_start

        time.sleep(0.5)  # Rate limit

    precision = total_overlap / total_predicted if total_predicted > 0 else 0
    recall = total_overlap / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\nCharacter-Level Results:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM baseline on code hallucination dataset"
    )
    parser.add_argument("--model", type=str, default="moonshotai/kimi-k2-instruct-0905")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--evaluation_type",
        type=str,
        default="example_level",
        choices=["example_level", "char_level", "both"],
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of test samples (for quick testing)",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.3, help="Fraction of data to use as test set"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load data
    data_path = Path(args.data_path)
    raw_data = json.loads(data_path.read_text())

    # Convert to HallucinationSample objects
    samples = []
    for item in raw_data:
        # Handle the dataset literal type - use "ragtruth" as fallback
        dataset = item.get("dataset", "ragtruth")
        if dataset not in ("ragtruth", "ragbench"):
            dataset = "ragtruth"  # Fallback for compatibility
        language = item.get("language", "en")
        if language not in ("en", "de"):
            language = "en"

        samples.append(
            HallucinationSample(
                prompt=item["prompt"],
                answer=item["answer"],
                labels=item["labels"],
                split=item.get("split", "test"),
                task_type=item.get("task_type", "code_generation"),
                dataset=dataset,
                language=language,
            )
        )

    # Split into test set
    import random

    random.seed(args.seed)
    random.shuffle(samples)

    test_size = int(len(samples) * args.test_ratio)
    test_samples = samples[:test_size]

    if args.max_samples:
        test_samples = test_samples[: args.max_samples]

    n_positive = sum(1 for s in test_samples if s.labels)
    n_negative = sum(1 for s in test_samples if not s.labels)

    print(f"Dataset: {data_path}")
    print(f"Total samples: {len(samples)}")
    print(f"Test samples: {len(test_samples)} (positive: {n_positive}, negative: {n_negative})")
    print(f"Model: {args.model}")
    print(f"API base: {os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')}")

    # Setup client
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )

    # Run evaluation
    results = {}

    if args.evaluation_type in ("example_level", "both"):
        print("\n" + "=" * 60)
        print("EXAMPLE-LEVEL EVALUATION")
        print("=" * 60)
        results["example_level"] = evaluate_example_level(test_samples, client, args.model)

    if args.evaluation_type in ("char_level", "both"):
        print("\n" + "=" * 60)
        print("CHARACTER-LEVEL EVALUATION")
        print("=" * 60)
        results["char_level"] = evaluate_char_level(test_samples, client, args.model)

    # Save results
    output_path = data_path.parent / f"eval_results_{args.model.replace('/', '_')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
