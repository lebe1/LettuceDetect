#!/usr/bin/env python3
"""Train a hallucination detector on the code hallucination dataset.

Can optionally combine with RAGTruth data for mixed training.

Usage:
    # Train on code hallucination data only
    python scripts/train_code_hallucination.py \
        --code-data-path data/code_hallucination_lettucedetect_v2.json \
        --model-name answerdotai/ModernBERT-base \
        --output-dir output/code_hallucination_detector

    # Train on both code + RAGTruth data
    python scripts/train_code_hallucination.py \
        --code-data-path data/code_hallucination_lettucedetect_v2.json \
        --ragtruth-path data/ragtruth/ragtruth_data.json \
        --model-name answerdotai/ModernBERT-base \
        --output-dir output/code_hallucination_detector
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
    HallucinationDataset,
    HallucinationSample,
)
from lettucedetect.models.trainer import Trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_code_data(path: str) -> list[HallucinationSample]:
    """Load code hallucination dataset."""
    data = json.loads(Path(path).read_text())
    samples = []
    for item in data:
        # Map non-standard dataset values to "ragtruth" for type compat
        dataset = item.get("dataset", "ragtruth")
        if dataset not in ("ragtruth", "ragbench"):
            dataset = "ragtruth"
        language = item.get("language", "en")
        if language not in ("en", "de"):
            language = "en"

        samples.append(
            HallucinationSample(
                prompt=item["prompt"],
                answer=item["answer"],
                labels=item["labels"],
                split=item.get("split", "train"),
                task_type=item.get("task_type", "code_generation"),
                dataset=dataset,
                language=language,
            )
        )
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Train hallucination detector on code data")
    parser.add_argument(
        "--code-data-path", type=str, required=True, help="Path to code hallucination dataset JSON"
    )
    parser.add_argument(
        "--ragtruth-path",
        type=str,
        default=None,
        help="Optional path to RAGTruth data for mixed training",
    )
    parser.add_argument("--model-name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--output-dir", type=str, default="output/code_hallucination_detector")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load code hallucination data
    print(f"Loading code hallucination data from {args.code_data_path}")
    code_samples = load_code_data(args.code_data_path)

    n_clean = sum(1 for s in code_samples if not s.labels)
    n_hall = sum(1 for s in code_samples if s.labels)
    print(f"  Total: {len(code_samples)} (clean: {n_clean}, hallucinated: {n_hall})")

    # Split code data into train/dev
    random.shuffle(code_samples)
    dev_size = int(len(code_samples) * args.dev_ratio)
    train_samples = code_samples[dev_size:]
    dev_samples = code_samples[:dev_size]

    # Optionally add RAGTruth data
    if args.ragtruth_path:
        print(f"\nLoading RAGTruth data from {args.ragtruth_path}")
        ragtruth_data = HallucinationData.from_json(
            json.loads(Path(args.ragtruth_path).read_text())
        )
        ragtruth_train = [s for s in ragtruth_data.samples if s.split == "train"]
        ragtruth_dev = [s for s in ragtruth_data.samples if s.split in ("dev", "test")]

        print(f"  RAGTruth train: {len(ragtruth_train)}, dev: {len(ragtruth_dev)}")
        train_samples.extend(ragtruth_train)
        dev_samples.extend(ragtruth_dev)

    print("\nFinal splits:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Dev:   {len(dev_samples)}")

    # Setup tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)

    train_dataset = HallucinationDataset(train_samples, tokenizer, max_length=args.max_length)
    dev_dataset = HallucinationDataset(dev_samples, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        trust_remote_code=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        test_loader=dev_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=args.output_dir,
    )

    print(f"\nStarting training for {args.epochs} epochs...")
    trainer.train()
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
