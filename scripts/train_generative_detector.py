#!/usr/bin/env python3
"""Train a generative hallucination span detector via SFT.

Fine-tunes a decoder LLM (e.g. Qwen3.5-2B) to detect hallucinated spans
in code by generating structured JSON output.

This is Approach D: the model reads context + answer and generates:
    {"hallucinated_spans": [{"text": "...", "explanation": "..."}]}
or for clean samples:
    {"hallucinated_spans": []}

Usage:
    # Train with LoRA on a single GPU
    python scripts/train_generative_detector.py \
        --code-data-path data/code_hallucination/code_hallucination_data.json \
        --model-name Qwen/Qwen3.5-2B \
        --output-dir output/generative_detector

    # With custom settings
    python scripts/train_generative_detector.py \
        --code-data-path data/code_hallucination/code_hallucination_data.json \
        --model-name Qwen/Qwen3.5-2B \
        --output-dir output/generative_detector \
        --lora-r 16 \
        --batch-size 2 \
        --epochs 3 \
        --max-length 4096
"""

import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

SYSTEM_PROMPT = (
    "You are a code hallucination detector. Given source code context and a code answer, "
    "identify any hallucinated spans — code that is factually wrong, uses non-existent APIs, "
    "has incorrect logic, or doesn't match the source context.\n\n"
    "Respond with JSON only:\n"
    '{"hallucinated_spans": [{"text": "exact hallucinated text", "explanation": "why it is wrong"}]}\n\n'
    "If the answer is correct, respond with:\n"
    '{"hallucinated_spans": []}'
)


def build_training_pairs(data_path: str) -> list[dict]:
    """Build (input, output) pairs from the code hallucination dataset.

    For hallucinated samples: output lists the hallucinated spans with explanations.
    For clean samples: output is {"hallucinated_spans": []}.
    """
    data = json.loads(Path(data_path).read_text())
    pairs = []

    for item in data:
        prompt = item["prompt"]
        answer = item["answer"]
        labels = item.get("labels", [])
        split = item.get("split", "train")

        user_msg = f"Context:\n{prompt}\n\nAnswer to check:\n{answer}"

        if labels:
            spans = []
            for label in labels:
                start = label["start"]
                end = label["end"]
                text = answer[start:end]
                if text.strip():
                    spans.append(
                        {
                            "text": text,
                            "explanation": f"{label.get('label', 'hallucination')} error in code",
                        }
                    )
            assistant_msg = json.dumps({"hallucinated_spans": spans})
        else:
            assistant_msg = json.dumps({"hallucinated_spans": []})

        pairs.append(
            {
                "system": SYSTEM_PROMPT,
                "user": user_msg,
                "assistant": assistant_msg,
                "split": split,
            }
        )

    return pairs


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning with chat-formatted inputs."""

    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for pair in pairs:
            messages = [
                {"role": "system", "content": pair["system"]},
                {"role": "user", "content": pair["user"]},
                {"role": "assistant", "content": pair["assistant"]},
            ]

            # Tokenize full conversation
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            full_ids = tokenizer(
                full_text, truncation=True, max_length=max_length, return_tensors="pt"
            )

            # Tokenize without assistant response to find where labels start
            messages_no_assistant = [
                {"role": "system", "content": pair["system"]},
                {"role": "user", "content": pair["user"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages_no_assistant, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(
                prompt_text, truncation=True, max_length=max_length, return_tensors="pt"
            )
            prompt_len = prompt_ids["input_ids"].shape[1]

            input_ids = full_ids["input_ids"].squeeze(0)
            attention_mask = full_ids["attention_mask"].squeeze(0)

            # Labels: -100 for prompt tokens (masked), actual ids for assistant tokens
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(ex["input_ids"].shape[0] for ex in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for ex in batch:
        pad_len = max_len - ex["input_ids"].shape[0]
        input_ids.append(torch.cat([ex["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(
            torch.cat([ex["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels.append(torch.cat([ex["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def evaluate(model, dataloader, device):
    """Compute average loss on validation set."""
    model.eval()
    total_loss = 0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_steps += 1

    return total_loss / max(total_steps, 1)


def main():
    parser = argparse.ArgumentParser(description="Train generative hallucination span detector")
    parser.add_argument("--code-data-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output-dir", type=str, default="output/generative_detector")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build training pairs
    print(f"Loading data from {args.code_data_path}")
    pairs = build_training_pairs(args.code_data_path)

    train_pairs = [p for p in pairs if p["split"] == "train"]
    dev_pairs = [p for p in pairs if p["split"] == "dev"]
    test_pairs = [p for p in pairs if p["split"] == "test"]

    print(f"Train: {len(train_pairs)}, Dev: {len(dev_pairs)}, Test (held out): {len(test_pairs)}")

    random.shuffle(train_pairs)

    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Build datasets
    print("Tokenizing datasets...")
    train_dataset = SFTDataset(train_pairs, tokenizer, max_length=args.max_length)
    dev_dataset = SFTDataset(dev_pairs, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    print(f"Train examples: {len(train_dataset)}, Dev examples: {len(dev_dataset)}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dev_loss = float("inf")

    print(f"\nTraining for {args.epochs} epochs ({total_steps} steps)...")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                avg_loss = epoch_loss / (step + 1)
                print(
                    f"  Epoch {epoch + 1} step {step + 1}/{len(train_loader)} loss={avg_loss:.4f}"
                )

        # Evaluate
        train_avg = epoch_loss / len(train_loader)
        dev_loss = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch + 1}: train_loss={train_avg:.4f}, dev_loss={dev_loss:.4f}")

        # Save best
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            model.save_pretrained(output_dir / "best")
            tokenizer.save_pretrained(output_dir / "best")
            print(f"  Saved best model (dev_loss={dev_loss:.4f})")

    # Save final
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    print(f"\nTraining complete. Best dev loss: {best_dev_loss:.4f}")
    print(f"Models saved to {output_dir}")


if __name__ == "__main__":
    main()
