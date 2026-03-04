import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from hallucination_dataset import HallucinationData, HallucinationSample
from tqdm import tqdm
from transformers import AutoTokenizer


def analyze_lengths(
    samples: List[HallucinationSample],
    tokenizer: AutoTokenizer,
    max_length: int = 4096,
) -> Dict[str, Any]:
    """Analyze the token lengths of samples and identify outliers.

    Args:
        samples: List of HallucinationSample objects
        tokenizer: Tokenizer to use
        max_length: Maximum allowed length

    Returns:
        Dictionary containing analysis results

    """
    lengths = []
    outliers = []
    split_stats = defaultdict(list)

    for sample in tqdm(samples, desc="Analyzing lengths"):
        # Tokenize combined context and answer
        tokens = tokenizer(
            sample.prompt,
            sample.answer,
            truncation=False,  # No truncation to get true lengths
            add_special_tokens=True,
        )

        length = len(tokens["input_ids"])
        lengths.append(length)
        split_stats[sample.split].append(length)

        if length > max_length:
            outliers.append(
                {
                    "index": len(lengths) - 1,
                    "length": length,
                    "prompt_length": len(tokenizer(sample.prompt)["input_ids"]),
                    "answer_length": len(tokenizer(sample.answer)["input_ids"]),
                    "split": sample.split,
                }
            )

    # Calculate statistics
    stats = {
        "total_samples": len(samples),
        "max_length": max(lengths),
        "min_length": min(lengths),
        "mean_length": sum(lengths) / len(lengths),
        "num_outliers": len(outliers),
        "outliers": outliers,
        "split_stats": {
            split: {
                "count": len(lens),
                "max": max(lens),
                "min": min(lens),
                "mean": sum(lens) / len(lens),
                "outliers": len([l for l in lens if l > max_length]),
            }
            for split, lens in split_stats.items()
        },
    }

    return stats


def plot_length_distribution(lengths: List[int], output_path: str, max_length: int):
    """Plot the distribution of sequence lengths."""
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50)
    plt.axvline(x=max_length, color="r", linestyle="--", label=f"Max Length ({max_length})")
    plt.xlabel("Sequence Length")
    plt.ylabel("Count")
    plt.title("Distribution of Sequence Lengths")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def filter_dataset(
    data: HallucinationData, max_length: int, tokenizer: AutoTokenizer, output_path: Path
) -> HallucinationData:
    """Filter out samples that exceed max_length."""
    filtered_samples = []

    for sample in tqdm(data.samples, desc="Filtering samples"):
        tokens = tokenizer(
            sample.prompt,
            sample.answer,
            truncation=False,
            add_special_tokens=True,
        )

        if len(tokens["input_ids"]) <= max_length:
            filtered_samples.append(sample)

    filtered_data = HallucinationData(samples=filtered_samples)

    # Save filtered dataset
    with open(output_path, "w") as f:
        json.dump(filtered_data.to_json(), f, indent=2)

    return filtered_data


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and filter dataset based on sequence lengths"
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name for tokenizer",
    )
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset
    with open(args.input_file) as f:
        data = HallucinationData.from_json(json.load(f))

    # Analyze lengths
    stats = analyze_lengths(data.samples, tokenizer, args.max_length)

    # Save statistics
    with open(output_dir / "length_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Plot distribution
    plot_length_distribution(
        [
            len(tokenizer(s.prompt, s.answer, add_special_tokens=True)["input_ids"])
            for s in data.samples
        ],
        str(output_dir / "length_distribution.png"),
        args.max_length,
    )

    # Filter dataset
    filtered_data = filter_dataset(
        data, args.max_length, tokenizer, output_dir / "filtered_dataset.json"
    )

    print(f"\nAnalysis complete. Results saved to {output_dir}")
    print(f"Original dataset size: {len(data.samples)}")
    print(f"Filtered dataset size: {len(filtered_data.samples)}")
    print(f"Removed {len(data.samples) - len(filtered_data.samples)} samples")

    # Print split-wise statistics
    print("\nSplit-wise statistics:")
    for split, stats in stats["split_stats"].items():
        print(f"\n{split.upper()}:")
        print(f"  Total samples: {stats['count']}")
        print(f"  Samples exceeding max length: {stats['outliers']}")
        print(f"  Max length: {stats['max']}")
        print(f"  Mean length: {stats['mean']:.2f}")


if __name__ == "__main__":
    main()
