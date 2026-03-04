#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

from lettucedetect.datasets.hallucination_dataset import HallucinationData

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("filter_data")


def filter_dataset(
    input_file: Path, output_file: Path, split: str = "test", task_type: str = "Data2txt"
) -> None:
    """Filter dataset to include only samples with specific split and task type.

    :param input_file: Path to the input JSON file
    :param output_file: Path to save the filtered JSON file
    :param split: The split to filter for (default: 'test')
    :param task_type: The task type to filter for (default: 'Data2txt')
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        with open(input_file, "r") as f:
            data = json.load(f)

        dataset = HallucinationData.from_json(data)
        logger.info(f"Loaded {len(dataset.samples)} samples from {input_file}")
    except Exception as e:
        logger.error(f"Error loading input data: {e}")
        raise

    # Filter samples
    filtered_samples = [
        sample
        for sample in dataset.samples
        if sample.split == split and sample.task_type == task_type
    ]

    # Create filtered dataset
    filtered_dataset = HallucinationData(samples=filtered_samples)

    # Save filtered data
    with open(output_file, "w") as f:
        json.dump(filtered_dataset.to_json(), f, indent=2)

    logger.info(f"Filtered dataset from {len(dataset.samples)} to {len(filtered_samples)} samples")
    logger.info(f"Saved filtered dataset to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter hallucination dataset to specific split and task type"
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to save filtered JSON file")
    parser.add_argument(
        "--split", type=str, default="test", help="Split to filter for (default: 'test')"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="Data2txt",
        help="Task type to filter for (default: 'Data2txt')",
    )

    args = parser.parse_args()

    filter_dataset(Path(args.input), Path(args.output), args.split, args.task_type)


if __name__ == "__main__":
    main()
