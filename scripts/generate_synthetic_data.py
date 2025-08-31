#!/usr/bin/env python3
"""Generate synthetic hallucination data using RAGFactChecker."""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

from lettucedetect import HallucinationGenerator
from lettucedetect.detectors.prompt_utils import PromptUtils

# Setup rich logging
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
    )

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging with rich output if available."""
    level = logging.DEBUG if verbose else logging.INFO

    if RICH_AVAILABLE:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
    else:
        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    return logging.getLogger(__name__)


def load_rag_mini_bioasq(split: str = "train", filter_min_words: int = 10) -> List[Dict[str, Any]]:
    """Load rag-mini-bioasq dataset and prepare for generation."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required. Install with: pip install datasets")

    logger = logging.getLogger(__name__)
    logger.info(f"Loading rag-mini-bioasq dataset ({split} split)...")

    # Load dataset
    qa_dataset = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")
    corpus_dataset = load_dataset("enelpol/rag-mini-bioasq", "text-corpus")

    # Create corpus lookup
    corpus_lookup = {item["id"]: item["passage"] for item in corpus_dataset["test"]}

    # Process data
    processed_data = []
    for item in qa_dataset[split]:
        passage_ids = item["relevant_passage_ids"]
        context_passages = [corpus_lookup.get(pid, None) for pid in passage_ids]
        context_passages = [p for p in context_passages if p is not None]

        # Filter by answer length
        if len(item["answer"].split()) >= filter_min_words:
            processed_data.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "context": context_passages,
                }
            )

    logger.info(
        f"Loaded {len(processed_data)} samples after filtering (min {filter_min_words} words)"
    )
    return processed_data


def load_custom_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load custom dataset from JSON file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading custom dataset from {file_path}...")

    with open(file_path) as f:
        data = json.load(f)

    # Validate format
    required_fields = ["question", "context"]
    for i, item in enumerate(data):
        for field in required_fields:
            if field not in item:
                raise ValueError(f"Missing required field '{field}' in item {i}")

    logger.info(f"Loaded {len(data)} samples from custom dataset")
    return data


async def generate_batch_async(
    generator: HallucinationGenerator,
    samples: List[Dict[str, Any]],
    method: str = "answer_based",
    error_types: Optional[List[str]] = None,
    intensity: float = 0.3,
) -> List[Dict[str, Any]]:
    """Generate hallucinated data for a batch of samples."""
    logger = logging.getLogger(__name__)

    if method == "answer_based":
        # Use existing answers
        contexts = [sample["context"] for sample in samples]
        questions = [sample["question"] for sample in samples]
        answers = [sample["answer"] for sample in samples]

        result = await generator.generate_batch_async(
            contexts=contexts,
            questions=questions,
            answers=answers,
            error_types=error_types,
            intensity=intensity,
        )
    else:
        # Context-based generation
        contexts = [sample["context"] for sample in samples]
        questions = [sample["question"] for sample in samples]

        result = await generator.generate_batch_async(
            contexts=contexts, questions=questions, error_types=error_types, intensity=intensity
        )

    return result.results if hasattr(result, "results") else result


def convert_to_ragtruth_format(
    samples: List[Dict[str, Any]],
    results: List[Any],
    language: str = "en",
    dataset_name: str = "synthetic",
) -> List[Dict[str, Any]]:
    """Convert generation results to RAGTruth format."""
    ragtruth_data = []

    for i, (sample, result) in enumerate(zip(samples, results)):
        # Format context using prompt utils
        formatted_prompt = PromptUtils.format_context(
            sample["context"], sample["question"], lang=language
        )

        # Original answer (non-hallucinated)
        if hasattr(result, "generated_non_hlcntn_answer"):
            real_answer = result.generated_non_hlcntn_answer
        else:
            real_answer = sample.get("answer", "")

        ragtruth_data.append(
            {
                "prompt": formatted_prompt,
                "answer": real_answer,
                "labels": [],
                "split": "train",
                "task_type": "qa",
                "dataset": dataset_name,
                "language": language,
            }
        )

        # Hallucinated answer with labels
        if hasattr(result, "generated_hlcntn_answer"):
            hallucinated_answer = result.generated_hlcntn_answer
            hallucinated_labels = []

            # Create span labels from hallucinated parts
            if hasattr(result, "hlcntn_part") and result.hlcntn_part:
                for part in result.hlcntn_part:
                    if isinstance(part, str) and part in hallucinated_answer:
                        start = hallucinated_answer.find(part)
                        if start != -1:
                            hallucinated_labels.append(
                                {"start": start, "end": start + len(part), "label": "hallucinated"}
                            )

            ragtruth_data.append(
                {
                    "prompt": formatted_prompt,
                    "answer": hallucinated_answer,
                    "labels": hallucinated_labels,
                    "split": "train",
                    "task_type": "qa",
                    "dataset": dataset_name,
                    "language": language,
                }
            )

    return ragtruth_data


async def generate_synthetic_data(
    samples: List[Dict[str, Any]],
    num_samples: int,
    model: str = "gpt-4o",
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    method: str = "answer_based",
    error_types: Optional[List[str]] = None,
    intensity: float = 0.3,
    batch_size: int = 10,
    output_format: str = "json",
    language: str = "en",
    dataset_name: str = "synthetic",
) -> List[Dict[str, Any]]:
    """Generate synthetic hallucination data."""
    logger = logging.getLogger(__name__)

    # Initialize generator
    generator = HallucinationGenerator(
        method="rag_fact_checker", model=model, base_url=base_url, temperature=temperature
    )

    # Sample data if needed
    if num_samples < len(samples):
        samples = random.sample(samples, num_samples)
        logger.info(f"Randomly sampled {num_samples} examples from dataset")
    else:
        samples = samples[:num_samples]

    # Process in batches
    all_results = []

    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Generating hallucinations ({method})", total=len(samples))

            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]

                try:
                    batch_results = await generate_batch_async(
                        generator, batch, method, error_types, intensity
                    )
                    all_results.extend(batch_results)

                    progress.update(task, advance=len(batch))
                    logger.debug(
                        f"Completed batch {i // batch_size + 1}/{(len(samples) + batch_size - 1) // batch_size}"
                    )

                except Exception as e:
                    logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
                    continue
    else:
        # Fallback without rich
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(samples) + batch_size - 1) // batch_size}"
            )

            try:
                batch_results = await generate_batch_async(
                    generator, batch, method, error_types, intensity
                )
                all_results.extend(batch_results)

            except Exception as e:
                logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
                continue

    logger.info(f"Generated {len(all_results)} hallucination samples")

    # Convert to requested format
    if output_format == "ragtruth":
        return convert_to_ragtruth_format(samples, all_results, language, dataset_name)
    else:
        # Standard JSON format
        formatted_results = []
        for sample, result in zip(samples, all_results):
            formatted_result = {
                "question": sample["question"],
                "context": sample["context"],
                "method": method,
                "model": model,
                "temperature": temperature,
            }

            if hasattr(result, "generated_non_hlcntn_answer"):
                formatted_result["original_answer"] = result.generated_non_hlcntn_answer
            if hasattr(result, "generated_hlcntn_answer"):
                formatted_result["hallucinated_answer"] = result.generated_hlcntn_answer
            if hasattr(result, "hlcntn_part"):
                formatted_result["hallucinated_parts"] = result.hlcntn_part

            formatted_results.append(formatted_result)

        return formatted_results


def print_statistics(results: List[Dict[str, Any]], output_format: str):
    """Print generation statistics."""
    logger = logging.getLogger(__name__)

    if not results:
        logger.warning("No results to analyze")
        return

    total_samples = len(results)

    if output_format == "ragtruth":
        # Count hallucinated vs non-hallucinated samples
        hallucinated_count = sum(1 for r in results if r.get("labels"))
        non_hallucinated_count = total_samples - hallucinated_count

        logger.info("📊 Generation Statistics:")
        logger.info(f"   Total samples: {total_samples}")
        logger.info(f"   Hallucinated samples: {hallucinated_count}")
        logger.info(f"   Non-hallucinated samples: {non_hallucinated_count}")

        if hallucinated_count > 0:
            # Average number of hallucination spans
            total_spans = sum(len(r.get("labels", [])) for r in results if r.get("labels"))
            avg_spans = total_spans / hallucinated_count
            logger.info(f"   Average spans per hallucinated sample: {avg_spans:.1f}")
    else:
        logger.info("📊 Generation Statistics:")
        logger.info(f"   Total samples: {total_samples}")

        # Calculate average lengths
        if results and "hallucinated_answer" in results[0]:
            avg_hal_len = (
                sum(len(r["hallucinated_answer"].split()) for r in results) / total_samples
            )
            logger.info(f"   Average hallucinated answer length: {avg_hal_len:.1f} words")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic hallucination data using RAGFactChecker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from rag-mini-bioasq dataset
  python scripts/generate_synthetic_data.py \\
    --dataset rag-mini-bioasq \\
    --split train \\
    --num-samples 100 \\
    --model gpt-4o-mini \\
    --output data/synthetic_train.json

  # Generate with custom parameters  
  python scripts/generate_synthetic_data.py \\
    --dataset rag-mini-bioasq \\
    --split test \\
    --num-samples 50 \\
    --model gpt-4o \\
    --temperature 0.7 \\
    --error-types factual temporal numerical \\
    --intensity 0.5 \\
    --output-format ragtruth \\
    --output data/synthetic_test_ragtruth.json
        """,
    )

    # Data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--dataset", choices=["rag-mini-bioasq"], help="Use built-in dataset")
    data_group.add_argument("--custom-data", type=str, help="Path to custom JSON dataset file")

    # Dataset options
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples to generate (default: 100)"
    )
    parser.add_argument(
        "--filter-min-words",
        type=int,
        default=10,
        help="Minimum words in answer for filtering (default: 10)",
    )

    # Generation parameters
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument(
        "--base-url", type=str, help="Base URL for OpenAI-compatible API (for local models)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for generation (default: 0.0)"
    )
    parser.add_argument(
        "--method",
        choices=["context_based", "answer_based"],
        default="answer_based",
        help="Generation method (default: answer_based)",
    )
    parser.add_argument(
        "--error-types",
        nargs="+",
        choices=["factual", "temporal", "numerical", "logical", "causal"],
        default=None,
        help="Error types for answer-based generation (default: None)",
    )
    parser.add_argument(
        "--intensity", type=float, default=0.3, help="Error intensity 0.1-1.0 (default: 0.3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Batch size for processing (default: 5)"
    )

    # Output options
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument(
        "--output-format",
        choices=["json", "ragtruth"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--language", default="en", help="Language code for RAGTruth format (default: en)"
    )
    parser.add_argument(
        "--dataset-name",
        default="synthetic",
        help="Dataset name for RAGTruth format (default: synthetic)",
    )

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    # Load data
    try:
        if args.dataset == "rag-mini-bioasq":
            samples = load_rag_mini_bioasq(args.split, args.filter_min_words)
        else:
            samples = load_custom_dataset(args.custom_data)

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Validate parameters
    if args.num_samples <= 0:
        logger.error("Number of samples must be positive")
        sys.exit(1)

    if not (0.1 <= args.intensity <= 1.0):
        logger.error("Intensity must be between 0.1 and 1.0")
        sys.exit(1)

    # Generate data
    start_time = time.time()

    try:
        results = await generate_synthetic_data(
            samples=samples,
            num_samples=args.num_samples,
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            method=args.method,
            error_types=args.error_types,
            intensity=args.intensity,
            batch_size=args.batch_size,
            output_format=args.output_format,
            language=args.language,
            dataset_name=args.dataset_name,
        )

        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        elapsed_time = time.time() - start_time

        # Print statistics
        print_statistics(results, args.output_format)
        logger.info(f"Generated {len(results)} samples in {elapsed_time:.1f}s")
        logger.info(f"Results saved to {args.output}")

    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
