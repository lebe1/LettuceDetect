import argparse
import json
from pathlib import Path




from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
)

from lettucedetect.models.evaluator import (
    evaluate_detector_char_level,
    evaluate_detector_example_level
)

from lettucedetect.models.inference import HallucinationDetector

def evaluate_task_samples(
    samples,
    evaluation_type,
    detector=None,
):
    print(f"\nEvaluating model on {len(samples)} samples")

    if evaluation_type == "example_level":
        print("\n---- Example-Level Span Evaluation ----")
        metrics = evaluate_detector_example_level(detector, samples)
        return metrics
    ## TODO implement elif case for token level method for HallucinationDetector class in evaluator.py


    else:  # char_level
        print("\n---- Character-Level Span Evaluation ----")
        metrics = evaluate_detector_char_level(detector, samples)
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        return metrics


def load_data(data_path):
    data_path = Path(data_path)
    hallucination_data = HallucinationData.from_json(json.loads(data_path.read_text()))

    # Filter test samples from the data
    test_samples = [sample for sample in hallucination_data.samples if sample.split == "test"]

    # group samples by task type
    task_type_map = {}
    for sample in test_samples:
        if sample.task_type not in task_type_map:
            task_type_map[sample.task_type] = []
        task_type_map[sample.task_type].append(sample)
    return test_samples, task_type_map


def main():
    parser = argparse.ArgumentParser(description="Evaluate a hallucination detection model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the evaluation data (JSON format)",
    )
    parser.add_argument(
        "--evaluation_type",
        type=str,
        default="example_level",
        help="Evaluation type (token_level, example_level or char_level)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )

    args = parser.parse_args()

    test_samples, task_type_map = load_data(args.data_path)

    print(f"\nEvaluating model on test samples: {len(test_samples)}")
    model, tokenizer, device = None, None, None

        
    detector = HallucinationDetector(method="rule")

    # Evaluate each task type separately
    for task_type, samples in task_type_map.items():
        print(f"\nTask type: {task_type}")
        evaluate_task_samples(
            samples,
            args.evaluation_type,
            detector=detector,
        )

    # Evaluate the whole dataset
    print("\nTask type: whole dataset")
    evaluate_task_samples(
        test_samples,
        args.evaluation_type,
        detector=detector,
    )


if __name__ == "__main__":
    main()
