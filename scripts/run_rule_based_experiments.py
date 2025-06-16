import argparse
import json
import datetime
from pathlib import Path

from evaluate_rule import load_data, evaluate_task_samples_with_runtime
from lettucedetect.detectors.embedding import StaticEmbeddingDetector
from lettucedetect.detectors.rouge import RougeBasedDetector


def get_detectors():
    """Initialize both detectors."""
    return {
        "rouge": RougeBasedDetector(),
        "embedding": StaticEmbeddingDetector(),
    }


def save_results(base_output_path, model_method, evaluation_type, all_metrics, total_runtime):
    """Save results for a specific model to a file."""
    results = {
        "model": model_method,
        "evaluation_type": evaluation_type,
        #"timestamp": str(datetime.datetime.now()),
        "task_metrics": all_metrics,
        "total_runtime_seconds": total_runtime,
    }

    output_path = Path(base_output_path)
    output_path = output_path.with_name(f"{output_path.stem}_{model_method}{output_path.suffix}")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results for '{model_method}' to {output_path.resolve()}")


def run_experiment(detector, model_method, task_type_map, evaluation_type, test_samples):
    """Run evaluation for a single model, including per-task and whole dataset."""
    all_metrics = {}
    total_runtime = 0.0

    # Per-task evaluation
    for task_type, samples in task_type_map.items():
        print(f"\nEvaluating [{model_method.upper()}] - Task Type: {task_type}")
        metrics, runtime = evaluate_task_samples_with_runtime(samples, evaluation_type, detector)
        all_metrics[task_type] = {
            "metrics": metrics,
            "runtime_seconds": runtime
        }
        total_runtime += runtime

    # Whole dataset evaluation
    print(f"\nEvaluating [{model_method.upper()}] - Task Type: whole dataset")
    metrics, runtime = evaluate_task_samples_with_runtime(test_samples, evaluation_type, detector)
    all_metrics["whole_dataset"] = {
        "metrics": metrics,
        "runtime_seconds": runtime
    }
    total_runtime += runtime

    return all_metrics, total_runtime



def main():
    parser = argparse.ArgumentParser(description="Run hallucination detection experiments with all models")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the evaluation dataset (JSON)")
    parser.add_argument("--evaluation_type", type=str, default="example_level",
                        choices=["example_level", "token_level", "char_level"],
                        help="Evaluation type: token_level, example_level, or char_level")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Base path to save results. Results will be saved with model name appended.")

    args = parser.parse_args()

    # Load and group data
    test_samples, task_type_map = load_data(args.data_path)

    # Initialize both models
    detectors = get_detectors()
    print("DETECTORS",detectors)

    for model_method, detector in detectors.items():
        print(f"\n==== Running experiment for: {model_method.upper()} ====")
        all_metrics, total_runtime = run_experiment(
            detector, model_method, task_type_map, args.evaluation_type, test_samples
        )
        save_results(args.output_path, model_method, args.evaluation_type, all_metrics, total_runtime)


if __name__ == "__main__":
    main()
