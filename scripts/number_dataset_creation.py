import re
import json
from concurrent.futures import ThreadPoolExecutor

def extract_numbers(text):
    """Extract all numbers (integers and floats) from the text."""
    return set(re.findall(r'\d+(?:\.\d+)?', text))

def is_hallucinated_numbers(batch):
    """Return True if batch contains a hallucinated number."""
    context = batch['prompt']
    answer = batch['answer']

    context_numbers = extract_numbers(context)
    answer_numbers = extract_numbers(answer)

    hallucinated = [num for num in answer_numbers if num not in context_numbers]

    return bool(hallucinated)

def filter_batches_with_hallucinated_numbers(dataset):
    """Filter batches with hallucinated numbers in parallel."""
    filtered = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(is_hallucinated_numbers, dataset))
    for batch, has_hallucination in zip(dataset, results):
        if has_hallucination:
            filtered.append(batch)
    return filtered

# Example usage:
if __name__ == "__main__":
    # Load your dataset here (replace with actual loading if reading from file)
    with open("../data/ragtruth/ragtruth_data/.json", "r") as f:
        dataset = json.load(f)

    hallucinated_batches = filter_batches_with_hallucinated_numbers(dataset)

    # Save the filtered hallucinated-number-only dataset
    with open("hallucinated_numbers_only.json", "w") as f:
        json.dump(hallucinated_batches, f, indent=2)
