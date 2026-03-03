"""Phase 5: Assign answer format to each instance."""

import json
import random

from .config import FORMAT_TYPES, FORMAT_WEIGHTS, FORMATS_PATH, SOURCE_CACHE_DIR


def assign_format(source_data: dict) -> tuple[str, str]:
    """Assign a format type and build the answer for an instance.

    Returns (format_type, answer_text).
    Falls back if preferred format isn't available.
    """
    has_functions = bool(source_data.get("modified_functions"))
    has_edit = bool(source_data.get("edit_style"))
    has_fragment = bool(source_data.get("patch_code", "").strip())

    # Build available formats
    available = []
    if has_functions:
        available.append("complete_function")
    if has_edit:
        available.append("edit_style")
    if has_fragment:
        available.append("fragment")

    if not available:
        return None, None

    # Weighted random choice from available formats
    weights = []
    for fmt in available:
        idx = FORMAT_TYPES.index(fmt)
        weights.append(FORMAT_WEIGHTS[idx])

    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]

    chosen = random.choices(available, weights=weights, k=1)[0]

    # Build answer text
    if chosen == "complete_function":
        funcs = source_data["modified_functions"]
        # Take the first (or longest) modified function
        func = max(funcs, key=lambda f: len(f.get("patched", "")))
        answer = func["patched"]
    elif chosen == "edit_style":
        answer = source_data["edit_style"]
    else:  # fragment
        answer = source_data["patch_code"]

    return chosen, answer


def run(instances: list[dict], source_cache_dir=SOURCE_CACHE_DIR):
    """Run Phase 5: Assign formats and build answers.

    Returns list of dicts with instance_id, format_type, answer.
    """
    print("=" * 60)
    print("Phase 5: Answer Format Building")
    print("=" * 60)

    FORMATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    format_counts = {fmt: 0 for fmt in FORMAT_TYPES}
    skipped = 0

    for inst in instances:
        instance_id = inst["instance_id"]

        # Load source data from cache
        cache_path = source_cache_dir / f"{instance_id}.json"
        if not cache_path.exists():
            skipped += 1
            continue

        with open(cache_path) as f:
            source_data = json.load(f)

        fmt, answer = assign_format(source_data)
        if fmt is None:
            skipped += 1
            continue

        results.append(
            {
                "instance_id": instance_id,
                "format_type": fmt,
                "answer": answer,
            }
        )
        format_counts[fmt] += 1

    # Save
    with open(FORMATS_PATH, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"\nAssigned formats for {len(results)} instances (skipped {skipped})")
    for fmt, count in format_counts.items():
        pct = count * 100 // max(len(results), 1)
        print(f"  {fmt}: {count} ({pct}%)")

    return results


if __name__ == "__main__":
    from .swebench_loader import load_instances

    instances = load_instances()
    run(instances)
