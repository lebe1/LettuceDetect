#!/usr/bin/env python3
"""Enrich code hallucination dataset with actual source file contents from GitHub.

Takes the raw dataset and:
1. Fetches actual source files from GitHub at the base commit
2. Builds proper context (source files + docs + query)
3. Converts diff-format answers to actual code
4. Outputs in exact LettuceDetect HallucinationSample format
"""

import json
import os
import time

import requests

INPUT_PATH = "/Users/adamkovacs/projects/LettuceDetect/data/code_hallucination_dataset.json"
OUTPUT_PATH = (
    "/Users/adamkovacs/projects/LettuceDetect/data/code_hallucination_lettucedetect_v2.json"
)
GITHUB_RAW_BASE = "https://raw.githubusercontent.com"
MAX_FILE_CHARS = 12000  # Cap individual file size to avoid blowing up context


def fetch_file_from_github(repo: str, commit: str, filepath: str) -> str | None:
    """Fetch a file's contents from GitHub at a specific commit."""
    url = f"{GITHUB_RAW_BASE}/{repo}/{commit}/{filepath}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.text[:MAX_FILE_CHARS]
        return None
    except Exception as e:
        print(f"    Error fetching {filepath}: {e}")
        return None


def extract_code_from_patch(patch: str) -> str:
    """Extract just the added lines from a unified diff as the 'answer' code.

    Returns the new code (added lines) that represents what was generated.
    """
    added_lines = []
    in_hunk = False

    for line in patch.split("\n"):
        if line.startswith("@@"):
            in_hunk = True
            continue
        if line.startswith("diff --git") or line.startswith("---") or line.startswith("+++"):
            continue
        if in_hunk:
            if line.startswith("+"):
                added_lines.append(line[1:])  # Remove the '+' prefix
            elif line.startswith("-"):
                continue  # Skip removed lines
            else:
                # Context line - include to maintain readability
                if line.startswith(" "):
                    added_lines.append(line[1:])
                else:
                    added_lines.append(line)

    return "\n".join(added_lines)


def build_prompt(
    source_files: dict[str, str],
    documentation: dict[str, str],
    user_query: str,
) -> str:
    """Build the prompt (context) in the same style as RAGTruth.

    Format: all context concatenated, similar to how RAGTruth provides
    the source prompt that was given to the LLM.
    """
    parts = []

    # Source code files (the main context)
    for filepath, content in source_files.items():
        parts.append(f"File: {filepath}\n```python\n{content}\n```")

    # Documentation
    for lib, doc in documentation.items():
        parts.append(f"Documentation for {lib}:\n{doc}")

    # User query (acts as the "question" / instruction)
    parts.append(f"User request: {user_query}")

    return "\n\n".join(parts)


def compute_span_offsets_in_code(code: str, hallucinated_span: str) -> list[dict]:
    """Find character offsets of a hallucinated span within the answer code."""
    spans = []
    start = 0
    while True:
        idx = code.find(hallucinated_span, start)
        if idx == -1:
            break
        spans.append({"start": idx, "end": idx + len(hallucinated_span)})
        start = idx + 1
        break  # Take first occurrence only
    return spans


def process_sample(sample: dict, idx: int, total: int) -> list[dict]:
    """Process one raw sample into LettuceDetect format samples."""
    instance_id = sample["instance_id"]
    repo = sample["repo"]
    commit = sample["base_commit"]
    changed_files = sample["changed_files"]

    print(f"[{idx + 1}/{total}] {instance_id}")

    # Step 1: Fetch actual source files from GitHub
    source_files = {}
    for filepath in changed_files:
        # Some paths in SWE-bench have the 'a/' or weird format from diff parsing
        # Clean up: "models/fields/__init__.py b/django/db/models/fields/__init__.py"
        # Take the last valid-looking path
        clean_paths = [
            p.strip() for p in filepath.split(" ") if "/" in p and not p.startswith("a/")
        ]
        if not clean_paths:
            clean_paths = [filepath]

        for path in clean_paths:
            path = path.lstrip("b/")
            content = fetch_file_from_github(repo, commit, path)
            if content:
                source_files[path] = content
                print(f"  Fetched {path}: {len(content)} chars")
            else:
                print(f"  Failed to fetch {path}")
            time.sleep(0.3)  # Rate limit

    if not source_files:
        print("  SKIP: No source files fetched")
        return []

    # Step 2: Build the prompt (context)
    documentation = sample.get("documentation", {})
    user_query = sample.get("user_query", "")
    prompt = build_prompt(source_files, documentation, user_query)

    # Step 3: Create correct (negative) sample
    gold_code = extract_code_from_patch(sample["gold_patch"])
    samples_out = []

    if gold_code.strip():
        samples_out.append(
            {
                "prompt": prompt,
                "answer": gold_code,
                "labels": [],
                "split": "train",
                "task_type": "code_generation",
                "dataset": "code_hallucination_swebench",
                "language": "en",
            }
        )

    # Step 4: Create hallucinated (positive) samples
    for hall in sample.get("hallucinations", []):
        if not isinstance(hall, dict) or "hallucinated_patch" not in hall:
            continue

        hall_code = extract_code_from_patch(hall["hallucinated_patch"])
        if not hall_code.strip():
            continue

        # Compute span labels in the answer code
        labels = []
        for change in hall.get("changes", []):
            h_span = change.get("hallucinated", "")
            if h_span and h_span in hall_code:
                offsets = compute_span_offsets_in_code(hall_code, h_span)
                for offset in offsets:
                    labels.append(
                        {
                            "start": offset["start"],
                            "end": offset["end"],
                            "label": hall.get("type", "hallucinated"),
                        }
                    )

        if labels:
            samples_out.append(
                {
                    "prompt": prompt,
                    "answer": hall_code,
                    "labels": labels,
                    "split": "train",
                    "task_type": "code_generation",
                    "dataset": "code_hallucination_swebench",
                    "language": "en",
                }
            )

    print(
        f"  Generated {len(samples_out)} samples (1 correct + {len(samples_out) - 1} hallucinated)"
    )
    return samples_out


def main():
    print("=" * 60)
    print("Enriching Code Hallucination Dataset")
    print("Fetching source files + converting to LettuceDetect format")
    print("=" * 60)

    with open(INPUT_PATH) as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} raw samples\n")

    all_samples = []
    for i, sample in enumerate(raw_data):
        new_samples = process_sample(sample, i, len(raw_data))
        all_samples.extend(new_samples)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_samples, f, indent=2)

    # Stats
    print("\n" + "=" * 60)
    print("FINAL DATASET")
    print("=" * 60)
    print(f"Total samples: {len(all_samples)}")
    n_correct = sum(1 for s in all_samples if not s["labels"])
    n_hall = sum(1 for s in all_samples if s["labels"])
    print(f"Correct (negative): {n_correct}")
    print(f"Hallucinated (positive): {n_hall}")

    prompt_lens = [len(s["prompt"]) for s in all_samples]
    answer_lens = [len(s["answer"]) for s in all_samples]
    total_lens = [p + a for p, a in zip(prompt_lens, answer_lens)]

    print(
        f"\nPrompt chars  - min: {min(prompt_lens):,}, max: {max(prompt_lens):,}, avg: {sum(prompt_lens) // len(prompt_lens):,}"
    )
    print(
        f"Answer chars  - min: {min(answer_lens):,}, max: {max(answer_lens):,}, avg: {sum(answer_lens) // len(answer_lens):,}"
    )
    print(
        f"Total chars   - min: {min(total_lens):,}, max: {max(total_lens):,}, avg: {sum(total_lens) // len(total_lens):,}"
    )

    est_tokens = [t // 4 for t in total_lens]
    print(
        f"\nEst. tokens   - min: {min(est_tokens):,}, max: {max(est_tokens):,}, avg: {sum(est_tokens) // len(est_tokens):,}"
    )


if __name__ == "__main__":
    main()
