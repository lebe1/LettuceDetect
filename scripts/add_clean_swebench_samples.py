#!/usr/bin/env python3
"""Add clean (non-hallucinated) samples from remaining SWE-bench Lite instances.

Fetches source files from GitHub and builds LettuceDetect-format samples
with empty labels (= supported/correct code).
"""

import json
import re
import time

import requests
from datasets import load_dataset

INPUT_DATASET = (
    "/Users/adamkovacs/projects/LettuceDetect/data/code_hallucination_lettucedetect_v2.json"
)
RAW_DATASET = "/Users/adamkovacs/projects/LettuceDetect/data/code_hallucination_dataset.json"
OUTPUT_PATH = INPUT_DATASET  # Overwrite with merged data

GITHUB_RAW_BASE = "https://raw.githubusercontent.com"
MAX_FILE_CHARS = 12000
MAX_NEW_SAMPLES = 200  # Cap to avoid too many GitHub requests


def fetch_file_from_github(repo: str, commit: str, filepath: str) -> str | None:
    url = f"{GITHUB_RAW_BASE}/{repo}/{commit}/{filepath}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.text[:MAX_FILE_CHARS]
        return None
    except Exception as e:
        print(f"    Error fetching {filepath}: {e}")
        return None


def extract_changed_files(patch: str) -> list[str]:
    files = []
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            match = re.search(r"b/(.+)$", line)
            if match:
                files.append(match.group(1))
    return files


def extract_code_from_patch(patch: str) -> str:
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
                added_lines.append(line[1:])
            elif line.startswith("-"):
                continue
            else:
                if line.startswith(" "):
                    added_lines.append(line[1:])
                else:
                    added_lines.append(line)
    return "\n".join(added_lines)


def build_prompt(source_files: dict[str, str], user_query: str) -> str:
    parts = []
    for filepath, content in source_files.items():
        parts.append(f"File: {filepath}\n```python\n{content}\n```")
    parts.append(f"User request: {user_query}")
    return "\n\n".join(parts)


def main():
    # Load existing data
    with open(INPUT_DATASET) as f:
        existing_data = json.load(f)
    print(f"Existing dataset: {len(existing_data)} samples")

    # Get already-used instance IDs
    with open(RAW_DATASET) as f:
        raw_data = json.load(f)
    used_ids = {item["instance_id"] for item in raw_data}
    print(f"Already used instance IDs: {len(used_ids)}")

    # Load SWE-bench Lite
    print("Loading SWE-bench Lite...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    # Filter to unused instances
    remaining = [s for s in ds if s["instance_id"] not in used_ids]
    print(f"Remaining SWE-bench instances: {len(remaining)}")

    # Cap the number of new samples
    remaining = remaining[:MAX_NEW_SAMPLES]
    print(f"Processing {len(remaining)} instances...")

    new_samples = []
    for i, sample in enumerate(remaining):
        instance_id = sample["instance_id"]
        repo = sample["repo"]
        commit = sample["base_commit"]
        patch = sample["patch"]

        print(f"\n[{i + 1}/{len(remaining)}] {instance_id}")

        # Extract changed files
        changed_files = extract_changed_files(patch)
        if not changed_files:
            print("  SKIP: No changed files found")
            continue

        # Fetch source files from GitHub
        source_files = {}
        for filepath in changed_files[:3]:  # Limit to 3 files per sample
            content = fetch_file_from_github(repo, commit, filepath)
            if content:
                source_files[filepath] = content
                print(f"  Fetched {filepath}: {len(content)} chars")
            else:
                print(f"  Failed: {filepath}")
            time.sleep(0.3)

        if not source_files:
            print("  SKIP: No source files fetched")
            continue

        # Extract code from gold patch
        code = extract_code_from_patch(patch)
        if not code.strip():
            print("  SKIP: Empty code")
            continue

        # Build prompt with source files + user query
        user_query = sample["problem_statement"][:500]
        prompt = build_prompt(source_files, user_query)

        new_samples.append(
            {
                "prompt": prompt,
                "answer": code,
                "labels": [],
                "split": "train",
                "task_type": "code_generation",
                "dataset": "code_hallucination_swebench",
                "language": "en",
            }
        )

        # Progress check
        if (i + 1) % 20 == 0:
            print(f"\n  Progress: {len(new_samples)} clean samples so far\n")

    # Merge with existing data
    merged = existing_data + new_samples
    print(f"\nNew clean samples: {len(new_samples)}")
    print(f"Total merged: {len(merged)}")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    # Stats
    n_clean = sum(1 for s in merged if not s["labels"])
    n_hall = sum(1 for s in merged if s["labels"])
    print("\nFinal dataset:")
    print(f"  Clean: {n_clean} ({n_clean / len(merged) * 100:.1f}%)")
    print(f"  Hallucinated: {n_hall} ({n_hall / len(merged) * 100:.1f}%)")
    print(f"  Total: {len(merged)}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
