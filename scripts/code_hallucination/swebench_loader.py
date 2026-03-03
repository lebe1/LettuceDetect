"""Phase 1: Load SWE-bench instances from all splits."""

import json

from datasets import load_dataset

from .config import DATA_DIR, INSTANCES_PATH, SWEBENCH_FULL, SWEBENCH_LITE


def load_all_splits() -> list[dict]:
    """Load all SWE-bench splits and tag each instance.

    Returns list of dicts with fields:
        instance_id, repo, base_commit, patch, test_patch,
        problem_statement, hints_text, created_at, version,
        FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit,
        split, is_lite
    """
    # Load Lite instance IDs for tagging
    print("Loading SWE-bench Lite...")
    lite_ds = load_dataset(SWEBENCH_LITE, split="test")
    lite_ids = {row["instance_id"] for row in lite_ds}
    print(f"  Lite: {len(lite_ids)} instances")

    all_instances = []

    for split_name in ["train", "dev", "test"]:
        print(f"Loading SWE-bench {split_name} split...")
        ds = load_dataset(SWEBENCH_FULL, split=split_name)
        print(f"  {split_name}: {len(ds)} instances")

        for row in ds:
            instance = dict(row)
            instance["split"] = split_name
            instance["is_lite"] = instance["instance_id"] in lite_ids
            all_instances.append(instance)

    # Report stats
    repos_by_split = {}
    for inst in all_instances:
        split = inst["split"]
        repo = inst["repo"]
        if split not in repos_by_split:
            repos_by_split[split] = set()
        repos_by_split[split].add(repo)

    print(f"\nTotal: {len(all_instances)} instances")
    for split, repos in repos_by_split.items():
        count = sum(1 for i in all_instances if i["split"] == split)
        print(f"  {split}: {count} instances across {len(repos)} repos")

    n_lite = sum(1 for i in all_instances if i["is_lite"])
    print(f"  Lite-tagged: {n_lite}")

    # Verify zero repo overlap
    all_splits = list(repos_by_split.keys())
    for i, s1 in enumerate(all_splits):
        for s2 in all_splits[i + 1 :]:
            overlap = repos_by_split[s1] & repos_by_split[s2]
            if overlap:
                print(f"  WARNING: Repo overlap between {s1} and {s2}: {overlap}")
            else:
                print(f"  {s1} ∩ {s2}: 0 repo overlap ✓")

    return all_instances


def save_instances(instances: list[dict], path=INSTANCES_PATH):
    """Save instances to JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(instances, f, indent=2)
    print(f"Saved {len(instances)} instances to {path}")


def load_instances(path=INSTANCES_PATH) -> list[dict]:
    """Load previously saved instances."""
    with open(path) as f:
        return json.load(f)


def run():
    """Run Phase 1: Load and save all SWE-bench instances."""
    print("=" * 60)
    print("Phase 1: Load SWE-bench")
    print("=" * 60)
    instances = load_all_splits()
    save_instances(instances)
    return instances


if __name__ == "__main__":
    run()
