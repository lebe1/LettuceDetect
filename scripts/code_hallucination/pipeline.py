#!/usr/bin/env python3
"""Orchestrator for the code hallucination dataset pipeline.

Usage:
    # Run all phases
    python -m scripts.code_hallucination.pipeline --all

    # Run specific phase
    python -m scripts.code_hallucination.pipeline --phase 1

    # Test with a few examples
    python -m scripts.code_hallucination.pipeline --test 5

    # Override LLM settings via env vars:
    OPENAI_API_KEY=xxx API_BASE_URL=https://api.groq.com/openai/v1 MODEL=moonshotai/kimi-k2-instruct-0905 \
        python -m scripts.code_hallucination.pipeline --test 10
"""

import argparse
import json
import random

from .config import (
    API_BASE_URL,
    API_KEY,
    DATA_DIR,
    DOCS_PATH,
    FORMATS_PATH,
    HALLUCINATED_PATH,
    MODEL,
    QUERIES_PATH,
)


def load_jsonl_dict(path, key="instance_id", value_key=None) -> dict:
    """Load a JSONL file into a dict keyed by instance_id."""
    result = {}
    if not path.exists():
        return result
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if value_key:
                    result[entry[key]] = entry[value_key]
                else:
                    result[entry[key]] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    return result


def run_test(n: int = 5, api_key: str = API_KEY, base_url: str = API_BASE_URL, model: str = MODEL):
    """Run a quick test with n instances from the test split."""
    print("=" * 60)
    print(f"TEST MODE: Running pipeline with {n} instances")
    print(f"LLM: {model} @ {base_url}")
    print("=" * 60)

    from .swebench_loader import load_all_splits

    # Load and filter to test split
    all_instances = load_all_splits()
    test_instances = [i for i in all_instances if i["split"] == "test"]

    # Take n random instances
    rng = random.Random(42)
    selected = rng.sample(test_instances, min(n, len(test_instances)))
    print(f"Selected {len(selected)} test instances")

    # Save temporary instances
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    test_path = DATA_DIR / "test_instances.json"
    with open(test_path, "w") as f:
        json.dump(selected, f, indent=2)

    # Phase 2: Fetch sources (use GitHub API for test mode — no cloning needed)
    from .source_fetcher import run as run_fetch

    sources = run_fetch(selected, use_github_api=True)

    if not sources:
        print("No sources fetched, aborting test")
        return

    # Phase 3: Rewrite queries
    from .query_rewriter import run as run_queries

    run_queries(selected, api_key=api_key, base_url=base_url, model=model)

    # Phase 4: Context7 docs
    from .context7_docs import run as run_docs

    run_docs(selected)

    # Phase 5: Assign formats (needs LLM for code_with_explanation)
    from .format_builder import run as run_formats

    queries_dict = load_jsonl_dict(QUERIES_PATH, value_key="query")
    run_formats(selected, api_key=api_key, base_url=base_url, model=model, queries=queries_dict)

    # Phase 8: Select targets (before phase 6)
    from .splitter import select_hallucination_targets

    targets = select_hallucination_targets(selected)

    # Phase 6: Inject hallucinations
    from .hallucination_injector import run as run_inject

    formats = load_jsonl_dict(FORMATS_PATH)
    docs = load_jsonl_dict(DOCS_PATH, value_key="docs")
    to_inject = [i for i in selected if i["instance_id"] in targets]
    run_inject(
        to_inject, formats, queries_dict, docs=docs, api_key=api_key, base_url=base_url, model=model
    )

    # Phase 7: Assemble
    from .sample_assembler import run as run_assemble

    hallucinations = load_jsonl_dict(HALLUCINATED_PATH)
    samples, metadata = run_assemble(selected, queries_dict, docs, formats, hallucinations, targets)

    # Phase 9: Validate
    from .validator import run as run_validate

    run_validate(samples, metadata)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"Generated {len(samples)} samples from {n} test instances")

    # Show a sample
    if samples:
        print("\n--- Sample Example ---")
        s = samples[0]
        print(f"Prompt length: {len(s['prompt'])} chars")
        print(f"Answer length: {len(s['answer'])} chars")
        print(f"Labels: {len(s['labels'])}")
        print(f"Split: {s['split']}")
        print(f"Answer preview: {s['answer'][:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Code hallucination dataset pipeline")
    parser.add_argument(
        "--phase", nargs="+", type=int, choices=range(1, 10), help="Run specific phase(s)"
    )
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--test", type=int, metavar="N", help="Test with N instances")
    parser.add_argument("--api-key", type=str, default=API_KEY, help="LLM API key")
    parser.add_argument("--base-url", type=str, default=API_BASE_URL, help="LLM API base URL")
    parser.add_argument("--model", type=str, default=MODEL, help="LLM model name")
    args = parser.parse_args()

    if args.test:
        run_test(args.test, api_key=args.api_key, base_url=args.base_url, model=args.model)
        return

    if not args.phase and not args.all:
        parser.print_help()
        return

    phases = list(range(1, 10)) if args.all else args.phase

    for phase in sorted(phases):
        print(f"\n{'#' * 60}")
        print(f"# Running Phase {phase}")
        print(f"{'#' * 60}\n")

        if phase == 1:
            from .swebench_loader import run

            run()
        elif phase == 2:
            from .source_fetcher import run
            from .swebench_loader import load_instances

            run(load_instances())
        elif phase == 3:
            from .query_rewriter import run
            from .swebench_loader import load_instances

            run(load_instances(), api_key=args.api_key, base_url=args.base_url, model=args.model)
        elif phase == 4:
            from .context7_docs import run
            from .swebench_loader import load_instances

            run(load_instances())
        elif phase == 5:
            from .format_builder import run
            from .swebench_loader import load_instances

            queries = load_jsonl_dict(QUERIES_PATH, value_key="query")
            run(
                load_instances(),
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                queries=queries,
            )
        elif phase == 6:
            from .hallucination_injector import run
            from .splitter import select_hallucination_targets
            from .swebench_loader import load_instances

            instances = load_instances()
            formats = load_jsonl_dict(FORMATS_PATH)
            queries = load_jsonl_dict(QUERIES_PATH, value_key="query")
            docs = load_jsonl_dict(DOCS_PATH, value_key="docs")
            targets = select_hallucination_targets(instances)
            to_inject = [i for i in instances if i["instance_id"] in targets]
            run(
                to_inject,
                formats,
                queries,
                docs=docs,
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
            )
        elif phase == 7:
            from .sample_assembler import run
            from .splitter import select_hallucination_targets
            from .swebench_loader import load_instances

            instances = load_instances()
            queries = load_jsonl_dict(QUERIES_PATH, value_key="query")
            docs = load_jsonl_dict(DOCS_PATH, value_key="docs")
            formats = load_jsonl_dict(FORMATS_PATH)
            hallucinations = load_jsonl_dict(HALLUCINATED_PATH)
            targets = select_hallucination_targets(instances)
            run(instances, queries, docs, formats, hallucinations, targets)
        elif phase == 8:
            from .splitter import run
            from .swebench_loader import load_instances

            run(load_instances())
        elif phase == 9:
            from .validator import run

            run()

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
