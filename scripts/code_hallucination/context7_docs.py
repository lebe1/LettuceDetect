"""Phase 4: Fetch library documentation from Context7 API.

Only fetches docs for ~50% of instances (configurable via DOCS_RATIO).
The other 50% get empty docs, creating training variety — models learn
to handle both with-docs and without-docs scenarios.
"""

import json
import random
import re
import time

import requests

from .config import CONTEXT7_API_KEY, CONTEXT7_BASE, DOCS_PATH, DOCS_RATIO, MAX_CONTEXT7_CHARS

# Map repo paths / import names to likely library names for Context7
PATH_TO_LIB = {
    "django": "django",
    "astropy": "astropy",
    "sympy": "sympy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "requests": "requests",
    "flask": "flask",
    "pytest": "pytest",
    "sphinx": "sphinx",
    "xarray": "xarray",
    "seaborn": "seaborn",
    "pylint": "pylint",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "transformers": "transformers",
    "jax": "jax",
    "torch": "pytorch",
    "tensorflow": "tensorflow",
    "sqlalchemy": "sqlalchemy",
    "celery": "celery",
    "pydantic": "pydantic",
    "fastapi": "fastapi",
    "httpx": "httpx",
}


def extract_imports_from_patch(patch: str) -> list[str]:
    """Extract Python import statements from added lines in a patch."""
    imports = set()
    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            clean = line[1:].strip()
            if clean.startswith("import ") or clean.startswith("from "):
                match = re.match(r"(?:from|import)\s+([\w.]+)", clean)
                if match:
                    module = match.group(1).split(".")[0]
                    if module and not module.startswith("_"):
                        imports.add(module)
    return list(imports)


def extract_libraries_from_files(changed_files: list[str]) -> list[str]:
    """Infer libraries from file paths."""
    libs = set()
    for f in changed_files:
        for key, lib in PATH_TO_LIB.items():
            if key in f:
                libs.add(lib)
    return list(libs)


def fetch_context7_docs(
    library_name: str, query: str, max_chars: int = MAX_CONTEXT7_CHARS
) -> str | None:
    """Fetch documentation from Context7 for a library + query."""
    try:
        headers = {}
        if CONTEXT7_API_KEY:
            headers["Authorization"] = f"Bearer {CONTEXT7_API_KEY}"

        r = requests.get(
            f"{CONTEXT7_BASE}/libs/search",
            params={"query": query, "libraryName": library_name},
            headers=headers,
            timeout=10,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
        if not results:
            return None

        lib_id = results[0]["id"]

        r2 = requests.get(
            f"{CONTEXT7_BASE}/context",
            params={"libraryId": lib_id, "query": query, "type": "txt"},
            headers=headers,
            timeout=10,
        )
        if r2.status_code != 200:
            return None

        doc_text = r2.text[:max_chars]
        return doc_text if doc_text.strip() else None
    except Exception as e:
        print(f"  Context7 error for {library_name}: {e}")
        return None


def get_documentation_for_instance(
    changed_files: list[str], patch: str, problem_statement: str
) -> dict[str, str]:
    """Fetch documentation for libraries referenced in an instance."""
    imported_libs = extract_imports_from_patch(patch)
    path_libs = extract_libraries_from_files(changed_files)
    all_libs = list(set(imported_libs + path_libs))

    short_query = problem_statement[:200].replace("\n", " ").strip()

    docs = {}
    for lib in all_libs[:3]:
        doc = fetch_context7_docs(lib, short_query)
        if doc:
            docs[lib] = doc
        time.sleep(0.5)

    return docs


def select_docs_instances(
    instances: list[dict], ratio: float = DOCS_RATIO, seed: int = 42
) -> set[str]:
    """Select which instances should get documentation fetched.

    Returns set of instance_ids that should have docs.
    """
    rng = random.Random(seed)
    ids = [inst["instance_id"] for inst in instances]
    n_with_docs = int(len(ids) * ratio)
    rng.shuffle(ids)
    return set(ids[:n_with_docs])


def load_existing_docs(path=DOCS_PATH) -> dict[str, dict]:
    """Load already-fetched docs for resumability."""
    existing = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    existing[entry["instance_id"]] = entry["docs"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return existing


def run(instances: list[dict]):
    """Run Phase 4: Fetch documentation for selected instances (~50%)."""
    print("=" * 60)
    print("Phase 4: Context7 Documentation")
    print("=" * 60)

    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Select which instances get docs
    docs_ids = select_docs_instances(instances)
    print(
        f"Selected {len(docs_ids)}/{len(instances)} instances for documentation ({DOCS_RATIO:.0%})"
    )

    existing = load_existing_docs()
    print(f"Already fetched: {len(existing)} instances")

    to_process = [inst for inst in instances if inst["instance_id"] not in existing]
    print(f"Remaining: {len(to_process)} instances to process")

    processed = 0
    with_docs = 0
    skipped_by_ratio = 0

    with open(DOCS_PATH, "a") as f:
        for i, inst in enumerate(to_process):
            instance_id = inst["instance_id"]

            # Skip docs for instances not selected (write empty docs)
            if instance_id not in docs_ids:
                entry = {"instance_id": instance_id, "docs": {}}
                f.write(json.dumps(entry) + "\n")
                f.flush()
                processed += 1
                skipped_by_ratio += 1
                continue

            changed_files = inst.get("changed_files", [])
            if not changed_files:
                from .source_fetcher import extract_changed_files

                changed_files = extract_changed_files(inst["patch"])

            docs = get_documentation_for_instance(
                changed_files, inst["patch"], inst["problem_statement"]
            )

            entry = {"instance_id": instance_id, "docs": docs}
            f.write(json.dumps(entry) + "\n")
            f.flush()

            processed += 1
            if docs:
                with_docs += 1

            if processed % 100 == 0:
                print(
                    f"  Progress: {processed}/{len(to_process)} ({with_docs} with docs, {skipped_by_ratio} skipped)"
                )

    print(
        f"\nDone: {processed} processed, {with_docs} with docs, {skipped_by_ratio} skipped (no-docs by design)"
    )


if __name__ == "__main__":
    from .swebench_loader import load_instances

    instances = load_instances()
    run(instances)
