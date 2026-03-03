#!/usr/bin/env python3
"""Generate a code hallucination detection dataset from SWE-bench + Context7.

Pipeline:
1. Load SWE-bench Lite samples
2. Extract context files and imports from patches
3. Fetch relevant documentation via Context7
4. Transform issues into user-style queries (via LLM)
5. Inject hallucinations (structural + behavioral + semantic)
6. Convert to LettuceDetect training format
"""

import json
import os
import re
import textwrap
import time
from typing import Any

import requests
from openai import OpenAI

# === Config ===
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "moonshotai/kimi-k2-instruct-0905"
CONTEXT7_BASE = "https://context7.com/api/v2"
OUTPUT_PATH = "/Users/adamkovacs/projects/LettuceDetect/data/code_hallucination_dataset.json"
LETTUCEDETECT_OUTPUT_PATH = (
    "/Users/adamkovacs/projects/LettuceDetect/data/code_hallucination_lettucedetect.json"
)
NUM_SAMPLES = 50
RETRY_DELAY = 2
MAX_RETRIES = 3


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY to your Groq API key")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


# === Patch Parsing ===


def extract_changed_files(patch: str) -> list[str]:
    """Extract file paths changed in a unified diff."""
    files = []
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            match = re.search(r"b/(.+)$", line)
            if match:
                files.append(match.group(1))
    return files


def extract_imports_from_patch(patch: str) -> list[str]:
    """Extract Python import statements from added lines in a patch."""
    imports = set()
    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            clean = line[1:].strip()
            # Match import statements
            if clean.startswith("import ") or clean.startswith("from "):
                # Extract the top-level module
                match = re.match(r"(?:from|import)\s+([\w.]+)", clean)
                if match:
                    module = match.group(1).split(".")[0]
                    # Filter out local/relative imports and stdlib
                    if module and not module.startswith("_"):
                        imports.add(module)
    return list(imports)


def extract_libraries_from_files(changed_files: list[str]) -> list[str]:
    """Infer which external libraries might be relevant from file paths."""
    # Map repo paths to likely library names
    path_to_lib = {
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
    }
    libs = set()
    for f in changed_files:
        for key, lib in path_to_lib.items():
            if key in f:
                libs.add(lib)
    return list(libs)


# === Context7 Documentation ===


def fetch_context7_docs(library_name: str, query: str, max_chars: int = 2000) -> str | None:
    """Fetch documentation from Context7 for a library + query."""
    try:
        # Step 1: Resolve library ID
        r = requests.get(
            f"{CONTEXT7_BASE}/libs/search",
            params={"query": query, "libraryName": library_name},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
        if not results:
            return None

        lib_id = results[0]["id"]

        # Step 2: Get relevant docs
        r2 = requests.get(
            f"{CONTEXT7_BASE}/context",
            params={"libraryId": lib_id, "query": query, "type": "txt"},
            timeout=10,
        )
        if r2.status_code != 200:
            return None

        doc_text = r2.text[:max_chars]
        return doc_text if doc_text.strip() else None

    except Exception as e:
        print(f"  Context7 error for {library_name}: {e}")
        return None


def get_documentation_context(
    changed_files: list[str], patch: str, problem_statement: str
) -> dict[str, str]:
    """Fetch documentation for libraries referenced in the patch."""
    docs = {}

    # Get libraries from imports in the patch
    imported_libs = extract_imports_from_patch(patch)
    # Get libraries from file paths
    path_libs = extract_libraries_from_files(changed_files)

    all_libs = list(set(imported_libs + path_libs))

    # Extract a short query from the problem statement
    short_query = problem_statement[:200].replace("\n", " ").strip()

    for lib in all_libs[:3]:  # Limit to 3 libraries to avoid rate limits
        doc = fetch_context7_docs(lib, short_query)
        if doc:
            docs[lib] = doc

    return docs


# === LLM Calls ===


def llm_call(
    client: OpenAI, system: str, user: str, temperature: float = 0.7, max_tokens: int = 500
) -> str:
    """Make an LLM call with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  LLM error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def transform_to_user_query(client: OpenAI, problem_statement: str, repo: str) -> str:
    """Transform a GitHub issue into a realistic user-to-agent query."""
    system = textwrap.dedent("""\
        You transform GitHub issue descriptions into realistic user queries
        that a developer would type into an AI coding assistant (like Claude Code or Cursor).

        Rules:
        - Make it conversational and natural
        - Keep the core technical ask but remove GitHub formatting
        - Remove reproduction steps, stack traces, verbose details
        - Keep it to 1-3 sentences
        - Don't mention "issue" or "bug report"
        - Sound like someone asking for help, not filing a report
    """)
    user = f"Repository: {repo}\n\nGitHub Issue:\n{problem_statement[:3000]}"
    return llm_call(client, system, user, temperature=0.7, max_tokens=300)


def inject_hallucinations(
    client: OpenAI,
    gold_patch: str,
    user_query: str,
    problem_statement: str,
    repo: str,
    documentation: dict[str, str],
) -> dict[str, Any]:
    """Inject hallucinations into a gold patch."""
    doc_context = ""
    if documentation:
        doc_context = "\n\nRelevant library documentation:\n"
        for lib, doc in documentation.items():
            doc_context += f"\n--- {lib} docs ---\n{doc[:800]}\n"

    system = textwrap.dedent("""\
        You are a code hallucination injector for building a hallucination detection dataset.

        Given a correct code patch, user query, and documentation context, create THREE
        hallucinated versions of the patch:

        1. STRUCTURAL: Change a function call, import, or parameter to something that
           doesn't exist or is wrong. Code should still parse but reference non-existent
           APIs, wrong methods, or invented parameters. If documentation is provided,
           you can hallucinate by using API calls that contradict the docs.

        2. BEHAVIORAL: Use correct APIs but with wrong values or logic. Wrong defaults,
           off-by-one errors, swapped conditions, wrong argument values.

        3. SEMANTIC: Code that looks like it addresses the user's request but does
           something subtly different or opposite. This should be the most subtle -
           the code parses, uses real APIs, but fails to do what was asked.
           Examples: implementing global when per-item was asked, any() vs all(),
           catching exceptions instead of fixing root cause, inverted conditions.

        Respond in this exact JSON format (no markdown, no code blocks):
        {
            "hallucinations": [
                {
                    "type": "structural",
                    "hallucinated_patch": "the full modified patch text",
                    "changes": [
                        {
                            "original": "exact original code span",
                            "hallucinated": "what you changed it to",
                            "explanation": "why this is a hallucination"
                        }
                    ]
                },
                {
                    "type": "behavioral",
                    "hallucinated_patch": "...",
                    "changes": [...]
                },
                {
                    "type": "semantic",
                    "hallucinated_patch": "...",
                    "changes": [...]
                }
            ]
        }

        IMPORTANT:
        - Hallucinations must be PLAUSIBLE - something an LLM would realistically generate
        - Each change must be subtle, not obviously broken
        - Return ONLY valid JSON
    """)

    user = f"""Repository: {repo}

User's query: {user_query}

Original issue context: {problem_statement[:2000]}{doc_context}

Correct gold patch:
{gold_patch}

Generate three hallucinated versions of this patch."""

    raw = llm_call(client, system, user, temperature=0.8, max_tokens=4000)

    # Parse JSON from response
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return {"raw_response": raw, "parse_error": True}
    return {"raw_response": raw, "parse_error": True}


# === LettuceDetect Format Conversion ===


def compute_span_offsets(code: str, hallucinated_span: str) -> list[dict]:
    """Find the character offsets of a hallucinated span within code."""
    spans = []
    start = 0
    while True:
        idx = code.find(hallucinated_span, start)
        if idx == -1:
            break
        spans.append({"start": idx, "end": idx + len(hallucinated_span)})
        start = idx + 1
    return spans


def convert_to_lettucedetect_format(samples: list[dict]) -> list[dict]:
    """Convert enriched SWE-bench samples to LettuceDetect training format.

    For each hallucinated variant, create a training sample with:
    - prompt: context (codebase files + docs + user query)
    - answer: the hallucinated code
    - labels: span annotations of hallucinated parts
    """
    ld_samples = []

    for sample in samples:
        # Build context prompt from available information
        context_parts = []

        # Add user query as the "question"
        user_query = sample.get("user_query", "")

        # Add documentation context
        docs = sample.get("documentation", {})
        if docs:
            for lib, doc in docs.items():
                context_parts.append(f"Documentation for {lib}:\n{doc}")

        # Add the gold patch as reference context (what the correct code looks like)
        context_parts.append(f"Changed files: {', '.join(sample.get('changed_files', []))}")

        # Add original problem statement as additional context
        problem = sample.get("original_problem_statement", "")
        if problem:
            context_parts.append(f"Issue description:\n{problem[:1500]}")

        context_text = "\n\n".join(context_parts)

        # Create a "correct" sample (gold patch, no hallucination labels)
        gold_patch = sample.get("gold_patch", "")
        ld_samples.append(
            {
                "prompt": context_text,
                "answer": gold_patch,
                "labels": [],
                "split": "train",
                "task_type": "code_generation",
                "dataset": "code_hallucination_swebench",
                "language": "python",
                "instance_id": sample["instance_id"],
                "repo": sample["repo"],
                "user_query": user_query,
            }
        )

        # Create hallucinated samples
        for hall in sample.get("hallucinations", []):
            if isinstance(hall, dict) and "hallucinated_patch" in hall:
                h_patch = hall["hallucinated_patch"]
                labels = []

                for change in hall.get("changes", []):
                    h_span = change.get("hallucinated", "")
                    if h_span and h_span in h_patch:
                        offsets = compute_span_offsets(h_patch, h_span)
                        for offset in offsets[:1]:  # Take first occurrence
                            labels.append(
                                {
                                    "start": offset["start"],
                                    "end": offset["end"],
                                    "label": "hallucinated",
                                    "hallucination_type": hall.get("type", "unknown"),
                                    "explanation": change.get("explanation", ""),
                                    "original_span": change.get("original", ""),
                                }
                            )

                if labels:  # Only add if we found valid span offsets
                    ld_samples.append(
                        {
                            "prompt": context_text,
                            "answer": h_patch,
                            "labels": labels,
                            "split": "train",
                            "task_type": "code_generation",
                            "dataset": "code_hallucination_swebench",
                            "language": "python",
                            "instance_id": sample["instance_id"],
                            "repo": sample["repo"],
                            "user_query": user_query,
                            "hallucination_type": hall.get("type", "unknown"),
                        }
                    )

    return ld_samples


# === Main Pipeline ===


def process_sample(client: OpenAI, sample: dict, idx: int, total: int) -> dict | None:
    """Process a single SWE-bench sample."""
    instance_id = sample["instance_id"]
    repo = sample["repo"]
    problem_statement = sample["problem_statement"]
    gold_patch = sample["patch"]

    print(f"\n[{idx + 1}/{total}] {instance_id} ({repo})")

    # Step 1: Extract file and import info
    changed_files = extract_changed_files(gold_patch)
    print(f"  Files: {changed_files}")

    # Step 2: Fetch documentation via Context7
    print("  Fetching documentation from Context7...")
    documentation = get_documentation_context(changed_files, gold_patch, problem_statement)
    if documentation:
        print(f"  Got docs for: {list(documentation.keys())}")
    else:
        print("  No external docs found (repo-internal change)")

    # Step 3: Transform issue to user query
    print("  Generating user query...")
    try:
        user_query = transform_to_user_query(client, problem_statement, repo)
        print(f"  Query: {user_query[:120]}...")
    except Exception as e:
        print(f"  ERROR generating query: {e}")
        return None

    # Small delay to avoid rate limits
    time.sleep(1)

    # Step 4: Inject hallucinations
    print("  Injecting hallucinations...")
    try:
        hall_result = inject_hallucinations(
            client, gold_patch, user_query, problem_statement, repo, documentation
        )
    except Exception as e:
        print(f"  ERROR injecting hallucinations: {e}")
        return None

    if hall_result.get("parse_error"):
        print("  WARNING: Failed to parse hallucination JSON")
        return None

    hallucinations = hall_result.get("hallucinations", [])
    print(f"  Generated {len(hallucinations)} hallucination variants")
    for h in hallucinations:
        h_type = h.get("type", "?")
        n_changes = len(h.get("changes", []))
        print(f"    - {h_type}: {n_changes} changes")

    # Small delay between samples
    time.sleep(1)

    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": sample["base_commit"],
        "original_problem_statement": problem_statement,
        "user_query": user_query,
        "changed_files": changed_files,
        "gold_patch": gold_patch,
        "test_patch": sample["test_patch"],
        "fail_to_pass": sample["FAIL_TO_PASS"],
        "documentation": documentation,
        "hallucinations": hallucinations,
    }


def select_diverse_samples(dataset, n: int) -> list[int]:
    """Select diverse samples across different repos."""
    # Group by repo
    repo_indices: dict[str, list[int]] = {}
    for i in range(len(dataset)):
        repo = dataset[i]["repo"]
        if repo not in repo_indices:
            repo_indices[repo] = []
        repo_indices[repo].append(i)

    print(f"Repos available: {list(repo_indices.keys())}")
    print(f"Samples per repo: {', '.join(f'{k}: {len(v)}' for k, v in repo_indices.items())}")

    # Round-robin across repos
    selected = []
    repo_iters = {repo: iter(indices) for repo, indices in repo_indices.items()}
    repos = list(repo_indices.keys())
    repo_idx = 0

    while len(selected) < n:
        repo = repos[repo_idx % len(repos)]
        try:
            idx = next(repo_iters[repo])
            selected.append(idx)
        except StopIteration:
            repos.remove(repo)
            if not repos:
                break
        repo_idx += 1

    return selected[:n]


def main():
    from datasets import load_dataset

    client = get_client()

    print("=" * 60)
    print("Code Hallucination Dataset Generator")
    print("SWE-bench + Context7 + LLM Injection")
    print("=" * 60)

    # Load SWE-bench Lite
    print("\nLoading SWE-bench Lite...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    print(f"Loaded {len(ds)} samples")

    # Select diverse samples
    indices = select_diverse_samples(ds, NUM_SAMPLES)
    print(f"\nSelected {len(indices)} diverse samples")

    # Process each sample
    results = []
    failed = 0

    for i, idx in enumerate(indices):
        sample = ds[idx]
        result = process_sample(client, sample, i, len(indices))
        if result:
            results.append(result)
        else:
            failed += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(
                f"\n  === Progress: {i + 1}/{len(indices)}, success: {len(results)}, failed: {failed} ===\n"
            )

    # Save raw results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} raw samples to {OUTPUT_PATH}")

    # Convert to LettuceDetect format
    print("\nConverting to LettuceDetect training format...")
    ld_samples = convert_to_lettucedetect_format(results)

    with open(LETTUCEDETECT_OUTPUT_PATH, "w") as f:
        json.dump(ld_samples, f, indent=2)
    print(f"Saved {len(ld_samples)} LettuceDetect samples to {LETTUCEDETECT_OUTPUT_PATH}")

    # Statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"SWE-bench samples processed: {len(results)}")
    print(f"Failed samples: {failed}")

    repos = set(r["repo"] for r in results)
    print(f"Repos covered: {len(repos)}")
    for repo in sorted(repos):
        count = sum(1 for r in results if r["repo"] == repo)
        print(f"  {repo}: {count}")

    n_with_docs = sum(1 for r in results if r.get("documentation"))
    print(f"\nSamples with Context7 docs: {n_with_docs}/{len(results)}")

    hall_counts = {"structural": 0, "behavioral": 0, "semantic": 0}
    for r in results:
        for h in r.get("hallucinations", []):
            t = h.get("type", "unknown")
            if t in hall_counts:
                hall_counts[t] += 1
    print(f"Hallucination variants: {hall_counts}")

    print(f"\nLettuceDetect training samples: {len(ld_samples)}")
    n_positive = sum(1 for s in ld_samples if s.get("labels"))
    n_negative = sum(1 for s in ld_samples if not s.get("labels"))
    print(f"  Hallucinated (positive): {n_positive}")
    print(f"  Correct (negative): {n_negative}")


if __name__ == "__main__":
    main()
