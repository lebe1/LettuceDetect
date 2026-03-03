#!/usr/bin/env python3
"""Prototype: Transform SWE-bench into a code hallucination detection dataset.

Pipeline:
1. Load SWE-bench sample (issue, repo, patch)
2. Extract context files from the patch (what files were relevant)
3. Transform issue into user-style query (via LLM)
4. Inject hallucinations into the gold patch (structural + semantic)
5. Output annotated samples
"""

import json
import os
import re
import textwrap
from typing import Any

from openai import OpenAI

# Groq API (OpenAI-compatible)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "moonshotai/kimi-k2-instruct-0905"


def get_client() -> OpenAI:
    """Get Groq client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY to your Groq API key")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def extract_changed_files(patch: str) -> list[str]:
    """Extract file paths that were changed from a unified diff patch."""
    files = []
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            # Extract b/ path (the destination file)
            match = re.search(r"b/(.+)$", line)
            if match:
                files.append(match.group(1))
    return files


def extract_patch_changes(patch: str) -> list[dict]:
    """Parse a unified diff into structured changes per file."""
    file_diffs = []
    current_file = None
    current_hunks = []

    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            if current_file:
                file_diffs.append({"file": current_file, "diff": "\n".join(current_hunks)})
            match = re.search(r"b/(.+)$", line)
            current_file = match.group(1) if match else "unknown"
            current_hunks = [line]
        elif current_file:
            current_hunks.append(line)

    if current_file:
        file_diffs.append({"file": current_file, "diff": "\n".join(current_hunks)})

    return file_diffs


def transform_to_user_query(client: OpenAI, problem_statement: str, repo: str) -> str:
    """Transform a GitHub issue into a realistic user-to-agent query."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": textwrap.dedent("""\
                    You transform GitHub issue descriptions into realistic user queries
                    that a developer would type into an AI coding assistant (like Claude Code
                    or Cursor).

                    Rules:
                    - Make it conversational and natural, like someone typing into a chat
                    - Keep the core technical ask but remove GitHub-specific formatting
                    - Remove reproduction steps, stack traces, and verbose details
                    - Keep it to 1-3 sentences
                    - Don't mention "issue" or "bug report"
                    - Make it sound like someone asking for help, not filing a report

                    Examples:
                    Issue: "BUG: DataFrame.merge raises TypeError when merging on columns with different dtypes"
                    Query: "pd.merge is crashing with a TypeError when I try to merge two dataframes where the join columns have different dtypes. Can you fix the merge to handle dtype coercion?"

                    Issue: "Enable quiet mode/no-verbose in CLI for programmatic use"
                    Query: "Can you add a --quiet flag to the CLI? I need to use it in scripts and the verbose output is getting in the way."
                """),
            },
            {
                "role": "user",
                "content": f"Repository: {repo}\n\nGitHub Issue:\n{problem_statement[:3000]}",
            },
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def inject_hallucinations(
    client: OpenAI,
    gold_patch: str,
    user_query: str,
    problem_statement: str,
    repo: str,
) -> dict[str, Any]:
    """Inject hallucinations into a gold patch, returning annotated hallucinated code."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": textwrap.dedent("""\
                    You are a code hallucination injector for building a hallucination detection dataset.

                    Given a correct code patch and the user's query, create THREE hallucinated versions:

                    1. STRUCTURAL hallucination: Change a function call, import, or parameter to
                       something that doesn't exist or is wrong for this codebase. The code should
                       still parse but reference non-existent APIs, wrong method names, or invented
                       parameters.

                    2. BEHAVIORAL hallucination: Use correct APIs but with wrong values or logic
                       that would produce incorrect behavior. For example, wrong default values,
                       off-by-one errors, or swapped conditions.

                    3. SEMANTIC hallucination: Code that looks like it addresses the user's request
                       but actually does something subtly different or opposite. This is the hardest
                       type - the code should parse, use real APIs, but fail to do what was asked.
                       Examples: implementing global when per-item was asked, catching exceptions
                       instead of fixing the root cause, or implementing the inverse logic.

                    For EACH hallucinated version, respond in this exact JSON format:
                    {
                        "hallucinations": [
                            {
                                "type": "structural",
                                "hallucinated_patch": "the full modified patch",
                                "changes": [
                                    {
                                        "original": "the original correct code span",
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
                    - The hallucinated patches must be plausible — something an LLM would realistically generate
                    - Each change must be subtle, not obviously broken
                    - Return ONLY valid JSON, no markdown code blocks
                """),
            },
            {
                "role": "user",
                "content": f"""Repository: {repo}

User's query: {user_query}

Original issue context: {problem_statement[:2000]}

Correct gold patch:
{gold_patch}

Generate three hallucinated versions of this patch.""",
            },
        ],
        temperature=0.8,
        max_tokens=4000,
    )

    raw = response.choices[0].message.content.strip()
    # Try to extract JSON from the response
    # Sometimes LLMs wrap it in ```json blocks
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return {"raw_response": raw, "parse_error": True}
    return {"raw_response": raw, "parse_error": True}


def process_sample(client: OpenAI, sample: dict) -> dict:
    """Process a single SWE-bench sample into a hallucination detection sample."""
    instance_id = sample["instance_id"]
    repo = sample["repo"]
    problem_statement = sample["problem_statement"]
    gold_patch = sample["patch"]

    print(f"\n{'=' * 60}")
    print(f"Processing: {instance_id}")
    print(f"Repo: {repo}")
    print(f"{'=' * 60}")

    # Step 1: Extract changed files from the patch
    changed_files = extract_changed_files(gold_patch)
    print(f"\nChanged files: {changed_files}")

    # Step 2: Transform issue into user query
    print("\nTransforming issue into user query...")
    user_query = transform_to_user_query(client, problem_statement, repo)
    print(f"User query: {user_query}")

    # Step 3: Inject hallucinations
    print("\nInjecting hallucinations...")
    hallucination_result = inject_hallucinations(
        client, gold_patch, user_query, problem_statement, repo
    )

    if hallucination_result.get("parse_error"):
        print("WARNING: Failed to parse hallucination response")
        print(
            f"Raw response (first 500 chars): {hallucination_result.get('raw_response', '')[:500]}"
        )

    # Build the output sample
    output = {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": sample["base_commit"],
        "original_problem_statement": problem_statement,
        "user_query": user_query,
        "changed_files": changed_files,
        "gold_patch": gold_patch,
        "test_patch": sample["test_patch"],
        "fail_to_pass": sample["FAIL_TO_PASS"],
        "hallucinations": hallucination_result.get("hallucinations", []),
    }

    # Print summary
    if not hallucination_result.get("parse_error"):
        for h in hallucination_result.get("hallucinations", []):
            print(f"\n  [{h.get('type', 'unknown').upper()}]")
            for c in h.get("changes", []):
                print(f"    Original:      {c.get('original', 'N/A')[:80]}")
                print(f"    Hallucinated:  {c.get('hallucinated', 'N/A')[:80]}")
                print(f"    Explanation:   {c.get('explanation', 'N/A')[:100]}")

    return output


def main():
    """Run prototype on a few SWE-bench Lite samples."""
    from datasets import load_dataset

    client = get_client()

    # Load SWE-bench Lite (300 curated samples)
    print("Loading SWE-bench Lite...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    # Pick a few diverse samples to test
    # Choose samples from different repos with different patch sizes
    test_indices = [0, 5, 10]  # Start with 3 samples
    results = []

    for idx in test_indices:
        sample = ds[idx]
        try:
            result = process_sample(client, sample)
            results.append(result)
        except Exception as e:
            print(f"\nERROR processing {sample['instance_id']}: {e}")
            continue

    # Save results
    output_path = "/Users/adamkovacs/projects/LettuceDetect/data/code_hallucination_prototype.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nSaved {len(results)} samples to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n{r['instance_id']} ({r['repo']})")
        print(f"  Query: {r['user_query'][:100]}...")
        print(f"  Files: {r['changed_files']}")
        n_hall = len(r.get("hallucinations", []))
        print(f"  Hallucinations generated: {n_hall}")
        for h in r.get("hallucinations", []):
            print(f"    - {h.get('type', 'unknown')}: {len(h.get('changes', []))} changes")


if __name__ == "__main__":
    main()
