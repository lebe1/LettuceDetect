"""Phase 5: Assign answer format to each instance."""

import json
import random
import textwrap
import time

from openai import OpenAI

from .config import (
    API_BASE_URL,
    API_KEY,
    FORMAT_TYPES,
    FORMAT_WEIGHTS,
    FORMATS_PATH,
    LLM_TEMPERATURE,
    MAX_RETRIES,
    MODEL,
    RETRY_DELAY,
    SOURCE_CACHE_DIR,
)

EXPLANATION_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a helpful AI coding assistant (like Claude or Cursor).
    Given a user's coding question and the correct code fix, write a natural response
    that a developer would receive from an AI assistant.

    Your response MUST:
    - Start with a brief explanation (1-3 sentences) of what the issue is and how to fix it
    - Include the code in a properly formatted code block (```python)
    - Optionally end with a short note about what changed or why

    Your response must NOT:
    - Include phrases like "Here's the fix" or "I'll help you with that" — just explain directly
    - Be longer than necessary — keep it concise
    - Change the code in any way — use it exactly as provided
    - Add any imports or code not in the original

    Example style:
    The issue is that `process_data` uses `dict.items()` instead of iterating
    over the sorted keys, which causes non-deterministic output.

    ```python
    def process_data(data):
        for key in sorted(data.keys()):
            yield key, data[key]
    ```

    This ensures consistent ordering regardless of insertion order.
""")


def _generate_explanation(
    client: OpenAI, model: str, code: str, query: str, context: str
) -> str | None:
    """Use LLM to wrap code in a natural explanation."""
    user_msg = f"""User's question: {query}

Context (relevant source code):
{context[:3000]}

Correct code fix:
```python
{code}
```

Write a natural AI assistant response that includes this exact code."""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=2000,
            )
            result = response.choices[0].message.content.strip()
            # Verify the code is actually in the response
            if code[:50] in result or "```" in result:
                return result
            if attempt < MAX_RETRIES - 1:
                continue
            return None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  Explanation error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return None
    return None


def assign_format(source_data: dict) -> tuple[str, str]:
    """Assign a format type and build the answer for an instance.

    Returns (format_type, answer_text).
    Falls back if preferred format isn't available.

    Note: code_with_explanation is handled separately since it needs LLM calls.
    This function returns ("code_with_explanation", base_code) and the caller
    wraps it with an explanation.
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

    # code_with_explanation can use any base format
    all_available = available + ["code_with_explanation"]

    # Weighted random choice from available formats
    weights = []
    for fmt in all_available:
        idx = FORMAT_TYPES.index(fmt)
        weights.append(FORMAT_WEIGHTS[idx])

    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]

    chosen = random.choices(all_available, weights=weights, k=1)[0]

    # Build answer text
    if chosen == "code_with_explanation":
        # Pick the best base code to wrap with explanation
        if has_functions:
            funcs = source_data["modified_functions"]
            func = max(funcs, key=lambda f: len(f.get("patched", "")))
            answer = func["patched"]
        elif has_fragment:
            answer = source_data["patch_code"]
        elif has_edit:
            answer = source_data["edit_style"]
        return "code_with_explanation", answer
    elif chosen == "complete_function":
        funcs = source_data["modified_functions"]
        func = max(funcs, key=lambda f: len(f.get("patched", "")))
        answer = func["patched"]
    elif chosen == "edit_style":
        answer = source_data["edit_style"]
    else:  # fragment
        answer = source_data["patch_code"]

    return chosen, answer


def run(
    instances: list[dict],
    source_cache_dir=SOURCE_CACHE_DIR,
    api_key: str = API_KEY,
    base_url: str = API_BASE_URL,
    model: str = MODEL,
    queries: dict[str, str] | None = None,
):
    """Run Phase 5: Assign formats and build answers.

    Returns list of dicts with instance_id, format_type, answer.
    """
    print("=" * 60)
    print("Phase 5: Answer Format Building")
    print("=" * 60)

    FORMATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if queries is None:
        queries = {}

    # Only init LLM client if we'll need it (lazy)
    client = None

    results = []
    format_counts = {fmt: 0 for fmt in FORMAT_TYPES}
    skipped = 0
    explanation_failures = 0

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

        # Generate explanation wrapper for code_with_explanation format
        if fmt == "code_with_explanation":
            if client is None:
                client = OpenAI(api_key=api_key, base_url=base_url)
                print(f"  LLM client initialized for code_with_explanation ({base_url})")

            query = queries.get(instance_id, inst.get("problem_statement", "")[:500])
            context = source_data.get("patch_code", "")
            explained = _generate_explanation(client, model, answer, query, context)

            if explained is None:
                # Fallback: use raw code as fragment
                fmt = "fragment"
                explanation_failures += 1
            else:
                answer = explained

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
    if explanation_failures:
        print(f"  Explanation generation failures (fell back to fragment): {explanation_failures}")
    for fmt, count in format_counts.items():
        pct = count * 100 // max(len(results), 1)
        print(f"  {fmt}: {count} ({pct}%)")

    return results


if __name__ == "__main__":
    from .swebench_loader import load_instances

    instances = load_instances()
    run(instances)
