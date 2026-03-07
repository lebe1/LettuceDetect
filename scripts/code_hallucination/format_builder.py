"""Phase 5: Assign answer format to each instance.

Supports both sequential (remote API) and async batch (local vLLM) modes.
Set BATCH_SIZE>1 env var for parallel requests to local vLLM.
"""

import asyncio
import json
import random
import textwrap
import time

from openai import AsyncOpenAI, OpenAI

from .config import (
    API_BASE_URL,
    API_KEY,
    BATCH_SIZE,
    FORMAT_TYPES,
    FORMAT_WEIGHTS,
    FORMATS_PATH,
    LLM_TEMPERATURE,
    MAX_RETRIES,
    MODEL,
    RETRY_DELAY,
    SOURCE_CACHE_DIR,
    token_limit_kwargs,
)

EXPLANATION_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a helpful AI coding assistant (like Claude or Cursor).
    Given a user's coding question and the correct code fix, write a natural response
    that a developer would receive from an AI assistant.

    Your response MUST:
    - Start with 1-2 sentences explaining what was wrong and how to fix it
    - Include the code in a properly formatted code block (```python)
    - Do NOT add anything after the code block

    Your response must NOT:
    - Include phrases like "Here's the fix" or "I'll help you with that"
    - Be longer than 2 sentences of explanation + the code block
    - Change the code in any way — use it exactly as provided
    - Add any imports or code not in the original

    Example:
    The `process_data` function uses `dict.items()` instead of iterating over sorted keys, causing non-deterministic output.

    ```python
    def process_data(data):
        for key in sorted(data.keys()):
            yield key, data[key]
    ```
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
                **token_limit_kwargs(model, 200),
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


async def _generate_explanation_async(
    aclient: AsyncOpenAI, model: str, code: str, query: str, context: str
) -> str | None:
    """Async version of _generate_explanation for batch processing."""
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
            response = await aclient.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=LLM_TEMPERATURE,
                **token_limit_kwargs(model, 200),
            )
            result = response.choices[0].message.content.strip()
            if code[:50] in result or "```" in result:
                return result
            if attempt < MAX_RETRIES - 1:
                continue
            return None
        except Exception:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
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

    Uses async batch processing when BATCH_SIZE > 1 (for local vLLM).
    Falls back to sequential processing for remote APIs (BATCH_SIZE=1).
    """
    print("=" * 60)
    print("Phase 5: Answer Format Building")
    print("=" * 60)

    FORMATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if queries is None:
        queries = {}

    # Load existing for resumability
    existing = {}
    if FORMATS_PATH.exists():
        with open(FORMATS_PATH) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    existing[entry["instance_id"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"Already processed: {len(existing)} formats")

    to_process = [inst for inst in instances if inst["instance_id"] not in existing]
    print(f"Remaining: {len(to_process)} instances to process")
    print(f"Batch size: {BATCH_SIZE}")

    # First pass: assign formats for all instances (no LLM needed)
    # Collect which ones need explanation generation
    needs_explanation = []  # (instance_id, code, query, context)
    entries_no_llm = []  # entries that don't need LLM

    for inst in to_process:
        instance_id = inst["instance_id"]

        cache_path = source_cache_dir / f"{instance_id}.json"
        if not cache_path.exists():
            continue

        with open(cache_path) as fp:
            source_data = json.load(fp)

        fmt, answer = assign_format(source_data)
        if fmt is None:
            continue

        if fmt == "code_with_explanation":
            query = queries.get(instance_id, inst.get("problem_statement", "")[:500])
            context = source_data.get("patch_code", "")
            needs_explanation.append((instance_id, answer, query, context, fmt))
        else:
            entries_no_llm.append(
                {
                    "instance_id": instance_id,
                    "format_type": fmt,
                    "answer": answer,
                }
            )

    # Write non-LLM entries immediately
    results = list(existing.values())
    format_counts = {fmt: 0 for fmt in FORMAT_TYPES}
    for entry in results:
        fmt = entry.get("format_type")
        if fmt in format_counts:
            format_counts[fmt] += 1

    processed = 0
    explanation_failures = 0

    with open(FORMATS_PATH, "a") as f:
        for entry in entries_no_llm:
            f.write(json.dumps(entry) + "\n")
            results.append(entry)
            format_counts[entry["format_type"]] += 1
            processed += 1
        f.flush()

    print(f"  Assigned {len(entries_no_llm)} non-LLM formats")
    print(f"  Need LLM explanation: {len(needs_explanation)} instances")

    # Second pass: generate explanations (batched or sequential)
    if needs_explanation:
        if BATCH_SIZE > 1:
            explanation_failures = _run_explanations_batched(
                needs_explanation, format_counts, results, api_key, base_url, model
            )
        else:
            explanation_failures = _run_explanations_sequential(
                needs_explanation, format_counts, results, api_key, base_url, model
            )

    processed += len(needs_explanation)

    print(f"\nAssigned formats for {len(results)} instances")
    if explanation_failures:
        print(f"  Explanation generation failures (fell back to fragment): {explanation_failures}")
    for fmt, count in format_counts.items():
        pct = count * 100 // max(len(results), 1)
        print(f"  {fmt}: {count} ({pct}%)")

    return results


def _run_explanations_sequential(
    needs_explanation, format_counts, results, api_key, base_url, model
):
    """Generate explanations sequentially (for remote APIs)."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    explanation_failures = 0
    processed = 0

    with open(FORMATS_PATH, "a") as f:
        for instance_id, code, query, context, _ in needs_explanation:
            explained = _generate_explanation(client, model, code, query, context)

            if explained is None:
                fmt = "fragment"
                answer = code
                explanation_failures += 1
            else:
                fmt = "code_with_explanation"
                answer = explained

            entry = {
                "instance_id": instance_id,
                "format_type": fmt,
                "answer": answer,
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()
            results.append(entry)
            format_counts[fmt] += 1
            processed += 1

            if processed % 100 == 0:
                print(
                    f"  Phase 5 (explanations): {processed}/{len(needs_explanation)} "
                    f"(failures: {explanation_failures})"
                )

    return explanation_failures


def _run_explanations_batched(needs_explanation, format_counts, results, api_key, base_url, model):
    """Generate explanations with async batching (for local vLLM)."""
    aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)
    explanation_failures = 0
    processed = 0

    async def process_batches():
        nonlocal explanation_failures, processed

        with open(FORMATS_PATH, "a") as f:
            for batch_start in range(0, len(needs_explanation), BATCH_SIZE):
                batch = needs_explanation[batch_start : batch_start + BATCH_SIZE]

                tasks = []
                for instance_id, code, query, context, _ in batch:
                    tasks.append(_generate_explanation_async(aclient, model, code, query, context))

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for (instance_id, code, query, context, _), explained in zip(batch, batch_results):
                    if isinstance(explained, Exception) or explained is None:
                        fmt = "fragment"
                        answer = code
                        explanation_failures += 1
                    else:
                        fmt = "code_with_explanation"
                        answer = explained

                    entry = {
                        "instance_id": instance_id,
                        "format_type": fmt,
                        "answer": answer,
                    }
                    f.write(json.dumps(entry) + "\n")
                    results.append(entry)
                    format_counts[fmt] += 1
                    processed += 1

                f.flush()

                if processed % 100 == 0 or batch_start + BATCH_SIZE >= len(needs_explanation):
                    print(
                        f"  Phase 5 (explanations): {processed}/{len(needs_explanation)} "
                        f"(failures: {explanation_failures})"
                    )

    asyncio.run(process_batches())
    return explanation_failures


if __name__ == "__main__":
    from .swebench_loader import load_instances

    instances = load_instances()
    run(instances)
