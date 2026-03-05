"""Phase 6: Inject hallucinations using LLM with JSON span annotations.

Supports both sequential (remote API) and async batch (local vLLM) modes.
Set BATCH_SIZE>1 env var for parallel requests to local vLLM.
"""

import asyncio
import json
import re
import textwrap
import time

from openai import AsyncOpenAI, OpenAI

from .config import (
    API_BASE_URL,
    API_KEY,
    BATCH_SIZE,
    HALLUCINATED_PATH,
    HALLUCINATION_TEMPERATURE,
    HALLUCINATION_TYPES,
    MAX_RETRIES,
    MODEL,
    RETRY_DELAY,
)

INJECTION_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a code hallucination injector for building a hallucination detection dataset.

    Given a correct answer (which may be pure code OR code with natural language explanation)
    and context, create a hallucinated version with specific types of errors.

    Hallucination types:
    - STRUCTURAL: Change a function call, import, or parameter to something that
      doesn't exist or is wrong. Code should still parse but reference non-existent
      APIs, wrong methods, or invented parameters.
    - BEHAVIORAL: Use correct APIs but with wrong values or logic. Wrong defaults,
      off-by-one errors, swapped conditions, wrong argument values.
    - SEMANTIC: Code that looks like it addresses the user's request but does
      something subtly different or opposite. The code parses, uses real APIs,
      but fails to do what was asked. If library documentation is provided,
      you can make the code contradict the documented API (wrong parameter names,
      wrong return types, deprecated usage, etc.).
      For answers with explanations, you may also make the explanation contradict
      the code or describe incorrect behavior.

    Rules:
    - Make 2-3 DISTINCT changes spread across different parts of the answer
    - Each changed span must be 20-150 characters long (not too short, not too long)
    - Total hallucinated text must be LESS THAN 40% of the original answer length
    - Keep most of the answer CORRECT — do NOT rewrite the entire thing
    - Changes should be in different functions/blocks/paragraphs, not adjacent lines
    - Make changes PLAUSIBLE — something an LLM would realistically generate
    - Changes must be SUBTLE, not obviously broken
    - The code in the hallucinated answer must still be syntactically valid
    - Do NOT add comments explaining or hinting at the hallucination (no "# wrong",
      "# error", "# typo", "# nonexistent", etc.) — the errors must be invisible
      to someone skimming the answer
    - If the answer contains both code and explanation, inject errors in BOTH parts
      (e.g. wrong API in code + misleading description in text)
    - Preserve the overall structure: keep markdown formatting, code blocks, etc.

    Respond in this exact JSON format (no markdown, no code blocks):
    {
        "hallucinated_code": "the full modified answer with hallucinations injected",
        "changes": [
            {
                "original": "exact original text that was changed",
                "hallucinated": "what you changed it to",
                "explanation": "why this is a hallucination"
            }
        ]
    }

    IMPORTANT:
    - You MUST include 2-3 changes in the "changes" array
    - "original" must be an exact substring of the correct answer
    - "hallucinated" must be an exact substring of your hallucinated answer
    - Each "hallucinated" value must be at least 20 characters long
    - Return ONLY valid JSON, nothing else
""")


def inject_hallucination(
    client: OpenAI,
    model: str,
    clean_answer: str,
    hall_type: str,
    user_query: str = "",
    context: str = "",
    documentation: dict[str, str] | None = None,
) -> dict | None:
    """Inject a hallucination and get back structured JSON with spans.

    Returns dict with 'hallucinated_code' and 'changes', or None if failed.
    """
    docs_section = ""
    if documentation:
        docs_parts = [f"Documentation for {lib}:\n{doc}" for lib, doc in documentation.items()]
        docs_section = (
            "\n\nLibrary documentation (the hallucination could contradict this):\n"
            + "\n\n".join(docs_parts)
        )

    user_msg = f"""Hallucination type to inject: {hall_type.upper()}

User's original request: {user_query}

Context (source code):
{context}{docs_section}

Correct code to modify:
{clean_answer}

Generate a hallucinated version with {hall_type} error(s). Return JSON only."""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": INJECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=HALLUCINATION_TEMPERATURE,
                max_tokens=4000,
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON from response
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if not json_match:
                if attempt < MAX_RETRIES - 1:
                    continue
                return None

            result = json.loads(json_match.group())

            if "hallucinated_code" not in result or "changes" not in result:
                if attempt < MAX_RETRIES - 1:
                    continue
                return None

            # Verify the hallucinated code is actually different
            if result["hallucinated_code"].strip() == clean_answer.strip():
                if attempt < MAX_RETRIES - 1:
                    continue
                return None

            return result

        except (json.JSONDecodeError, Exception) as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  Injection error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return None


def compute_span_offsets(hallucinated_code: str, hallucinated_span: str) -> list[dict]:
    """Find character offsets of a hallucinated span within the answer code."""
    spans = []
    idx = hallucinated_code.find(hallucinated_span)
    if idx != -1:
        spans.append({"start": idx, "end": idx + len(hallucinated_span)})
    return spans


def build_labels_from_changes(
    hallucinated_code: str, changes: list[dict], hall_type: str
) -> list[dict]:
    """Build span labels by finding each hallucinated string in the code.

    Only includes spans where the hallucinated text is actually found in the answer.
    """
    labels = []
    for change in changes:
        h_span = change.get("hallucinated", "")
        if not h_span or len(h_span) < 15:
            continue
        if h_span not in hallucinated_code:
            continue

        offsets = compute_span_offsets(hallucinated_code, h_span)
        for offset in offsets[:1]:  # First occurrence only
            labels.append(
                {
                    "start": offset["start"],
                    "end": offset["end"],
                    "label": hall_type,
                }
            )

    return labels


def load_existing_hallucinations(path=HALLUCINATED_PATH) -> dict[str, dict]:
    """Load already-processed hallucinations for resumability."""
    existing = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    existing[entry["instance_id"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
    return existing


async def _inject_one_async(
    aclient: AsyncOpenAI,
    model: str,
    clean_answer: str,
    hall_type: str,
    user_query: str,
    context: str,
    documentation: dict[str, str] | None = None,
) -> dict | None:
    """Async version of inject_hallucination for batch processing."""
    docs_section = ""
    if documentation:
        docs_parts = [f"Documentation for {lib}:\n{doc}" for lib, doc in documentation.items()]
        docs_section = (
            "\n\nLibrary documentation (the hallucination could contradict this):\n"
            + "\n\n".join(docs_parts)
        )

    user_msg = f"""Hallucination type to inject: {hall_type.upper()}

User's original request: {user_query}

Context (source code):
{context}{docs_section}

Correct code to modify:
{clean_answer}

Generate a hallucinated version with {hall_type} error(s). Return JSON only."""

    for attempt in range(MAX_RETRIES):
        try:
            response = await aclient.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": INJECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=HALLUCINATION_TEMPERATURE,
                max_tokens=4000,
            )
            raw = response.choices[0].message.content.strip()
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if not json_match:
                continue
            result = json.loads(json_match.group())
            if "hallucinated_code" not in result or "changes" not in result:
                continue
            if result["hallucinated_code"].strip() == clean_answer.strip():
                continue
            return result
        except Exception:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return None
    return None


def _validate_labels(hallucinated_code: str, labels: list[dict]) -> tuple[bool, str]:
    """Validate that hallucination labels meet quality thresholds.

    :return: (is_valid, reason) tuple.
    """
    if not labels:
        return False, "no_labels"

    total_span = sum(lab["end"] - lab["start"] for lab in labels)
    code_len = len(hallucinated_code) if hallucinated_code else 1
    coverage = total_span / code_len

    if coverage > 0.60:
        return False, f"coverage_too_high ({coverage:.0%})"

    for lab in labels:
        span_len = lab["end"] - lab["start"]
        if span_len < 15:
            return False, f"span_too_short ({span_len} chars)"

    return True, ""


def _process_result(result, instance_id, hall_type, fmt_data, model):
    """Process a single injection result into a JSONL entry."""
    if result is None:
        return None
    hallucinated_code = result["hallucinated_code"]
    changes = result.get("changes", [])
    labels = build_labels_from_changes(hallucinated_code, changes, hall_type)

    valid, reason = _validate_labels(hallucinated_code, labels)
    if not valid:
        return None

    return {
        "instance_id": instance_id,
        "hallucinated_answer": hallucinated_code,
        "labels": labels,
        "hallucination_type": hall_type,
        "injector_model": model,
        "format_type": fmt_data.get("format_type", "fragment"),
    }


def run(
    instances_to_inject: list[dict],
    formats: dict[str, dict],
    queries: dict[str, str],
    docs: dict[str, dict] | None = None,
    api_key: str = API_KEY,
    base_url: str = API_BASE_URL,
    model: str = MODEL,
):
    """Run Phase 6: Inject hallucinations into selected instances.

    Uses async batch processing when BATCH_SIZE > 1 (for local vLLM).
    Falls back to sequential processing for remote APIs (BATCH_SIZE=1).
    """
    print("=" * 60)
    print("Phase 6: Hallucination Injection")
    print("=" * 60)

    if docs is None:
        docs = {}

    HALLUCINATED_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using {base_url} with model {model}")
    print(f"Batch size: {BATCH_SIZE}")

    existing = load_existing_hallucinations()
    print(f"Already processed: {len(existing)}")

    to_process = [
        inst
        for inst in instances_to_inject
        if inst["instance_id"] not in existing and inst["instance_id"] in formats
    ]
    print(f"Remaining: {len(to_process)} instances to inject")

    if BATCH_SIZE > 1:
        results = _run_batched(to_process, formats, queries, docs, api_key, base_url, model)
    else:
        results = _run_sequential(to_process, formats, queries, docs, api_key, base_url, model)

    # Stats
    type_counts = {}
    for r in results:
        t = r["hallucination_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print("By type:", type_counts)

    if results:
        avg_spans = sum(len(r["labels"]) for r in results) / len(results)
        span_sizes = [lab["end"] - lab["start"] for r in results for lab in r["labels"]]
        print(f"Avg spans per sample: {avg_spans:.1f}")
        print(
            f"Span sizes: min={min(span_sizes)}, max={max(span_sizes)}, avg={sum(span_sizes) // len(span_sizes)}"
        )

    return results


def _run_sequential(to_process, formats, queries, docs, api_key, base_url, model):
    """Sequential processing for remote APIs (rate-limited)."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    processed = 0
    failed = 0
    no_spans = 0
    results = []

    with open(HALLUCINATED_PATH, "a") as f:
        for i, inst in enumerate(to_process):
            instance_id = inst["instance_id"]
            fmt_data = formats.get(instance_id, {})
            clean_answer = fmt_data.get("answer", "")
            if not clean_answer:
                failed += 1
                continue

            hall_type = HALLUCINATION_TYPES[i % len(HALLUCINATION_TYPES)]
            query = queries.get(instance_id, "")
            context = inst.get("problem_statement", "")
            instance_docs = docs.get(instance_id, {})

            # Try injection with up to 2 quality retries
            entry = None
            for attempt in range(3):
                result = inject_hallucination(
                    client,
                    model,
                    clean_answer,
                    hall_type,
                    query,
                    context,
                    documentation=instance_docs,
                )
                entry = _process_result(result, instance_id, hall_type, fmt_data, model)
                if entry is not None:
                    break

            if entry is None:
                if result is not None:
                    no_spans += 1
                failed += 1
                continue

            f.write(json.dumps(entry) + "\n")
            f.flush()
            results.append(entry)
            processed += 1

            if processed % 50 == 0:
                print(f"  Progress: {processed}/{len(to_process)} (failed: {failed})")

    print(f"\nDone: {processed} injected, {failed} failed ({no_spans} had no matchable spans)")
    return results


def _run_batched(to_process, formats, queries, docs, api_key, base_url, model):
    """Async batch processing for local vLLM (no rate limiting needed)."""
    aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)
    processed = 0
    failed = 0
    no_spans = 0
    results = []

    async def process_batches():
        nonlocal processed, failed, no_spans

        with open(HALLUCINATED_PATH, "a") as f:
            for batch_start in range(0, len(to_process), BATCH_SIZE):
                batch = to_process[batch_start : batch_start + BATCH_SIZE]

                # Build async tasks for the batch
                tasks = []
                batch_meta = []
                for i, inst in enumerate(batch):
                    global_idx = batch_start + i
                    instance_id = inst["instance_id"]
                    fmt_data = formats.get(instance_id, {})
                    clean_answer = fmt_data.get("answer", "")
                    if not clean_answer:
                        failed += 1
                        continue

                    hall_type = HALLUCINATION_TYPES[global_idx % len(HALLUCINATION_TYPES)]
                    query = queries.get(instance_id, "")
                    context = inst.get("problem_statement", "")
                    instance_docs = docs.get(instance_id, {})

                    tasks.append(
                        _inject_one_async(
                            aclient,
                            model,
                            clean_answer,
                            hall_type,
                            query,
                            context,
                            documentation=instance_docs,
                        )
                    )
                    batch_meta.append((instance_id, hall_type, fmt_data))

                if not tasks:
                    continue

                # Run batch concurrently
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and write immediately
                for result, (instance_id, hall_type, fmt_data) in zip(batch_results, batch_meta):
                    if isinstance(result, Exception):
                        failed += 1
                        continue

                    entry = _process_result(result, instance_id, hall_type, fmt_data, model)
                    if entry is None:
                        if result is not None:
                            no_spans += 1
                        failed += 1
                        continue

                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    results.append(entry)
                    processed += 1

                if processed % 50 == 0 or batch_start + BATCH_SIZE >= len(to_process):
                    total = processed + failed
                    print(
                        f"  Progress: {total}/{len(to_process)} ({processed} ok, {failed} failed)"
                    )

    asyncio.run(process_batches())
    print(f"\nDone: {processed} injected, {failed} failed ({no_spans} had no matchable spans)")
    return results


if __name__ == "__main__":
    print("Run via pipeline.py")
