"""Phase 6: Inject hallucinations using LLM with JSON span annotations."""

import json
import re
import textwrap
import time

from openai import OpenAI

from .config import (
    API_BASE_URL,
    API_KEY,
    HALLUCINATED_PATH,
    HALLUCINATION_TEMPERATURE,
    HALLUCINATION_TYPES,
    MAX_RETRIES,
    MODEL,
    RETRY_DELAY,
)

INJECTION_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a code hallucination injector for building a hallucination detection dataset.

    Given correct code and context, create a hallucinated version with a specific type of error.

    Hallucination types:
    - STRUCTURAL: Change a function call, import, or parameter to something that
      doesn't exist or is wrong. Code should still parse but reference non-existent
      APIs, wrong methods, or invented parameters.
    - BEHAVIORAL: Use correct APIs but with wrong values or logic. Wrong defaults,
      off-by-one errors, swapped conditions, wrong argument values.
    - SEMANTIC: Code that looks like it addresses the user's request but does
      something subtly different or opposite. The code parses, uses real APIs,
      but fails to do what was asked.

    Rules:
    - Make changes PLAUSIBLE - something an LLM would realistically generate
    - Changes must be SUBTLE, not obviously broken
    - The hallucinated code must still be syntactically valid
    - Make 1-3 changes, not more

    Respond in this exact JSON format (no markdown, no code blocks):
    {
        "hallucinated_code": "the full modified code with hallucinations injected",
        "changes": [
            {
                "original": "exact original code that was changed",
                "hallucinated": "what you changed it to",
                "explanation": "why this is a hallucination"
            }
        ]
    }

    IMPORTANT:
    - "original" must be an exact substring of the correct code
    - "hallucinated" must be an exact substring of your hallucinated_code
    - Return ONLY valid JSON, nothing else
""")


def inject_hallucination(
    client: OpenAI,
    model: str,
    clean_answer: str,
    hall_type: str,
    user_query: str = "",
    context: str = "",
) -> dict | None:
    """Inject a hallucination and get back structured JSON with spans.

    Returns dict with 'hallucinated_code' and 'changes', or None if failed.
    """
    user_msg = f"""Hallucination type to inject: {hall_type.upper()}

User's original request: {user_query[:500]}

Context (source code):
{context[:2000]}

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
        if not h_span or len(h_span) < 3:
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


def run(
    instances_to_inject: list[dict],
    formats: dict[str, dict],
    queries: dict[str, str],
    api_key: str = API_KEY,
    base_url: str = API_BASE_URL,
    model: str = MODEL,
):
    """Run Phase 6: Inject hallucinations into selected instances."""
    print("=" * 60)
    print("Phase 6: Hallucination Injection")
    print("=" * 60)

    HALLUCINATED_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Using {base_url} with model {model}")

    existing = load_existing_hallucinations()
    print(f"Already processed: {len(existing)}")

    to_process = [
        inst
        for inst in instances_to_inject
        if inst["instance_id"] not in existing and inst["instance_id"] in formats
    ]
    print(f"Remaining: {len(to_process)} instances to inject")

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

            # Round-robin hall type
            hall_type = HALLUCINATION_TYPES[i % len(HALLUCINATION_TYPES)]

            query = queries.get(instance_id, "")
            context = inst.get("problem_statement", "")[:2000]

            result = inject_hallucination(client, model, clean_answer, hall_type, query, context)

            if result is None:
                failed += 1
                continue

            hallucinated_code = result["hallucinated_code"]
            changes = result.get("changes", [])

            # Build labels by matching hallucinated spans in the code
            labels = build_labels_from_changes(hallucinated_code, changes, hall_type)

            if not labels:
                no_spans += 1
                failed += 1
                continue

            entry = {
                "instance_id": instance_id,
                "hallucinated_answer": hallucinated_code,
                "labels": labels,
                "hallucination_type": hall_type,
                "injector_model": model,
                "format_type": fmt_data.get("format_type", "fragment"),
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()

            results.append(entry)
            processed += 1

            if processed % 50 == 0:
                print(f"  Progress: {processed}/{len(to_process)} (failed: {failed})")

            time.sleep(0.5)

    print(f"\nDone: {processed} injected, {failed} failed ({no_spans} had no matchable spans)")

    # Stats
    type_counts = {}
    for r in results:
        t = r["hallucination_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print("By type:", type_counts)

    if results:
        avg_spans = sum(len(r["labels"]) for r in results) / len(results)
        span_sizes = [l["end"] - l["start"] for r in results for l in r["labels"]]
        print(f"Avg spans per sample: {avg_spans:.1f}")
        print(
            f"Span sizes: min={min(span_sizes)}, max={max(span_sizes)}, avg={sum(span_sizes) // len(span_sizes)}"
        )

    return results


if __name__ == "__main__":
    print("Run via pipeline.py")
