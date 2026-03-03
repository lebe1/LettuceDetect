"""Phase 3: Rewrite problem statements into natural user queries via LLM."""

import json
import textwrap
import time

from openai import OpenAI

from .config import (
    API_BASE_URL,
    API_KEY,
    LLM_TEMPERATURE,
    MAX_RETRIES,
    MODEL,
    QUERIES_PATH,
    RETRY_DELAY,
)

REWRITE_SYSTEM_PROMPT = textwrap.dedent("""\
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


def get_client(api_key: str = API_KEY, base_url: str = API_BASE_URL) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def llm_call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = 300,
) -> str:
    """Make an LLM call with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
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


def rewrite_query(client: OpenAI, model: str, problem_statement: str, repo: str) -> str:
    """Rewrite a problem statement into a natural user query."""
    user_msg = f"Repository: {repo}\n\nGitHub Issue:\n{problem_statement[:3000]}"
    return llm_call(client, model, REWRITE_SYSTEM_PROMPT, user_msg)


def load_existing_queries(path=QUERIES_PATH) -> dict[str, str]:
    """Load already-processed queries for resumability."""
    existing = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    existing[entry["instance_id"]] = entry["query"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return existing


def run(
    instances: list[dict],
    api_key: str = API_KEY,
    base_url: str = API_BASE_URL,
    model: str = MODEL,
):
    """Run Phase 3: Rewrite all queries."""
    print("=" * 60)
    print("Phase 3: Query Rewriting")
    print("=" * 60)

    QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = get_client(api_key, base_url)
    print(f"Using {base_url} with model {model}")

    # Load existing for resumability
    existing = load_existing_queries()
    print(f"Already processed: {len(existing)} queries")

    to_process = [inst for inst in instances if inst["instance_id"] not in existing]
    print(f"Remaining: {len(to_process)} queries to process")

    processed = 0
    failed = 0

    with open(QUERIES_PATH, "a") as f:
        for i, inst in enumerate(to_process):
            instance_id = inst["instance_id"]
            repo = inst["repo"]
            problem = inst["problem_statement"]

            try:
                query = rewrite_query(client, model, problem, repo)
                entry = {"instance_id": instance_id, "query": query}
                f.write(json.dumps(entry) + "\n")
                f.flush()
                processed += 1

                if processed % 50 == 0:
                    print(f"  Progress: {processed}/{len(to_process)} (failed: {failed})")

                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"  ERROR {instance_id}: {e}")
                failed += 1
                time.sleep(2)

    print(f"\nDone: {processed} new queries, {failed} failed")
    total = len(existing) + processed
    print(f"Total queries: {total}")


if __name__ == "__main__":
    from .swebench_loader import load_instances

    instances = load_instances()
    run(instances)
