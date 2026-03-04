# Configuration

All pipeline configuration is centralized in `scripts/code_hallucination/config.py`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (none) | API key for the LLM provider |
| `API_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible API endpoint |
| `MODEL` | `moonshotai/kimi-k2-instruct-0905` | Model name |
| `BATCH_SIZE` | `1` | Concurrent requests. Set >1 for local vLLM to saturate GPU |
| `CONTEXT7_API_KEY` | (none) | API key for Context7 documentation service |

These can also be overridden via CLI flags (`--api-key`, `--base-url`, `--model`).

## Dataset Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HALLUCINATION_RATIO` | `0.4` | Fraction of instances that get hallucination injection |
| `DOCS_RATIO` | `0.5` | Fraction of instances that get Context7 documentation |
| `MAX_FILE_CHARS` | `12000` | Maximum characters per source file |
| `MAX_CONTEXT7_CHARS` | `4000` | Maximum characters per library doc |
| `LLM_TEMPERATURE` | `0.7` | Temperature for query rewriting |
| `HALLUCINATION_TEMPERATURE` | `0.8` | Temperature for hallucination injection (higher for variety) |
| `MAX_RETRIES` | `3` | API retry attempts |
| `RETRY_DELAY` | `2.0` | Base delay between retries (seconds) |

## Answer Format Weights

| Format | Weight | Description |
|--------|--------|-------------|
| `complete_function` | 0.4 | Full patched function body via AST |
| `edit_style` | 0.3 | "In file X, replace Y with Z" |
| `fragment` | 0.3 | Added/changed lines from diff |

## Hallucination Types

Assigned round-robin across injected instances:

- **structural** — Non-existent APIs, wrong methods, invented parameters
- **behavioral** — Wrong values, logic errors, swapped conditions
- **semantic** — Code that looks correct but does something subtly different

## File Paths

All data is stored under `data/code_hallucination/`:

| Path | Description |
|------|-------------|
| `swebench_instances.json` | Phase 1: loaded instances |
| `repos/` | Phase 2: bare git clones |
| `source_cache/` | Phase 2: per-instance source data |
| `queries.jsonl` | Phase 3: rewritten queries |
| `documentation.jsonl` | Phase 4: library docs |
| `formats.jsonl` | Phase 5: assigned formats |
| `hallucinated_samples.jsonl` | Phase 6: injected hallucinations |
| `code_hallucination_data.json` | Phase 7: final dataset |
| `code_hallucination_metadata.json` | Phase 7: metadata |
| `validation_report.txt` | Phase 9: quality report |

## Data Sources

| Source | Dataset ID |
|--------|-----------|
| SWE-bench (full) | `princeton-nlp/SWE-bench` |
| SWE-bench Lite | `princeton-nlp/SWE-bench_Lite` |
