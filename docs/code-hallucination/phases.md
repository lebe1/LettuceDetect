# Pipeline Phases

Detailed documentation for each of the 9 pipeline phases.

## Phase 1: Load SWE-bench

**Module:** `swebench_loader.py`

Loads all SWE-bench splits from HuggingFace and tags each instance with split and Lite membership.

| Split | Instances | Repos | Source |
|-------|-----------|-------|--------|
| Train | 19,008 | 35 | `princeton-nlp/SWE-bench` train |
| Dev | 225 | 6 | `princeton-nlp/SWE-bench` dev |
| Test | 2,294 | 12 | `princeton-nlp/SWE-bench` test |
| Lite | 300 | 12 | `princeton-nlp/SWE-bench_Lite` (subset of test) |

**Key function:** `load_all_splits() -> list[dict]`

Each instance includes: `instance_id`, `repo`, `base_commit`, `patch`, `problem_statement`, `split`, `is_lite`.

**Output:** `data/code_hallucination/swebench_instances.json`

---

## Phase 2: Fetch Sources

**Module:** `source_fetcher.py`

Clones repositories and extracts source code at the base commit for each instance. Builds three answer format variants.

### Strategy

- **Default:** Clone repos as bare git repos to `data/code_hallucination/repos/`. Use `git show {commit}:{path}` for instant file access.
- **Test mode:** Use GitHub raw API (`raw.githubusercontent.com`) — slower but no cloning needed.
- **Fallback:** If cloning fails, automatically falls back to GitHub API.

### What it extracts per instance

| Field | Description |
|-------|-------------|
| `changed_files` | File paths modified by the gold patch |
| `source_files` | Original source code at base commit |
| `patch_code` | Added/changed lines from the diff (fragment format) |
| `edit_style` | "In file X, replace Y with Z" format |
| `modified_functions` | AST-extracted functions that changed (complete function format) |

### Key functions

- `extract_changed_files(patch)` — Parse unified diff for file paths (anchored regex, not `lstrip("b/")`)
- `clone_repo(repo)` — `git clone --bare` with 30min timeout
- `fetch_file_at_commit(repo_dir, commit, filepath)` — `git show` for file contents
- `apply_patch_and_get_file(repo_dir, commit, patch, filepath)` — Apply patch in temp worktree
- `extract_modified_functions(original, patched)` — AST-based function diff

**Output:** `data/code_hallucination/source_cache/{instance_id}.json`

---

## Phase 3: Rewrite Queries

**Module:** `query_rewriter.py`

Transforms raw GitHub issue `problem_statement` fields into natural developer queries using an LLM.

### Example

**Before (raw issue):**
> BUG: DataFrame.groupby with as_index=False gives wrong result when grouping by single column with duplicate name. Steps to reproduce: ...

**After (rewritten):**
> I'm getting wrong results when using DataFrame.groupby with as_index=False on a column that has a duplicate name. How do I fix this?

### Prompt strategy

The LLM is instructed to:

- Write conversational, natural language
- Extract the core technical ask
- Remove GitHub formatting, reproduction steps, tracebacks
- Keep to 1-3 sentences

### Resumability

Writes results to JSONL incrementally. On restart, skips already-processed `instance_id`s.

**Output:** `data/code_hallucination/queries.jsonl`

---

## Phase 4: Fetch Documentation

**Module:** `context7_docs.py`

Fetches library documentation via the [Context7](https://context7.com) API for **50% of instances** (configurable via `DOCS_RATIO`).

### Library detection

Detects libraries from:
1. Import statements in the patch (`import django`, `from sklearn import ...`)
2. File paths (`django/http/response.py` → django)

Maps to Context7 library names via a predefined dictionary.

### 50/50 split rationale

Half of samples include documentation context, half don't. This creates training variety — models learn to detect hallucinations both with and without documentation support.

Instances not selected for docs still get an entry written with empty docs (by design, not failure).

**Output:** `data/code_hallucination/documentation.jsonl`

---

## Phase 5: Assign Answer Formats

**Module:** `format_builder.py`

Each instance gets exactly one answer format, chosen by weighted random selection from available options.

### Format types

**Complete function** (weight: 0.4)
```python
def validate_response(self, response):
    if response.status_code != 200:
        raise ValidationError(f"Unexpected status: {response.status_code}")
    return response.json()
```
Extracted via Python AST from the patched source. Only available when changes are inside a function (~60% of patches).

**Edit-style** (weight: 0.3)
```
In file django/http/response.py, replace:
    def set_cookie(self, key, value=""):
        self.cookies[key] = value
with:
    def set_cookie(self, key, value="", max_age=None):
        self.cookies[key] = value
        if max_age is not None:
            self.cookies[key]["max-age"] = max_age
```
Available for all patches where changed regions can be extracted.

**Fragment** (weight: 0.3)
```python
if max_age is not None:
    self.cookies[key]["max-age"] = max_age
    self.cookies[key]["expires"] = http_date(time.time() + max_age)
```
Added/changed lines from the diff with surrounding context.

**Output:** `data/code_hallucination/formats.jsonl`

---

## Phase 6: Inject Hallucinations

**Module:** `hallucination_injector.py`

Uses an LLM to inject realistic hallucinations into selected instances (determined by Phase 8). Returns structured JSON with span annotations.

### Hallucination types (round-robin)

| Type | Description | Example |
|------|-------------|---------|
| **Structural** | Non-existent APIs, wrong methods, invented parameters | `response.json_decode()` instead of `response.json()` |
| **Behavioral** | Wrong values, logic errors, off-by-one, swapped conditions | `if status >= 200` instead of `if status == 200` |
| **Semantic** | Code that looks right but does something subtly different | Sorting ascending instead of descending |

### JSON-based span extraction

The LLM returns structured output:

```json
{
  "hallucinated_code": "def fix(self):\n    self.data = response.json_decode()\n    ...",
  "changes": [
    {
      "original": "response.json()",
      "hallucinated": "response.json_decode()",
      "explanation": "json_decode() is not a valid method on Response objects"
    }
  ]
}
```

Spans are found by string-matching each `change["hallucinated"]` in `hallucinated_code`. This produces clean, meaningful spans (minimum 3 chars) with zero noise.

### Quality metrics (from 100-sample test runs)

| Metric | Value |
|--------|-------|
| Noise-only samples | 0% |
| Min span length | 10 chars |
| Avg span length | 70 chars |
| Avg spans per sample | 1.2 |

**Output:** `data/code_hallucination/hallucinated_samples.jsonl`

---

## Phase 7: Assemble Samples

**Module:** `sample_assembler.py`

Combines all intermediate data into the final `HallucinationSample` format.

### Prompt construction

```
File: path/to/file.py
```python
<source code at base commit>
```

Documentation for django:
<Context7 docs if available>

User request: <rewritten query>
```

### Sample types

**Clean samples** (~60%): Gold patch answer, empty labels, from instances NOT selected for injection.

**Hallucinated samples** (~40%): LLM-modified answer with character-level span annotations.

### Outputs

- `data/code_hallucination/code_hallucination_data.json` — List of samples
- `data/code_hallucination/code_hallucination_metadata.json` — Metadata (instance_id, repo, format_type, hallucination_type, injector_model, is_hallucinated)

---

## Phase 8: Select Hallucination Targets

**Module:** `splitter.py`

Selects which instances receive hallucination injection. Applies the hallucination ratio (default 40%) **uniformly within each split** to maintain consistent class distribution.

```
Train: ~7,600 hallucinated + ~11,400 clean = ~19,000
Dev:   ~90 hallucinated + ~135 clean = ~225
Test:  ~920 hallucinated + ~1,374 clean = ~2,294
```

!!! note
    Phase 8 runs **before** Phase 6 in the pipeline (target selection must happen before injection).

**Output:** Set of `instance_id`s (used in-memory by Phase 6 and Phase 7)

---

## Phase 9: Validate

**Module:** `validator.py`

Runs automated quality checks and generates a report.

### Checks performed

| Check | Description |
|-------|-------------|
| **Span validity** | No negative offsets, empty spans, or out-of-bounds |
| **Span coverage** | Distribution of hallucinated text ratio; flags <2% or >80% |
| **Distributions** | Format type, hallucination type, injector model, repo, split |
| **Near-duplicates** | Jaccard similarity >0.95 on sampled answer pairs |
| **AST parseability** | For complete_function format, checks if answer parses as valid Python |
| **Length statistics** | Prompt and answer character length ranges |

**Output:** `data/code_hallucination/validation_report.txt`
