"""Phase 2: Clone repos and fetch source files via git show."""

import ast
import json
import re
import subprocess
import tempfile
from pathlib import Path

import requests

from .config import MAX_FILE_CHARS, REPOS_DIR, SOURCE_CACHE_DIR

GITHUB_RAW_BASE = "https://raw.githubusercontent.com"


def extract_changed_files(patch: str) -> list[str]:
    """Extract file paths from a unified diff using anchored regex.

    Fixed: Uses re.match on 'diff --git' lines instead of re.search(r"b/(.+)$")
    which was ambiguous on paths containing 'b/'.
    """
    files = []
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            if match:
                files.append(match.group(2))
    return files


def clone_repo(repo: str, repos_dir: Path = REPOS_DIR) -> Path | None:
    """Clone a repo if not already present. Returns repo directory path."""
    repos_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = repos_dir / repo.replace("/", "__")

    if repo_dir.exists():
        return repo_dir

    print(f"  Cloning {repo}...")
    try:
        result = subprocess.run(
            ["git", "clone", "--bare", f"https://github.com/{repo}.git", str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min for large repos
        )
        if result.returncode != 0:
            print(f"  ERROR cloning {repo}: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT cloning {repo} (>30 min)")
        return None

    return repo_dir


def fetch_file_from_github(repo: str, commit: str, filepath: str) -> str | None:
    """Fallback: fetch a file from GitHub raw API (for when repo isn't cloned)."""
    url = f"{GITHUB_RAW_BASE}/{repo}/{commit}/{filepath}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.text[:MAX_FILE_CHARS]
        return None
    except Exception:
        return None


def fetch_file_at_commit(repo_dir: Path, commit: str, filepath: str) -> str | None:
    """Fetch a file's contents at a specific commit using git show."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{filepath}"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout[:MAX_FILE_CHARS]
        return None
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"    Error fetching {filepath}@{commit[:8]}: {e}")
        return None


def apply_patch_and_get_file(repo_dir: Path, commit: str, patch: str, filepath: str) -> str | None:
    """Apply a patch to get the post-fix version of a file.

    Uses a temporary worktree to apply the patch cleanly.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temporary worktree at the base commit
            result = subprocess.run(
                ["git", "worktree", "add", tmpdir, commit],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                return None

            # Apply the patch
            result = subprocess.run(
                ["git", "apply", "--allow-empty", "-"],
                input=patch,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return None

            # Read the patched file
            patched_path = Path(tmpdir) / filepath
            if patched_path.exists():
                content = patched_path.read_text()[:MAX_FILE_CHARS]
                return content

            # Clean up worktree
            subprocess.run(
                ["git", "worktree", "remove", "--force", tmpdir],
                cwd=str(repo_dir),
                capture_output=True,
                timeout=30,
            )
            return None
    except Exception as e:
        print(f"    Error applying patch for {filepath}: {e}")
        return None


def extract_modified_functions(original_source: str, patched_source: str) -> list[dict]:
    """Extract functions that were modified between original and patched source.

    Returns list of dicts with 'name', 'original', 'patched' function bodies.
    """

    def get_functions(source: str) -> dict[str, str]:
        """Parse source and extract function name -> source mapping."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}

        funcs = {}
        lines = source.split("\n")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno - 1
                end = node.end_lineno
                func_source = "\n".join(lines[start:end])
                funcs[node.name] = func_source
        return funcs

    orig_funcs = get_functions(original_source)
    patched_funcs = get_functions(patched_source)

    modified = []
    for name, patched_body in patched_funcs.items():
        orig_body = orig_funcs.get(name)
        if orig_body and orig_body != patched_body:
            modified.append(
                {
                    "name": name,
                    "original": orig_body,
                    "patched": patched_body,
                }
            )
        elif not orig_body:
            # New function added by patch
            modified.append(
                {
                    "name": name,
                    "original": None,
                    "patched": patched_body,
                }
            )

    return modified


def apply_patch_in_memory(original_source: str, patch: str, filepath: str) -> str | None:
    """Apply a unified diff patch to source code in memory, without git.

    Parses the unified diff to extract hunks for the target file,
    then applies them line-by-line to produce the patched source.

    :param original_source: The original file content.
    :param patch: The full unified diff (may contain multiple files).
    :param filepath: The specific file to extract and apply hunks for.
    :return: Patched source string, or None if application fails.
    """
    # Split patch into per-file sections
    file_patch_lines = []
    in_target_file = False

    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            in_target_file = match is not None and match.group(2) == filepath
            continue
        if in_target_file:
            file_patch_lines.append(line)

    if not file_patch_lines:
        return None

    # Parse hunks from the file-specific patch lines
    hunks = []
    current_hunk = None

    for line in file_patch_lines:
        if line.startswith("@@"):
            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            hunk_match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if hunk_match:
                current_hunk = {
                    "old_start": int(hunk_match.group(1)),
                    "lines": [],
                }
                hunks.append(current_hunk)
        elif line.startswith("---") or line.startswith("+++"):
            continue
        elif current_hunk is not None:
            current_hunk["lines"].append(line)

    if not hunks:
        return None

    # Apply hunks to original source
    original_lines = original_source.split("\n")
    result_lines = []
    orig_idx = 0  # 0-based index into original_lines

    try:
        for hunk in hunks:
            hunk_start = hunk["old_start"] - 1  # Convert to 0-based

            # Copy unchanged lines before this hunk
            while orig_idx < hunk_start and orig_idx < len(original_lines):
                result_lines.append(original_lines[orig_idx])
                orig_idx += 1

            # Apply hunk lines
            for line in hunk["lines"]:
                if line.startswith("+"):
                    result_lines.append(line[1:])
                elif line.startswith("-"):
                    orig_idx += 1  # Skip the removed line
                elif line.startswith(" "):
                    result_lines.append(line[1:])
                    orig_idx += 1
                elif line == "":
                    # Empty line in diff context — treat as context
                    if orig_idx < len(original_lines):
                        result_lines.append(original_lines[orig_idx])
                        orig_idx += 1

        # Copy remaining lines after last hunk
        while orig_idx < len(original_lines):
            result_lines.append(original_lines[orig_idx])
            orig_idx += 1

        return "\n".join(result_lines)
    except (IndexError, ValueError):
        return None


def extract_code_from_patch(patch: str) -> str:
    """Extract added/changed lines from a unified diff as code fragment.

    Fixed: Uses removeprefix("b/") instead of lstrip("b/").
    """
    added_lines = []
    in_hunk = False

    for line in patch.split("\n"):
        if line.startswith("@@"):
            in_hunk = True
            continue
        if line.startswith("diff --git") or line.startswith("---") or line.startswith("+++"):
            continue
        if in_hunk:
            if line.startswith("+"):
                added_lines.append(line[1:])
            elif line.startswith("-"):
                continue
            else:
                if line.startswith(" "):
                    added_lines.append(line[1:])
                else:
                    added_lines.append(line)

    return "\n".join(added_lines)


def build_edit_style_answer(patch: str, changed_files: list[str]) -> str | None:
    """Build an edit-style answer from a patch.

    Format:
        In file path/to/file.py, replace:
        ```python
        old code
        ```
        with:
        ```python
        new code
        ```
    """
    edits = []

    current_file = None
    old_lines = []
    new_lines = []
    in_hunk = False

    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            # Flush previous hunk
            if current_file and (old_lines or new_lines):
                edits.append(
                    {
                        "file": current_file,
                        "old": "\n".join(old_lines),
                        "new": "\n".join(new_lines),
                    }
                )
                old_lines = []
                new_lines = []
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            if match:
                current_file = match.group(2)
            in_hunk = False
            continue

        if line.startswith("@@"):
            # Flush previous hunk in same file
            if old_lines or new_lines:
                edits.append(
                    {
                        "file": current_file,
                        "old": "\n".join(old_lines),
                        "new": "\n".join(new_lines),
                    }
                )
                old_lines = []
                new_lines = []
            in_hunk = True
            continue

        if line.startswith("---") or line.startswith("+++"):
            continue

        if in_hunk:
            if line.startswith("-"):
                old_lines.append(line[1:])
            elif line.startswith("+"):
                new_lines.append(line[1:])
            # Context lines are skipped in edit-style

    # Flush last hunk
    if current_file and (old_lines or new_lines):
        edits.append(
            {
                "file": current_file,
                "old": "\n".join(old_lines),
                "new": "\n".join(new_lines),
            }
        )

    if not edits:
        return None

    parts = []
    for edit in edits:
        if edit["old"] and edit["new"]:
            parts.append(
                f"In file {edit['file']}, replace:\n```python\n{edit['old']}\n```\nwith:\n```python\n{edit['new']}\n```"
            )
        elif edit["new"]:
            parts.append(f"In file {edit['file']}, add:\n```python\n{edit['new']}\n```")

    return "\n\n".join(parts) if parts else None


def fetch_source_for_instance(
    instance: dict, repos_dir: Path = REPOS_DIR, use_github_api: bool = False
) -> dict | None:
    """Fetch all source files for a single SWE-bench instance.

    Args:
        instance: SWE-bench instance dict
        repos_dir: Directory for cloned repos
        use_github_api: If True, use GitHub raw API instead of git clone

    Returns dict with:
        instance_id, changed_files, source_files, patch_code, edit_style,
        modified_functions
    Or None if fetching failed.

    """
    repo = instance["repo"]
    commit = instance["base_commit"]
    patch = instance["patch"]

    # Extract changed files from patch
    changed_files = extract_changed_files(patch)
    if not changed_files:
        return None

    # Fetch source files
    source_files = {}
    repo_dir = None

    if use_github_api:
        # Use GitHub raw API (slower but no cloning needed)
        import time

        for filepath in changed_files:
            content = fetch_file_from_github(repo, commit, filepath)
            if content:
                source_files[filepath] = content
            time.sleep(0.3)
    else:
        # Clone repo and use git show (fast for bulk)
        repo_dir = clone_repo(repo, repos_dir)
        if repo_dir is None:
            # Fallback to GitHub API
            import time

            print(f"    Falling back to GitHub API for {repo}")
            for filepath in changed_files:
                content = fetch_file_from_github(repo, commit, filepath)
                if content:
                    source_files[filepath] = content
                time.sleep(0.3)
        else:
            for filepath in changed_files:
                content = fetch_file_at_commit(repo_dir, commit, filepath)
                if content:
                    source_files[filepath] = content

    if not source_files:
        return None

    # Build different answer formats
    # Fragment format
    patch_code = extract_code_from_patch(patch)

    # Edit-style format
    edit_style = build_edit_style_answer(patch, changed_files)

    # Complete function format — extract modified functions
    modified_functions = []
    for filepath in changed_files:
        if filepath not in source_files:
            continue
        if repo_dir is not None:
            patched_source = apply_patch_and_get_file(repo_dir, commit, patch, filepath)
        else:
            patched_source = apply_patch_in_memory(source_files[filepath], patch, filepath)
        if patched_source:
            funcs = extract_modified_functions(source_files[filepath], patched_source)
            for func in funcs:
                func["file"] = filepath
            modified_functions.extend(funcs)

    return {
        "instance_id": instance["instance_id"],
        "changed_files": changed_files,
        "source_files": source_files,
        "patch_code": patch_code,
        "edit_style": edit_style,
        "modified_functions": modified_functions,
    }


def run(instances: list[dict], use_github_api: bool = False):
    """Run Phase 2: Fetch source files for all instances.

    Args:
        instances: List of SWE-bench instances
        use_github_api: If True, use GitHub raw API instead of cloning repos

    """
    print("=" * 60)
    print("Phase 2: Source File Fetching")
    print("=" * 60)

    SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not use_github_api:
        REPOS_DIR.mkdir(parents=True, exist_ok=True)
        # Group by repo for efficient cloning
        repos = set(inst["repo"] for inst in instances)
        print(f"Need to clone {len(repos)} repos")
        for repo in sorted(repos):
            clone_repo(repo)
    else:
        print("Using GitHub raw API (no cloning)")

    # Fetch sources per instance
    results = []
    failed = 0

    for i, instance in enumerate(instances):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(instances)} ({len(results)} success, {failed} failed)")

        # Skip if already cached
        cache_path = SOURCE_CACHE_DIR / f"{instance['instance_id']}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                results.append(json.load(f))
            continue

        result = fetch_source_for_instance(instance, use_github_api=use_github_api)
        if result:
            results.append(result)
            # Cache result
            cache_path = SOURCE_CACHE_DIR / f"{instance['instance_id']}.json"
            with open(cache_path, "w") as f:
                json.dump(result, f)
        else:
            failed += 1

    print(f"\nDone: {len(results)} success, {failed} failed out of {len(instances)}")

    # Report format availability
    n_fragment = sum(1 for r in results if r.get("patch_code", "").strip())
    n_edit = sum(1 for r in results if r.get("edit_style"))
    n_function = sum(1 for r in results if r.get("modified_functions"))
    print("Format availability:")
    print(f"  Fragment: {n_fragment}")
    print(f"  Edit-style: {n_edit}")
    print(f"  Complete function: {n_function}")

    return results


if __name__ == "__main__":
    from .swebench_loader import load_instances

    instances = load_instances()
    run(instances)
