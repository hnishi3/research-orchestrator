from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List


def _git(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)


def _parse_numstat(text: str) -> int:
    total = 0
    for line in (text or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        a, d = parts[0], parts[1]
        try:
            add = int(a) if a.isdigit() else 0
            dele = int(d) if d.isdigit() else 0
        except ValueError:
            continue
        total += add + dele
    return total


def _ensure_git_baseline(workspace: Path, *, iteration: int) -> None:
    """Commit all current workspace files so git diff reflects only this iteration's changes."""
    if not (workspace / ".git").exists():
        return
    _git(["git", "add", "-A"], cwd=workspace)
    _git(
        ["git", "commit", "-m", f"baseline-iter-{iteration:03d}", "--allow-empty", "--no-gpg-sign"],
        cwd=workspace,
    )


_REVIEW_EXCLUDE_PREFIXES = (
    ".venv", "venv/", ".venv/", "node_modules/", "__pycache__/",
    ".tox/", ".nox/", ".mypy_cache/", ".pytest_cache/",
)
_REVIEW_EXCLUDE_SEGMENTS = ("/site-packages/",)


def _is_review_excluded(path: str) -> bool:
    """Return True if *path* should be excluded from review targets."""
    for prefix in _REVIEW_EXCLUDE_PREFIXES:
        if path.startswith(prefix):
            return True
    for seg in _REVIEW_EXCLUDE_SEGMENTS:
        if seg in path:
            return True
    return False


def _list_git_changed_paths(workspace: Path) -> List[str]:
    if not (workspace / ".git").exists():
        return []

    paths: set[str] = set()
    for cmd in (["git", "diff", "--name-only"], ["git", "diff", "--cached", "--name-only"]):
        proc = _git(cmd, cwd=workspace)
        for p in proc.stdout.splitlines():
            p = p.strip()
            if p and not _is_review_excluded(p):
                paths.add(p)

    untracked_proc = _git(["git", "ls-files", "--others", "--exclude-standard"], cwd=workspace)
    for p in untracked_proc.stdout.splitlines():
        p = p.strip()
        if p and not _is_review_excluded(p):
            paths.add(p)
    return sorted(paths)


def compute_git_change_summary(
    workspace: Path,
    *,
    max_untracked_files: int = 50,
    max_untracked_total_lines: int = 5000,
) -> Dict[str, Any]:
    """Best-effort summary of workspace changes (git diff + untracked)."""

    if not (workspace / ".git").exists():
        return {"is_git": False, "changed_lines": 0, "changed_files": 0, "changed_paths": []}

    changed_paths = _list_git_changed_paths(workspace)

    changed_lines = 0
    changed_lines += _parse_numstat(_git(["git", "diff", "--numstat"], cwd=workspace).stdout)
    changed_lines += _parse_numstat(_git(["git", "diff", "--cached", "--numstat"], cwd=workspace).stdout)

    # Roughly account for untracked files by counting their lines (best effort).
    untracked_proc = _git(["git", "ls-files", "--others", "--exclude-standard"], cwd=workspace)
    untracked = [p for p in untracked_proc.stdout.splitlines() if p.strip()]

    counted_files = 0
    counted_lines = 0
    for rel in untracked:
        if counted_files >= max_untracked_files or counted_lines >= max_untracked_total_lines:
            break
        p = (workspace / rel).resolve()
        try:
            if not p.is_file():
                continue
            lines = p.read_text(encoding="utf-8", errors="replace").count("\n") + 1
        except OSError:
            continue
        counted_files += 1
        counted_lines += lines
    changed_lines += min(counted_lines, max_untracked_total_lines)

    return {
        "is_git": True,
        "changed_lines": int(changed_lines),
        "changed_files": int(len(changed_paths)),
        "changed_paths": changed_paths,
    }
