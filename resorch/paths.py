from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass(frozen=True)
class RepoPaths:
    root: Path

    @property
    def state_dir(self) -> Path:
        return self.root / ".orchestrator"

    @property
    def db_path(self) -> Path:
        return self.state_dir / "ledger.db"

    @property
    def logs_dir(self) -> Path:
        return self.state_dir / "logs"

    @property
    def workspaces_dir(self) -> Path:
        return self.root / "workspaces"


def _is_logically_within(child: Path, parent: Path) -> bool:
    """Check if *child* is logically within *parent*, ignoring symlinks on child.

    Uses ``os.path.abspath`` (resolves ``.`` and ``..`` but NOT symlinks) on
    *child*, while fully resolving *parent*.
    """
    parent_resolved = str(parent.resolve())
    child_abs = os.path.abspath(str(child))
    return child_abs == parent_resolved or child_abs.startswith(parent_resolved + os.sep)


def resolve_within_workspace(workspace: Path, user_path: Union[str, Path], *, label: str = "output path") -> Path:
    """Resolve *user_path* relative to *workspace* and ensure it stays inside.

    Returns the absolute ``Path``.  Raises ``SystemExit`` if the path escapes
    the workspace directory.  Symlinks inside the workspace are tolerated:
    the check uses the *logical* path (before following symlinks) so that
    symlinked directories (e.g. ``data/`` inherited from a predecessor) pass.
    """
    workspace_resolved = Path(workspace).resolve()
    p = Path(user_path)
    candidate = p if p.is_absolute() else (workspace_resolved / p)

    # Phase 1: logical containment (does NOT follow symlinks on candidate)
    if _is_logically_within(candidate, workspace_resolved):
        return Path(os.path.abspath(str(candidate)))

    # Phase 2: resolved containment (follows all symlinks — original fallback)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError:
        raise SystemExit(
            f"{label} must be within workspace ({workspace_resolved}): {resolved}"
        )
    return resolved


def find_repo_root(start: Path) -> Optional[Path]:
    cur = start.resolve()
    for _ in range(50):
        if (cur / "AGENTS.md").exists() and (cur / "resorch").is_dir():
            return cur
        if cur.parent == cur:
            return None
        cur = cur.parent
    return None


def resolve_repo_paths(repo_root: Optional[str] = None) -> RepoPaths:
    env_root = os.environ.get("RESORCH_ROOT")
    if repo_root:
        root = Path(repo_root).expanduser().resolve()
    elif env_root:
        root = Path(env_root).expanduser().resolve()
    else:
        root = find_repo_root(Path.cwd())
        if root is None:
            raise SystemExit("Could not find repo root (expected AGENTS.md + resorch/). Use --repo-root or set RESORCH_ROOT.")
    return RepoPaths(root=root)

