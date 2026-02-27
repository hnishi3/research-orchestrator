from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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

