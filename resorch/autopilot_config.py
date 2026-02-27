from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class ReviewRecommendation:
    level: str  # none|soft|hard
    reasons: List[str]
    targets: List[str]


def _load_yaml(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SystemExit(f"Invalid YAML (expected object): {path}")
    return raw


def load_review_policy(repo_root: Path, workspace: Optional[Path] = None) -> Dict[str, Any]:
    """Load review policy with workspace-level override support.

    Resolution order:
      1. workspace/configs/review_policy.yaml  (if workspace given and file exists)
      2. repo_root/configs/review_policy.yaml   (global fallback, if exists)
      3. empty dict                             (no config found)
    """
    if workspace is not None:
        ws_path = Path(workspace) / "configs" / "review_policy.yaml"
        if ws_path.exists():
            return _load_yaml(ws_path)
    global_path = repo_root / "configs" / "review_policy.yaml"
    if global_path.exists():
        return _load_yaml(global_path)
    return {}


def load_pivot_policy(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / "configs" / "pivot_policy.yaml"
    if not path.exists():
        return {}
    return _load_yaml(path)


def load_plan_schema(repo_root: Path) -> Dict[str, Any]:
    return json.loads((repo_root / "schemas" / "autopilot_plan.schema.json").read_text(encoding="utf-8"))
