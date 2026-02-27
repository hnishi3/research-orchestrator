from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.artifacts import put_artifact
from resorch.ledger import Ledger
from resorch.projects import get_project
from resorch.utils import utc_now_iso


@dataclass(frozen=True)
class VisualInspectionStatus:
    enabled: bool
    marker_path: str
    pending_figures: List[str]


def _figure_globs(policy: Dict[str, Any]) -> List[str]:
    vi = policy.get("visual_inspection") or {}
    if not isinstance(vi, dict):
        vi = {}
    globs = vi.get("figure_globs") or []
    if not isinstance(globs, list) or not globs:
        return ["results/fig/*.png", "results/fig/*.jpg", "results/fig/*.jpeg", "results/fig/*.svg"]
    return [str(x) for x in globs if x]


def _marker_path(policy: Dict[str, Any]) -> str:
    vi = policy.get("visual_inspection") or {}
    if not isinstance(vi, dict):
        vi = {}
    return str(vi.get("marker_path") or "results/fig/visual_inspection.ok")


def get_visual_inspection_status(*, policy: Dict[str, Any], workspace: Path) -> VisualInspectionStatus:
    enabled = bool(policy.get("requires_visual_inspection", False))
    marker_rel = _marker_path(policy)
    marker_abs = (workspace / marker_rel).resolve()
    marker_mtime = marker_abs.stat().st_mtime if marker_abs.exists() else 0.0

    pending: List[str] = []
    if enabled:
        for pat in _figure_globs(policy):
            for p in workspace.glob(pat):
                if not p.is_file():
                    continue
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if mtime > marker_mtime:
                    try:
                        pending.append(p.resolve().relative_to(workspace).as_posix())
                    except ValueError:
                        continue
        pending = sorted(set(pending))

    return VisualInspectionStatus(enabled=enabled, marker_path=marker_rel, pending_figures=pending)


def approve_visual_inspection(
    *,
    ledger: Ledger,
    project_id: str,
    marker_path: str = "results/fig/visual_inspection.ok",
    note: str = "",
) -> Dict[str, Any]:
    project = get_project(ledger, project_id)
    content = "\n".join(
        [
            "# Visual inspection approval",
            "",
            f"- approved_at: {utc_now_iso()}",
            f"- note: {note}",
            "",
        ]
    )
    art = put_artifact(
        ledger=ledger,
        project=project,
        relative_path=str(marker_path),
        content=content,
        mode="overwrite",
        kind="visual_inspection_ok",
    )
    return {"marker_path": str(marker_path), "artifact": art}

