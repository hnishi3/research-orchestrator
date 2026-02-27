from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.ledger import Ledger
from resorch.projects import set_project_stage

_ACTIVE_STAGES = {"intake", "analysis", "experiment", "implementation", "writing"}
_STAGE_BONUS = {
    "intake": 3,
    "analysis": 2,
    "experiment": 2,
    "implementation": 1,
    "writing": 1,
}
_FAIL_STATUSES = {"failed", "blocked", "rate_limited"}


@dataclass
class ProjectState:
    project_id: str
    title: str
    stage: str
    staleness_hours: float
    primary_metric_delta: Optional[float]
    fail_streak: int
    priority_score: float


def compute_priority(ps: ProjectState) -> float:
    stage_bonus = float(_STAGE_BONUS.get(str(ps.stage or "").strip().lower(), 0))
    return float(ps.staleness_hours) * 0.1 - float(ps.fail_streak) * 5.0 + stage_bonus


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("value", "mean", "current", "best", "baseline", "delta_vs_baseline"):
            if key in value:
                out = _to_float(value.get(key))
                if out is not None:
                    return out
        return None
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _parse_ts(value: str) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _staleness_hours(updated_at: str) -> float:
    ts = _parse_ts(updated_at)
    if ts is None:
        return 0.0
    delta = datetime.now(timezone.utc) - ts.astimezone(timezone.utc)
    return max(0.0, delta.total_seconds() / 3600.0)


def _project_fail_streak(ledger: Ledger, project_id: str) -> int:
    tasks = ledger.list_tasks(project_id=project_id)
    if not tasks:
        return 0
    ordered = sorted(
        tasks,
        key=lambda t: (str(t.get("updated_at") or ""), str(t.get("id") or "")),
        reverse=True,
    )
    streak = 0
    for task in ordered:
        status = str(task.get("status") or "").strip().lower()
        if status in _FAIL_STATUSES:
            streak += 1
            continue
        break
    return streak


def _read_primary_metric_delta(project: Dict[str, Any]) -> Optional[float]:
    workspace = Path(str(project.get("repo_path") or "")).resolve()
    scoreboard = workspace / "results" / "scoreboard.json"
    try:
        raw = scoreboard.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    pm = data.get("primary_metric")
    if not isinstance(pm, dict):
        pm = {}
    delta = _to_float(pm.get("delta_vs_baseline"))
    if delta is not None:
        return delta
    current = _to_float(pm.get("current"))
    baseline = _to_float(pm.get("baseline"))
    if current is not None and baseline is not None:
        return current - baseline
    return None


def _build_project_state(ledger: Ledger, project: Dict[str, Any]) -> ProjectState:
    state = ProjectState(
        project_id=str(project.get("id") or ""),
        title=str(project.get("title") or ""),
        stage=str(project.get("stage") or ""),
        staleness_hours=_staleness_hours(str(project.get("updated_at") or "")),
        primary_metric_delta=_read_primary_metric_delta(project),
        fail_streak=_project_fail_streak(ledger, str(project.get("id") or "")),
        priority_score=0.0,
    )
    state.priority_score = compute_priority(state)
    return state


def run_portfolio_cycle(
    ledger: Ledger,
    max_projects: int = 3,
    steps_per_project: int = 5,
    dry_run: bool = False,
) -> Dict[str, Any]:
    max_projects = max(0, int(max_projects))
    steps_per_project = max(0, int(steps_per_project))
    result: Dict[str, Any] = {
        "projects_evaluated": 0,
        "projects_selected": 0,
        "projects_executed": 0,
        "projects_completed": 0,
        "playbook_extractions": 0,
        "decision_log": [],
    }

    projects = ledger.list_projects()
    active_projects = [p for p in projects if str(p.get("stage") or "").strip().lower() in _ACTIVE_STAGES]
    result["projects_evaluated"] = len(active_projects)
    if not active_projects:
        result["decision_log"].append({"event": "no_active_projects"})
        return result

    states = [_build_project_state(ledger, p) for p in active_projects]
    states.sort(key=lambda ps: (ps.priority_score, ps.staleness_hours, ps.project_id), reverse=True)
    selected = states[:max_projects]
    result["projects_selected"] = len(selected)

    result["decision_log"].append(
        {
            "event": "selection",
            "max_projects": max_projects,
            "project_ids": [ps.project_id for ps in selected],
            "priorities": [
                {
                    "project_id": ps.project_id,
                    "priority_score": ps.priority_score,
                    "stage": ps.stage,
                    "staleness_hours": ps.staleness_hours,
                    "fail_streak": ps.fail_streak,
                    "primary_metric_delta": ps.primary_metric_delta,
                }
                for ps in states
            ],
        }
    )

    if dry_run:
        result["decision_log"].append({"event": "dry_run", "executed": 0})
        return result

    for ps in selected:
        objective = f"Portfolio cycle execution for '{ps.title}' (project: {ps.project_id})."
        try:
            from resorch.agent_loop import run_agent_loop
        except ImportError as exc:
            result["decision_log"].append(
                {"event": "agent_loop_unavailable", "project_id": ps.project_id, "error": str(exc)}
            )
            continue
        try:
            run_out = run_agent_loop(
                ledger=ledger,
                project_id=ps.project_id,
                objective=objective,
                max_steps=steps_per_project,
                dry_run=False,
                config_path=None,
            )
        except Exception as exc:  # noqa: BLE001
            result["decision_log"].append(
                {"event": "execution_error", "project_id": ps.project_id, "error": str(exc)}
            )
            continue

        result["projects_executed"] += 1
        stopped_reason = str(run_out.get("stopped_reason") or "")
        result["decision_log"].append(
            {
                "event": "executed",
                "project_id": ps.project_id,
                "stopped_reason": stopped_reason,
                "steps_run": len(run_out.get("steps") or []),
            }
        )

        if stopped_reason.strip().lower().startswith("done"):
            set_project_stage(ledger, ps.project_id, "done")
            result["projects_completed"] += 1
            result["playbook_extractions"] += 1
            result["decision_log"].append(
                {
                    "event": "completed",
                    "project_id": ps.project_id,
                    "new_stage": "done",
                    "playbook_extraction": "attempted",
                }
            )

    return result
