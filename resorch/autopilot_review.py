from __future__ import annotations

import fnmatch
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.autopilot_config import ReviewRecommendation
from resorch.ledger import Ledger
from resorch.stage_gates import compute_gate_env, evaluate_transitions, load_stage_transitions


def _parse_iso_z(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    t = str(ts).strip()
    if not t:
        return None
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(t)
    except ValueError:
        return None


def compute_failure_streaks(ledger: Ledger, *, project_id: str, limit: int = 200) -> Dict[str, int]:
    """Compute recent failure streaks from task_runs (more reliable than tasks.status)."""

    rows = ledger._exec(
        """
        SELECT tr.task_id, tr.status
        FROM task_runs tr
        JOIN tasks t ON t.id = tr.task_id
        WHERE t.project_id = ?
        ORDER BY tr.started_at DESC
        LIMIT ?
        """,
        (project_id, limit),
    ).fetchall()

    any_streak = 0
    for r in rows:
        st = str(r.get("status") or "")
        if st == "success":
            break
        if st in {"failed", "blocked"}:
            any_streak += 1

    same_task_streak = 0
    last_task_id = str(rows[0]["task_id"]) if rows else ""
    if last_task_id:
        for r in rows:
            if str(r.get("task_id") or "") != last_task_id:
                break  # Stop at first different task (true consecutive streak)
            st = str(r.get("status") or "")
            if st == "success":
                break
            if st in {"failed", "blocked"}:
                same_task_streak += 1

    return {"any": int(any_streak), "same_task": int(same_task_streak)}


def _detect_external_fetch(tasks_ran: List[Dict[str, Any]]) -> bool:
    for t in tasks_ran:
        if str(t.get("type") or "") != "shell_exec":
            continue
        spec = t.get("spec") or {}
        if not isinstance(spec, dict):
            continue
        cmd = spec.get("command")
        if isinstance(cmd, list):
            cmd_str = " ".join(str(x) for x in cmd)
        else:
            cmd_str = str(cmd or "")
        hay = cmd_str.lower()
        markers = [
            "curl ",
            "wget ",
            "pip install",
            "pip3 install",
            "poetry add",
            "npm install",
            "pnpm add",
            "yarn add",
            "apt-get install",
            "apt install",
            "brew install",
            "git clone",
        ]
        if any(m in hay for m in markers):
            return True
    return False


def _any_path_matches(paths: List[str], patterns: List[str]) -> bool:
    for p in paths:
        for pat in patterns:
            if fnmatch.fnmatch(p, pat):
                return True
    return False


def _list_ready_stage_transitions(ledger: Ledger, *, project_id: str) -> List[Dict[str, Any]]:
    """Best-effort: evaluate stage transitions and report those ready."""

    config_path = ledger.paths.root / "configs" / "stage_transitions.yaml"
    if not config_path.exists():
        return []
    try:
        config = load_stage_transitions(config_path)
        env = compute_gate_env(ledger=ledger, project_id=project_id)
        evaluated = evaluate_transitions(config=config, env=env)
    except Exception as e:  # noqa: BLE001
        return [{"name": "stage_transitions_error", "decision": "unknown", "error": str(e)}]

    ready = []
    transitions = evaluated.get("transitions") or {}
    if isinstance(transitions, dict):
        for name, t in transitions.items():
            if not isinstance(t, dict):
                continue
            decision = str(t.get("decision") or "")
            if decision in {"auto_pass", "manual"}:
                ready.append({"name": str(name), "decision": decision})
    return ready


def _claims_created_since(ledger: Ledger, *, project_id: str, since_iso: str) -> bool:
    row = ledger._exec(
        """
        SELECT 1 FROM artifacts
        WHERE project_id = ? AND kind = 'claim_md' AND created_at >= ?
        LIMIT 1
        """,
        (project_id, since_iso),
    ).fetchone()
    return row is not None


def _any_stalled_jobs(ledger: Ledger, *, project_id: str, stall_minutes: int) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    stalled: List[Dict[str, Any]] = []
    for j in ledger.list_jobs(project_id=project_id, status=None, limit=200):
        status = str(j.get("status") or "")
        if status not in {"running", "submitted", "submitted_external", "running_external"}:
            continue
        started_at = _parse_iso_z(str(j.get("started_at") or ""))
        if started_at is None:
            continue
        age_min = (now - started_at).total_seconds() / 60.0
        if age_min >= float(stall_minutes):
            stalled.append(
                {
                    "id": j.get("id"),
                    "provider": j.get("provider"),
                    "kind": j.get("kind"),
                    "age_minutes": round(age_min, 1),
                }
            )
    return stalled


def _list_pending_external_jobs(ledger: Ledger, *, project_id: str) -> List[Dict[str, Any]]:
    pending: List[Dict[str, Any]] = []
    for j in ledger.list_jobs(project_id=project_id, status=None, limit=200):
        if str(j.get("provider") or "") != "compute":
            continue
        status = str(j.get("status") or "")
        if status not in {"submitted_external", "running_external"}:
            continue
        pending.append({"id": str(j.get("id") or ""), "status": status, "remote_id": j.get("remote_id")})
    return [p for p in pending if p.get("id")]


def recommend_review_from_policy(
    *,
    policy: Dict[str, Any],
    plan_self_confidence: Optional[float],
    plan_evidence_strength: Optional[float],
    git_changed_lines: int,
    git_changed_files: int,
    git_changed_paths: List[str],
    failure_streak_any: int,
    failure_streak_same_task: int,
    ready_stage_transitions: List[Dict[str, Any]],
    claim_created: bool,
    external_fetch_detected: bool,
    stalled_jobs: List[Dict[str, Any]],
    default_targets: List[str],
    plan_suggested: Optional[Dict[str, Any]] = None,
    stage_transition_requested: bool = False,
) -> ReviewRecommendation:
    reasons: List[str] = []
    level = "none"

    gap: Optional[float] = None
    if plan_self_confidence is not None and plan_evidence_strength is not None:
        gap = float(plan_self_confidence) - float(plan_evidence_strength)

    hard_gates = policy.get("hard_gates") or {}
    soft_triggers = policy.get("soft_triggers") or {}
    if not isinstance(hard_gates, dict):
        hard_gates = {}
    if not isinstance(soft_triggers, dict):
        soft_triggers = {}

    # --- Hard gates ---
    if bool(hard_gates.get("on_stage_transition")) and stage_transition_requested and ready_stage_transitions:
        level = "hard"
        reasons.append(
            f"stage_transition_ready={ready_stage_transitions[0]['name']} ({ready_stage_transitions[0]['decision']})"
        )

    if bool(hard_gates.get("on_claim_create")) and claim_created:
        level = "hard"
        reasons.append("claim_created")

    paper_spec = hard_gates.get("on_paper_artifact_change") or {}
    if isinstance(paper_spec, dict):
        pats = paper_spec.get("paths") or []
        if isinstance(pats, list) and _any_path_matches(git_changed_paths, [str(x) for x in pats if x]):
            level = "hard"
            reasons.append("paper_files_changed")

    dep_spec = hard_gates.get("on_dependency_change") or {}
    if isinstance(dep_spec, dict):
        pats2 = dep_spec.get("paths") or []
        if isinstance(pats2, list) and _any_path_matches(git_changed_paths, [str(x) for x in pats2 if x]):
            level = "hard"
            reasons.append("dependency_files_changed")

    if bool(hard_gates.get("on_external_fetch")) and external_fetch_detected:
        level = "hard"
        reasons.append("external_fetch_detected")

    # --- Soft triggers ---
    fail_spec = soft_triggers.get("fail_streak") or {}
    if isinstance(fail_spec, dict):
        try:
            same_task_th = int(fail_spec.get("same_task", 2))
        except (TypeError, ValueError):
            same_task_th = 2
        try:
            any_th = int(fail_spec.get("any", 3))
        except (TypeError, ValueError):
            any_th = 3

        if failure_streak_same_task >= same_task_th:
            if level != "hard":
                level = "soft"
            reasons.append(f"fail_streak.same_task={failure_streak_same_task} >= {same_task_th}")
        if failure_streak_any >= any_th:
            if level != "hard":
                level = "soft"
            reasons.append(f"fail_streak.any={failure_streak_any} >= {any_th}")

    if stalled_jobs and (soft_triggers.get("stall_minutes") is not None):
        if level != "hard":
            level = "soft"
        reasons.append(f"stalled_jobs count={len(stalled_jobs)}")

    gap_spec = soft_triggers.get("confidence_gap") or {}
    if gap is not None and isinstance(gap_spec, dict):
        try:
            th = float(gap_spec.get("threshold", 0.35))
        except (TypeError, ValueError):
            th = 0.35
        if gap >= th and level != "hard":
            level = "soft"
            reasons.append(f"confidence_gap={gap:.2f} >= {th}")

    diff_spec = soft_triggers.get("git_diff") or {}
    if isinstance(diff_spec, dict) and level != "hard":
        try:
            max_lines = int(diff_spec.get("max_lines", 200))
        except (TypeError, ValueError):
            max_lines = 200
        try:
            max_files = int(diff_spec.get("max_files", 8))
        except (TypeError, ValueError):
            max_files = 8

        if git_changed_lines >= max_lines:
            level = "soft"
            reasons.append(f"git_diff.lines={git_changed_lines} >= {max_lines}")
        if git_changed_files >= max_files:
            level = "soft"
            reasons.append(f"git_diff.files={git_changed_files} >= {max_files}")

    # Respect planner hint as a lower bound.
    if isinstance(plan_suggested, dict):
        hinted = str(plan_suggested.get("level") or "").strip()
        if hinted in {"soft", "hard"}:
            if hinted == "hard":
                level = "hard"
            elif level == "none":
                level = "soft"
            if plan_suggested.get("reasons"):
                reasons.extend([str(x) for x in (plan_suggested.get("reasons") or []) if x])

    targets = default_targets
    if isinstance(plan_suggested, dict) and isinstance(plan_suggested.get("targets"), list):
        targets = [str(x) for x in plan_suggested["targets"] if x]
    if not targets:
        targets = default_targets

    if level == "none":
        reasons = []

    return ReviewRecommendation(level=level, reasons=reasons, targets=targets)
