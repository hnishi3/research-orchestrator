from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.agent_loop import run_agent_loop
from resorch.artifacts import put_artifact
from resorch.ideas import get_idea as get_idea_fn
from resorch.ideas import list_ideas as list_ideas_fn
from resorch.ledger import Ledger
from resorch.projects import create_project, get_project
from resorch.utils import utc_now_iso


def _safe_ts() -> str:
    return utc_now_iso().replace("-", "").replace(":", "").replace("T", "").replace("Z", "")


def _safe_lock_filename(idea_id: str) -> str:
    return "".join((c if c.isalnum() or c in {"-", "_", "."} else "_") for c in idea_id) + ".lock.json"


def _as_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _primary_metric_current_mean(pm: Dict[str, Any]) -> Optional[float]:
    cur = pm.get("current")
    if isinstance(cur, dict):
        return _as_float(cur.get("mean"))
    return _as_float(cur)


def _try_reserve_idea(*, lock_dir: Path, idea_id: str, record: Dict[str, Any]) -> bool:
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / _safe_lock_filename(idea_id)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    except Exception:  # noqa: BLE001
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return True


def _read_scoreboard_primary_metric(workspace: Path) -> Dict[str, Any]:
    path = (workspace / "results" / "scoreboard.json").resolve()
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw) if raw.strip() else {}
    except (OSError, json.JSONDecodeError):
        obj = {}
    if not isinstance(obj, dict):
        obj = {}
    pm = obj.get("primary_metric")
    return pm if isinstance(pm, dict) else {}


def run_cohort(
    *,
    ledger: Ledger,
    base_project_id: str,
    objective: str,
    n: int = 3,
    max_steps: int = 3,
    dry_run: bool = True,
    config_path: Optional[str] = None,
    ideas_per_agent: int = 1,
) -> Dict[str, Any]:
    if n < 1:
        raise SystemExit("--n must be >= 1")
    if n > 20:
        raise SystemExit("--n too large (max 20)")
    if ideas_per_agent < 0:
        ideas_per_agent = 0
    if ideas_per_agent > 10:
        ideas_per_agent = 10

    base_project = get_project(ledger, base_project_id)
    base_ws = Path(base_project["repo_path"]).resolve()
    run_id = _safe_ts()

    # Reserve top ideas (by score) to avoid duplicate experiments.
    ideas = list_ideas_fn(ledger=ledger, project_id=base_project_id, limit=500)
    candidates = [it for it in ideas if str(it.get("status") or "") not in {"rejected", "done", "parked"}]
    lock_dir = (base_ws / "ideas" / "locks").resolve()

    assigned: List[List[str]] = [[] for _ in range(n)]
    cand_idx = 0
    for agent_idx in range(n):
        while len(assigned[agent_idx]) < ideas_per_agent and cand_idx < len(candidates):
            idea_id = str(candidates[cand_idx].get("id") or "")
            cand_idx += 1
            if not idea_id:
                continue
            ok = _try_reserve_idea(
                lock_dir=lock_dir,
                idea_id=idea_id,
                record={
                    "schema_version": 1,
                    "reserved_at": utc_now_iso(),
                    "base_project_id": base_project_id,
                    "run_id": run_id,
                    "agent_idx": int(agent_idx),
                    "idea_id": idea_id,
                },
            )
            if ok:
                assigned[agent_idx].append(idea_id)

    members: List[Dict[str, Any]] = []
    for agent_idx in range(n):
        member_id = f"{base_project_id}-cohort-{run_id[:8]}-{agent_idx+1}"
        if len(member_id) > 64:
            member_id = f"cohort-{run_id[:10]}-{agent_idx+1}"

        member_project = create_project(
            ledger=ledger,
            project_id=member_id,
            title=f"{base_project.get('title')} (cohort {run_id[:8]} #{agent_idx+1})",
            domain=str(base_project.get("domain") or ""),
            stage=str(base_project.get("stage") or "intake"),
            git_init=False,
        )
        member_ws = Path(member_project["repo_path"]).resolve()

        # Copy a small set of base artifacts (if present) to align constraints/context.
        for rel in ["constraints.yaml", "topic_brief.md", "notes/problem.md", "notes/method.md"]:
            src = (base_ws / rel).resolve()
            dst = (member_ws / rel).resolve()
            if src.exists() and src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

        assigned_ids = assigned[agent_idx]
        assigned_details: List[Dict[str, Any]] = []
        for iid in assigned_ids:
            try:
                row = get_idea_fn(ledger=ledger, idea_id=iid)
                data = row.get("data") if isinstance(row, dict) else {}
                if not isinstance(data, dict):
                    data = {}
                assigned_details.append(
                    {
                        "id": iid,
                        "title": str(data.get("title") or ""),
                        "status": str(data.get("status") or ""),
                        "one_sentence_claim": str(data.get("one_sentence_claim") or ""),
                    }
                )
            except Exception:  # noqa: BLE001
                assigned_details.append({"id": iid})

        (member_ws / "notes").mkdir(parents=True, exist_ok=True)
        (member_ws / "notes" / "cohort_assignment.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "base_project_id": base_project_id,
                    "run_id": run_id,
                    "agent_idx": int(agent_idx),
                    "assigned_ideas": assigned_details,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        idea_hint = ", ".join(assigned_ids) if assigned_ids else "(none)"
        member_objective = (
            f"[Cohort {run_id[:8]} member {agent_idx+1}/{n}]\n"
            f"Assigned idea_ids: {idea_hint}\n"
            "Try to explore a unique branch compared to other cohort members.\n\n"
            + objective
        )

        run_out = run_agent_loop(
            ledger=ledger,
            project_id=member_project["id"],
            objective=member_objective,
            max_steps=int(max_steps),
            dry_run=bool(dry_run),
            config_path=config_path,
        )

        pm = _read_scoreboard_primary_metric(member_ws)
        members.append(
            {
                "project_id": member_project["id"],
                "workspace": str(member_ws),
                "assigned_ideas": assigned_ids,
                "stopped_reason": run_out.get("stopped_reason"),
                "primary_metric": pm,
            }
        )

    # Lab meeting artifact under the base project.
    lines: List[str] = []
    lines.append("# Lab Meeting: Cohort Run\n\n")
    lines.append(f"- base_project_id: `{base_project_id}`\n")
    lines.append(f"- run_id: `{run_id}`\n")
    lines.append(f"- n: {n}\n")
    lines.append(f"- dry_run: {dry_run}\n")
    lines.append("\n## Members\n")
    for m in members:
        lines.append(f"\n### {m['project_id']}\n")
        lines.append(f"- assigned_ideas: {', '.join(m['assigned_ideas']) if m['assigned_ideas'] else '(none)'}\n")
        lines.append(f"- stopped_reason: {m.get('stopped_reason')}\n")
        pm = m.get("primary_metric") or {}
        if isinstance(pm, dict):
            cur = _primary_metric_current_mean(pm)
            lines.append(f"- primary_metric.current.mean: {cur if cur is not None else pm.get('current')}\n")
            lines.append(f"- primary_metric.best: {pm.get('best')}\n")

    # Best metric (simple): pick max current.mean if available, otherwise N/A.
    best = None
    for m in members:
        pm = m.get("primary_metric") or {}
        if not isinstance(pm, dict):
            continue
        cur = _primary_metric_current_mean(pm)
        direction = str(pm.get("direction") or "maximize").strip().lower()
        if cur is None:
            continue
        score = -float(cur) if direction == "minimize" else float(cur)
        if best is None or float(score) > float(best["score"]):
            best = {"project_id": m["project_id"], "value": float(cur), "direction": direction, "score": float(score)}
    lines.append("\n## Best (by primary_metric.current.mean)\n")
    if best is None:
        lines.append("- (no numeric primary metrics yet)\n")
    else:
        lines.append(f"- {best['project_id']}: {best['value']} ({best['direction']})\n")

    lines.append("\n## Next actions\n")
    lines.append("- Review each member's `notes/analysis_digest.md` and pick one branch to continue.\n")
    lines.append("- If using Idea Bank, link outcomes back to the base project ideas and update parked/rejected reasons.\n")

    out_rel = f"notes/lab_meeting/cohort-{run_id}.md"
    put_artifact(
        ledger=ledger,
        project=base_project,
        relative_path=out_rel,
        content="".join(lines),
        mode="overwrite",
        kind="lab_meeting_md",
    )

    return {"base_project_id": base_project_id, "run_id": run_id, "members": members, "lab_meeting_path": out_rel}
