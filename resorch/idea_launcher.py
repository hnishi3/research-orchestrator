from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.ideas import get_idea as get_idea_fn
from resorch.ideas import set_idea_status as set_idea_status_fn
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.topic_brief import write_topic_brief as write_topic_brief_fn
from resorch.utils import utc_now_iso


_LAUNCHABLE_STATUSES = {"candidate", "active", "smoke_passed", "selected"}


def _as_list_of_str(value: Any) -> List[str]:
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out


def _smoke_lines(smoke_row: Optional[Dict[str, Any]]) -> List[str]:
    if smoke_row is None:
        return ["- (no smoke test results recorded)"]

    result = smoke_row.get("result")
    if not isinstance(result, dict):
        result = {}

    lines: List[str] = []
    verdict = str(smoke_row.get("verdict") or result.get("verdict") or "unknown")
    lines.append(f"- verdict: `{verdict}`")

    started_at = str(smoke_row.get("started_at") or result.get("started_at") or "").strip()
    completed_at = str(smoke_row.get("completed_at") or result.get("completed_at") or "").strip()
    if started_at:
        lines.append(f"- started_at: `{started_at}`")
    if completed_at:
        lines.append(f"- completed_at: `{completed_at}`")

    metrics = result.get("metrics")
    if isinstance(metrics, list) and metrics:
        lines.append("- metrics:")
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            name = str(metric.get("name") or "").strip()
            if not name:
                continue
            value = metric.get("value")
            unit = str(metric.get("unit") or "").strip()
            unit_suffix = f" {unit}" if unit else ""
            lines.append(f"  - {name}: {value}{unit_suffix}")

    return lines


def _write_problem_md(
    *,
    workspace: Path,
    idea_id: str,
    idea_data: Dict[str, Any],
    objective: str,
    smoke_row: Optional[Dict[str, Any]],
    topic_brief_rel_path: str,
) -> Path:
    title = str(idea_data.get("title") or "").strip() or f"Idea {idea_id}"
    description = str(idea_data.get("description") or "").strip() or "(missing)"
    objectives = _as_list_of_str(idea_data.get("objectives"))
    success_criteria = _as_list_of_str(idea_data.get("success_criteria"))

    lines: List[str] = [
        "# Problem",
        "",
        f"- idea_id: `{idea_id}`",
        f"- created_at: `{utc_now_iso()}`",
        "",
        "## Idea",
        "",
        f"- title: {title}",
        f"- description: {description}",
        "",
        "## Objective",
        "",
        f"- {objective}",
        "",
        "## Objectives",
        "",
    ]
    if objectives:
        lines.extend(f"- {item}" for item in objectives)
    else:
        lines.append("- (none provided)")

    lines.extend(["", "## Success Criteria", ""])
    if success_criteria:
        lines.extend(f"- {item}" for item in success_criteria)
    else:
        lines.append("- (none provided)")

    lines.extend(["", "## Smoke Test Results", ""])
    lines.extend(_smoke_lines(smoke_row))

    lines.extend(
        [
            "",
            "## Linked Artifacts",
            "",
            f"- topic_brief: `{topic_brief_rel_path}`",
            "",
        ]
    )

    out_path = workspace / "notes" / "problem.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path


def commit_and_launch(
    *,
    ledger: Ledger,
    repo_paths: RepoPaths,
    idea_id: str,
    project_title: Optional[str] = None,
    domain: Optional[str] = None,
    objective: Optional[str] = None,
    dry_run: bool = False,
    max_steps: int = 1,
) -> Dict[str, Any]:
    """Full pipeline: idea → selected → topic_brief → project → launch-ready."""
    if repo_paths.root.resolve() != ledger.paths.root.resolve():
        raise SystemExit("repo_paths.root must match ledger.paths.root")

    steps_taken: List[str] = []

    idea = get_idea_fn(ledger=ledger, idea_id=idea_id)
    steps_taken.append("validated_idea_exists")

    current_status = str(idea.get("status") or "").strip()
    if current_status not in _LAUNCHABLE_STATUSES:
        allowed = ", ".join(sorted(_LAUNCHABLE_STATUSES))
        raise SystemExit(
            f"Idea {idea_id} is not launchable from status '{current_status}'. "
            f"Expected one of: {allowed}"
        )
    steps_taken.append("validated_launchable_status")

    source_project_id = str(idea.get("project_id") or "").strip()
    source_project = ledger.get_project(source_project_id)
    source_workspace = Path(str(source_project["repo_path"])).resolve()

    idea_data = idea.get("data")
    if not isinstance(idea_data, dict):
        idea_data = {}

    title_from_idea = str(idea_data.get("title") or "").strip() or f"Idea {idea_id}"
    selected_title = str(project_title).strip() if project_title else title_from_idea
    selected_domain = str(domain).strip() if domain else str(idea_data.get("domain") or "").strip()
    if not selected_domain:
        selected_domain = str(source_project.get("domain") or "").strip()
    selected_objective = str(objective).strip() if objective else str(idea_data.get("description") or "").strip()
    if not selected_objective:
        selected_objective = f"Execute idea {idea_id}: {title_from_idea}"

    if current_status != "selected":
        set_idea_status_fn(ledger=ledger, idea_id=idea_id, status="selected")
    steps_taken.append("set_idea_selected")

    topic_brief_result = write_topic_brief_fn(
        ledger=ledger,
        project_id=source_project_id,
        idea_id=idea_id,
        output_path="topic_brief.md",
        register_as_artifact=True,
        set_selected=False,
    )
    source_topic_brief_path = Path(str(topic_brief_result["output_path"])).resolve()
    steps_taken.append("generated_topic_brief")

    project = create_project(
        ledger=ledger,
        project_id=None,
        title=selected_title,
        domain=selected_domain,
        stage="intake",
        git_init=False,
        idea_id=idea_id,
    )
    workspace = Path(str(project["repo_path"])).resolve()
    steps_taken.append("created_project")

    target_topic_brief_path = (workspace / "topic_brief.md").resolve()
    target_topic_brief_path.write_text(
        source_topic_brief_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    steps_taken.append("copied_topic_brief")

    smoke_rows = ledger.list_smoke_tests(project_id=source_project_id, idea_id=idea_id, limit=1)
    smoke_latest: Optional[Dict[str, Any]] = None
    if smoke_rows:
        smoke_latest = dict(smoke_rows[0])
        smoke_latest["result"] = json.loads(smoke_latest.pop("result_json") or "{}")

    problem_path = _write_problem_md(
        workspace=workspace,
        idea_id=idea_id,
        idea_data=idea_data,
        objective=selected_objective,
        smoke_row=smoke_latest,
        topic_brief_rel_path="topic_brief.md",
    )
    steps_taken.append("wrote_problem_md")

    if not dry_run:
        set_idea_status_fn(ledger=ledger, idea_id=idea_id, status="in_progress")
        steps_taken.append("set_idea_in_progress")

    launch_cmd = (
        f"orchestrator agent run --project {shlex.quote(str(project['id']))} "
        f"--objective {shlex.quote(selected_objective)} --max-steps {int(max_steps)}"
    )

    return {
        "idea_id": idea_id,
        "project_id": str(project["id"]),
        "workspace_path": str(workspace),
        "source_workspace_path": str(source_workspace),
        "topic_brief_path": str(target_topic_brief_path),
        "problem_path": str(problem_path),
        "objective": selected_objective,
        "domain": selected_domain,
        "project_title": selected_title,
        "dry_run": bool(dry_run),
        "max_steps": int(max_steps),
        "agent_run_command": launch_cmd,
        "steps_taken": steps_taken,
    }
