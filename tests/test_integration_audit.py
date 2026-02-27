from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from resorch import autopilot, autopilot_planner, evidence_store, goal_alignment, interpretation_challenger, projects, visual_inspection
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project, get_project


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (repo_root / "schemas").mkdir(parents=True, exist_ok=True)
    (repo_root / "schemas" / "autopilot_plan.schema.json").write_text(
        json.dumps({"type": "object", "properties": {"project_id": {"type": "string"}}}) + "\n",
        encoding="utf-8",
    )
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_idea_to_agent_integration_points_count(tmp_path: Path) -> None:
    claimed_count = 3
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    point_1 = "topic_brief.md" in autopilot._default_planner_context_files(workspace)
    point_2 = "idea_id" in inspect.signature(projects.create_project).parameters
    build_prompt_source = inspect.getsource(autopilot._build_planner_prompt)
    point_3 = "_load_playbook_context(" in build_prompt_source

    verified_count = sum(bool(p) for p in (point_1, point_2, point_3))

    assert point_1, "integration point 1 missing: topic_brief.md not in _default_planner_context_files"
    assert point_2, "integration point 2 missing: create_project has no idea_id parameter"
    assert point_3, "integration point 3 missing: _build_planner_prompt does not call _load_playbook_context"
    assert verified_count == claimed_count


def test_verification_checks_count() -> None:
    claimed_count = 5

    check_1 = callable(getattr(goal_alignment, "check_goal_alignment", None))
    check_2 = callable(getattr(interpretation_challenger, "challenge_interpretation", None)) or callable(
        getattr(interpretation_challenger, "maybe_challenge_interpretation_from_workspace", None)
    )
    check_3 = callable(getattr(autopilot, "_pivot_no_improvement_trigger", None))
    check_4 = callable(getattr(visual_inspection, "get_visual_inspection_status", None))
    check_5 = callable(getattr(evidence_store, "validate_evidence_url", None))

    verified_count = sum(bool(c) for c in (check_1, check_2, check_3, check_4, check_5))

    assert check_1, "verification check missing: goal_alignment.check_goal_alignment"
    assert check_2, "verification check missing: interpretation challenger entrypoint"
    assert check_3, "verification check missing: autopilot._pivot_no_improvement_trigger"
    assert check_4, "verification check missing: visual_inspection.get_visual_inspection_status"
    assert check_5, "verification check missing: evidence_store.validate_evidence_url"
    assert verified_count == claimed_count


def test_integration_points_are_functional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="audit-project",
        title="Integration Audit Project",
        domain="biology",
        stage="analysis",
        git_init=False,
        idea_id="idea-audit-001",
    )
    workspace = Path(project["repo_path"]).resolve()

    default_files = autopilot._default_planner_context_files(workspace)
    assert "topic_brief.md" in default_files

    fetched = get_project(ledger, project["id"])
    assert fetched["meta"]["idea_id"] == "idea-audit-001"

    ledger.upsert_playbook_entry(
        entry_id="ptn_audit",
        topic="biology:robustness",
        rule={"summary": "Use stratified sampling for fair comparisons."},
    )
    playbook_context = autopilot._load_playbook_context(ledger, fetched)
    assert playbook_context.strip() != ""
    assert "ptn_audit" in playbook_context

    called: Dict[str, Any] = {"count": 0}

    def _spy_load_playbook_context(ledger_in: Ledger, project_in: Dict[str, Any]) -> str:
        called["count"] += 1
        assert ledger_in is ledger
        assert project_in["id"] == project["id"]
        return "Playbook lessons (from completed projects):\n- spy: injected\n\n"

    monkeypatch.setattr(autopilot_planner, "_load_playbook_context", _spy_load_playbook_context)
    prompt, _schema, _validator = autopilot._build_planner_prompt(
        ledger=ledger,
        project_id=project["id"],
        objective="Verify integration points with real calls.",
        iteration=0,
        max_actions=2,
        context_files=[],
    )

    assert called["count"] >= 1
    assert "- spy: injected" in prompt
