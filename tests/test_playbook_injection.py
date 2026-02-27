from __future__ import annotations

import json
from pathlib import Path

from resorch.autopilot import _build_planner_prompt, _default_planner_context_files, _load_playbook_context
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project


def _make_ledger(tmp_path: Path) -> Ledger:
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


def test_default_planner_context_files_include_playbook_yaml(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    (workspace / "playbook").mkdir(parents=True, exist_ok=True)
    (workspace / "playbook" / "lesson.yaml").write_text("id: lesson_1\n", encoding="utf-8")

    rels = _default_planner_context_files(workspace)

    assert "playbook/lesson.yaml" in rels


def test_load_playbook_context_returns_markdown_when_entries_exist(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p-playbook",
        title="Playbook Project",
        domain="bio",
        stage="analysis",
        git_init=False,
    )
    ledger.upsert_playbook_entry(
        entry_id="ptn_bio_1",
        topic="bio:prep",
        rule={"summary": "Validate leakage assumptions before model fitting."},
    )

    section = _load_playbook_context(ledger, project)

    assert section.startswith("Playbook lessons (from completed projects):\n")
    assert "- ptn_bio_1: Validate leakage assumptions before model fitting." in section


def test_load_playbook_context_returns_empty_when_no_entries(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p-empty-playbook",
        title="No Lessons",
        domain="bio",
        stage="analysis",
        git_init=False,
    )

    section = _load_playbook_context(ledger, project)

    assert section == ""


def test_planner_prompt_includes_playbook_context(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p-prompt",
        title="Prompt Project",
        domain="nlp",
        stage="analysis",
        git_init=False,
    )
    ledger.upsert_playbook_entry(
        entry_id="ptn_prompt_1",
        topic="nlp:grounding",
        rule={"summary": "Ground claims in current benchmark evidence."},
    )

    prompt, _schema, _validator = _build_planner_prompt(
        ledger=ledger,
        project_id=project["id"],
        objective="Check playbook prompt injection.",
        iteration=0,
        max_actions=2,
        context_files=[],
    )

    assert "Playbook lessons (from completed projects):" in prompt
    assert "- ptn_prompt_1: Ground claims in current benchmark evidence." in prompt

