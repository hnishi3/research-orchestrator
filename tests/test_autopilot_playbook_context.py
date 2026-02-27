from __future__ import annotations

import json
from pathlib import Path

from resorch.autopilot import (
    _build_planner_prompt,
    _default_planner_context_files,
    _load_playbook_context,
)
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project


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


def test_default_planner_context_includes_workspace_playbook_yaml(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    (workspace / "playbook").mkdir(parents=True, exist_ok=True)
    (workspace / "playbook" / "rule-1.yaml").write_text("id: rule_1\n", encoding="utf-8")

    rels = _default_planner_context_files(workspace)
    assert "playbook/rule-1.yaml" in rels


def test_load_playbook_context_filters_by_domain(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="nlp",
        stage="analysis",
        git_init=False,
    )
    ledger.upsert_playbook_entry(
        entry_id="ptn_nlp",
        topic="nlp:cleanup",
        rule={"summary": "Use stratified splits."},
    )
    ledger.upsert_playbook_entry(
        entry_id="ptn_cv",
        topic="cv:augmentation",
        rule={"summary": "Tune crop ratios."},
    )

    section = _load_playbook_context(ledger, project)
    assert "Playbook lessons (from completed projects):" in section
    assert "- ptn_nlp: Use stratified splits." in section
    assert "ptn_cv" not in section


def test_load_playbook_context_empty_domain_uses_default_limit(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p2",
        title="P2",
        domain="",
        stage="analysis",
        git_init=False,
    )
    for i in range(6):
        ledger.upsert_playbook_entry(
            entry_id=f"ptn_{i}",
            topic=f"topic_{i}",
            rule={"summary": f"summary {i}"},
        )

    section = _load_playbook_context(ledger, project)
    lines = [line for line in section.splitlines() if line.startswith("- ")]
    assert len(lines) == 5


def test_build_planner_prompt_injects_playbook_before_reference_materials(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p3",
        title="P3",
        domain="biology",
        stage="analysis",
        git_init=False,
    )
    ledger.upsert_playbook_entry(
        entry_id="ptn_prompt",
        topic="biology:benchmark",
        rule={"summary": "Check data leakage before model selection."},
    )

    prompt, _schema, _validator = _build_planner_prompt(
        ledger=ledger,
        project_id=project["id"],
        objective="Validate prompt sections.",
        iteration=0,
        max_actions=2,
        context_files=[],
    )
    pb_idx = prompt.find("Playbook lessons (from completed projects):")
    ref_idx = prompt.find("Reference materials (workspace files):")
    assert pb_idx != -1
    assert ref_idx != -1
    assert pb_idx < ref_idx
    assert "- ptn_prompt: Check data leakage before model selection." in prompt
