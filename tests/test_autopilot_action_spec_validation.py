from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from jsonschema import Draft202012Validator

from resorch.autopilot import (
    _repair_plan_actions_for_runtime,
    _validate_plan_action_semantics,
    run_autopilot_iteration,
)
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project


def _make_tmp_repo(tmp_path: Path) -> Tuple[Ledger, Dict[str, Any]]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "review").mkdir(parents=True, exist_ok=True)
    (repo_root / "review" / "review_result.schema.json").write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"recommendation": {"type": "string"}, "findings": {"type": "array"}},
                "required": ["recommendation", "findings"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )
    return ledger, project


def test_plan_schema_allows_empty_action_spec_but_semantics_reject_shell() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    schema = json.loads((repo_root / "schemas" / "autopilot_plan.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    plan = {
        "plan_id": "p",
        "project_id": "p1",
        "iteration": 0,
        "objective": "obj",
        "self_confidence": 0.5,
        "evidence_strength": 0.5,
        "actions": [{"title": "bad", "task_type": "shell_exec", "spec": {}}],
        "should_stop": False,
    }
    errors = sorted(validator.iter_errors(plan), key=lambda e: e.json_path)
    assert errors == []
    semantic_errors = _validate_plan_action_semantics(plan)
    assert len(semantic_errors) == 1
    assert "spec.command" in semantic_errors[0]


def test_validate_plan_action_semantics_accepts_compat_keys() -> None:
    plan = {
        "actions": [
            {"title": "shell", "task_type": "shell_exec", "spec": {"cmd": "echo ok"}},
            {"title": "code", "task_type": "codex_exec", "spec": {"instructions": "do x"}},
            {"title": "code2", "task_type": "codex_exec", "spec": {"instruction": "do y"}},
        ]
    }
    errors = _validate_plan_action_semantics(plan)
    assert errors == []


def test_validate_plan_action_semantics_rejects_missing_required_fields() -> None:
    plan = {
        "actions": [
            {"title": "bad shell", "task_type": "shell_exec", "spec": {}},
            {"title": "", "task_type": "codex_exec", "spec": {}},
        ]
    }
    errors = _validate_plan_action_semantics(plan)
    assert len(errors) == 2
    assert "spec.command" in errors[0]
    assert "spec.prompt" in errors[1]


def test_run_autopilot_skips_invalid_actions_but_keeps_valid_aliases(monkeypatch, tmp_path: Path) -> None:
    ledger, project = _make_tmp_repo(tmp_path)

    # Keep this focused on action validation; disable optional policy features.
    monkeypatch.setattr(
        "resorch.autopilot.load_review_policy",
        lambda _root, **_kw: {
            "review_phases": {},
            "goal_alignment": {"enabled": False},
            "interpretation_challenger": {"enabled": False},
        },
    )

    reuse_plan = {
        "plan_id": "plan-reuse",
        "project_id": project["id"],
        "iteration": 0,
        "objective": "obj",
        "self_confidence": 0.5,
        "evidence_strength": 0.5,
        "actions": [
            {"title": "", "task_type": "shell_exec", "spec": {}},
            {"title": "", "task_type": "codex_exec", "spec": {}},
            {"title": "good shell alias", "task_type": "shell_exec", "spec": {"cmd": "echo ok"}},
            {"title": "good codex alias", "task_type": "codex_exec", "spec": {"instructions": "Do X"}},
        ],
        "should_stop": False,
    }

    out = run_autopilot_iteration(
        ledger=ledger,
        project_id=project["id"],
        objective="obj",
        model="gpt-5.3-codex",
        iteration=0,
        dry_run=True,
        max_actions=6,
        background=False,
        config={"planner_provider": "codex_cli"},
        reuse_plan=reuse_plan,
    )

    created = out["tasks_created"]
    assert len(created) == 2
    specs = [c["spec"] for c in created]
    assert any(s.get("command") == "echo ok" for s in specs)
    assert any(s.get("prompt") == "Do X" for s in specs)


def test_run_autopilot_repairs_codex_prompt_from_action_title(monkeypatch, tmp_path: Path) -> None:
    ledger, project = _make_tmp_repo(tmp_path)

    monkeypatch.setattr(
        "resorch.autopilot.load_review_policy",
        lambda _root, **_kw: {
            "review_phases": {},
            "goal_alignment": {"enabled": False},
            "interpretation_challenger": {"enabled": False},
        },
    )

    reuse_plan = {
        "plan_id": "plan-reuse-2",
        "project_id": project["id"],
        "iteration": 0,
        "objective": "obj",
        "self_confidence": 0.5,
        "evidence_strength": 0.5,
        "actions": [
            {"title": "Inspect workspace and implement ingestion bootstrap", "task_type": "codex_exec", "spec": {}},
        ],
        "should_stop": False,
    }

    out = run_autopilot_iteration(
        ledger=ledger,
        project_id=project["id"],
        objective="obj",
        model="gpt-5.3-codex",
        iteration=0,
        dry_run=True,
        max_actions=6,
        background=False,
        config={"planner_provider": "codex_cli"},
        reuse_plan=reuse_plan,
    )

    created = out["tasks_created"]
    assert len(created) == 1
    prompt = str(created[0]["spec"].get("prompt") or "")
    assert "Inspect workspace and implement ingestion bootstrap" in prompt


def test_run_autopilot_repairs_shell_missing_command_to_codex(monkeypatch, tmp_path: Path) -> None:
    ledger, project = _make_tmp_repo(tmp_path)

    monkeypatch.setattr(
        "resorch.autopilot.load_review_policy",
        lambda _root, **_kw: {
            "review_phases": {},
            "goal_alignment": {"enabled": False},
            "interpretation_challenger": {"enabled": False},
        },
    )

    reuse_plan = {
        "plan_id": "plan-reuse-3",
        "project_id": project["id"],
        "iteration": 0,
        "objective": "obj",
        "self_confidence": 0.5,
        "evidence_strength": 0.5,
        "actions": [
            {"title": "Run project checks and summarize results", "task_type": "shell_exec", "spec": {}},
        ],
        "should_stop": False,
    }

    out = run_autopilot_iteration(
        ledger=ledger,
        project_id=project["id"],
        objective="obj",
        model="gpt-5.3-codex",
        iteration=0,
        dry_run=True,
        max_actions=6,
        background=False,
        config={"planner_provider": "codex_cli"},
        reuse_plan=reuse_plan,
    )

    created = out["tasks_created"]
    assert len(created) == 1
    assert created[0]["type"] == "codex_exec"
    prompt = str(created[0]["spec"].get("prompt") or "")
    assert "Run project checks and summarize results" in prompt


def test_repair_plan_actions_converts_invalid_shell_to_codex() -> None:
    plan = {
        "actions": [
            {"title": "Run tests and summarize failures", "task_type": "shell_exec", "spec": {}},
        ]
    }
    repaired = _repair_plan_actions_for_runtime(plan)
    actions = repaired.get("actions") or []
    assert len(actions) == 1
    assert actions[0].get("task_type") == "codex_exec"
    prompt = str((actions[0].get("spec") or {}).get("prompt") or "")
    assert "Run tests and summarize failures" in prompt


def test_repair_plan_actions_preserves_action_schema_shape() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    schema = json.loads((repo_root / "schemas" / "autopilot_plan.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    plan = {
        "plan_id": "p",
        "project_id": "p1",
        "iteration": 0,
        "objective": "obj",
        "self_confidence": 0.5,
        "evidence_strength": 0.5,
        "actions": [
            {"title": "Run checks", "task_type": "shell_exec", "spec": {}},
        ],
        "should_stop": False,
    }
    repaired = _repair_plan_actions_for_runtime(plan)
    errors = sorted(validator.iter_errors(repaired), key=lambda e: e.json_path)
    assert errors == []
    action = repaired["actions"][0]
    assert set(action.keys()) <= {"title", "task_type", "spec"}
