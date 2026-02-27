from __future__ import annotations

import sys
from types import ModuleType
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import pytest
from conftest import make_tmp_repo

from resorch.cli import build_parser
from resorch.portfolio import ProjectState, compute_priority, run_portfolio_cycle
from resorch.projects import create_project, get_project


def _iso_hours_ago(hours: float) -> str:
    ts = datetime.now(timezone.utc) - timedelta(hours=float(hours))
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _set_project_updated_at(ledger: Any, project_id: str, hours_ago: float) -> None:
    ledger._exec(
        "UPDATE projects SET updated_at = ? WHERE id = ?",
        (_iso_hours_ago(hours_ago), project_id),
    )
    ledger._maybe_commit()


def _insert_task_with_status(ledger: Any, project_id: str, status: str) -> None:
    ledger.insert_task(
        task_id=uuid4().hex,
        project_id=project_id,
        task_type="shell_exec",
        status=status,
        spec={},
        deps=[],
    )


def test_compute_priority_prefers_staler_project_when_other_factors_equal() -> None:
    stale = ProjectState("p-stale", "stale", "analysis", 100.0, None, 0, 0.0)
    fresh = ProjectState("p-fresh", "fresh", "analysis", 10.0, None, 0, 0.0)
    assert compute_priority(stale) > compute_priority(fresh)


def test_compute_priority_penalizes_fail_streak() -> None:
    healthy = ProjectState("p-ok", "ok", "analysis", 80.0, None, 0, 0.0)
    failing = ProjectState("p-fail", "fail", "analysis", 80.0, None, 2, 0.0)
    assert compute_priority(healthy) > compute_priority(failing)


def test_compute_priority_stage_bonus_intake_higher_than_writing() -> None:
    intake = ProjectState("p-intake", "intake", "intake", 20.0, None, 0, 0.0)
    writing = ProjectState("p-writing", "writing", "writing", 20.0, None, 0, 0.0)
    assert compute_priority(intake) > compute_priority(writing)


def test_portfolio_cycle_empty_ledger_dry_run(tmp_path: Path) -> None:
    ledger = make_tmp_repo(tmp_path)
    out = run_portfolio_cycle(ledger=ledger, dry_run=True)
    assert out["projects_evaluated"] == 0
    assert out["projects_selected"] == 0
    assert out["projects_executed"] == 0
    assert out["projects_completed"] == 0
    assert out["playbook_extractions"] == 0


def test_portfolio_cycle_excludes_done_and_parked(tmp_path: Path) -> None:
    ledger = make_tmp_repo(tmp_path)
    create_project(ledger=ledger, project_id="p_active", title="Active", domain="", stage="analysis", git_init=False)
    create_project(ledger=ledger, project_id="p_done", title="Done", domain="", stage="done", git_init=False)
    create_project(ledger=ledger, project_id="p_parked", title="Parked", domain="", stage="parked", git_init=False)

    out = run_portfolio_cycle(ledger=ledger, max_projects=5, dry_run=True)
    assert out["projects_evaluated"] == 1
    assert out["projects_selected"] == 1
    assert out["decision_log"][0]["project_ids"] == ["p_active"]


def test_portfolio_cycle_selects_top_k_by_priority(tmp_path: Path) -> None:
    ledger = make_tmp_repo(tmp_path)
    create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)
    create_project(ledger=ledger, project_id="p2", title="P2", domain="", stage="intake", git_init=False)
    create_project(ledger=ledger, project_id="p3", title="P3", domain="", stage="analysis", git_init=False)

    _set_project_updated_at(ledger, "p1", hours_ago=50)
    _set_project_updated_at(ledger, "p2", hours_ago=30)
    _set_project_updated_at(ledger, "p3", hours_ago=80)
    _insert_task_with_status(ledger, "p3", "failed")
    _insert_task_with_status(ledger, "p3", "failed")

    out = run_portfolio_cycle(ledger=ledger, max_projects=2, dry_run=True)
    assert out["projects_selected"] == 2
    assert out["decision_log"][0]["project_ids"] == ["p1", "p2"]


def test_portfolio_cycle_dry_run_does_not_execute_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = make_tmp_repo(tmp_path)
    create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)

    def _should_not_run(**_kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("run_agent_loop should not be called in dry_run mode")

    fake_agent_loop = ModuleType("resorch.agent_loop")
    fake_agent_loop.run_agent_loop = _should_not_run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "resorch.agent_loop", fake_agent_loop)
    out = run_portfolio_cycle(ledger=ledger, max_projects=1, dry_run=True)
    assert out["projects_executed"] == 0


def test_portfolio_cycle_executes_selected_projects_and_passes_step_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = make_tmp_repo(tmp_path)
    create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)
    create_project(ledger=ledger, project_id="p2", title="P2", domain="", stage="analysis", git_init=False)
    _set_project_updated_at(ledger, "p1", hours_ago=100)
    _set_project_updated_at(ledger, "p2", hours_ago=1)

    calls: List[Dict[str, Any]] = []

    def _fake_run_agent_loop(**kwargs: Any) -> Dict[str, Any]:
        calls.append(dict(kwargs))
        return {"project_id": kwargs["project_id"], "steps": [], "stopped_reason": "max_steps"}

    fake_agent_loop = ModuleType("resorch.agent_loop")
    fake_agent_loop.run_agent_loop = _fake_run_agent_loop  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "resorch.agent_loop", fake_agent_loop)
    out = run_portfolio_cycle(ledger=ledger, max_projects=1, steps_per_project=7, dry_run=False)
    assert out["projects_selected"] == 1
    assert out["projects_executed"] == 1
    assert out["projects_completed"] == 0
    assert calls[0]["project_id"] == "p1"
    assert calls[0]["max_steps"] == 7
    assert calls[0]["dry_run"] is False


def test_portfolio_cycle_done_marks_done_and_triggers_playbook_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = make_tmp_repo(tmp_path)
    create_project(ledger=ledger, project_id="p1", title="P1", domain="bio", stage="analysis", git_init=False)
    extracted: List[str] = []

    def _fake_run_agent_loop(**kwargs: Any) -> Dict[str, Any]:
        return {"project_id": kwargs["project_id"], "steps": [], "stopped_reason": "done"}

    def _fake_extract_and_save(ledger: Any, project_id: str, mode: str = "compact") -> Dict[str, Any]:
        extracted.append(project_id)
        return {"entry_id": "pb_x", "yaml_path": "playbook/extracted/x.yaml", "ledger_stored": True}

    fake_agent_loop = ModuleType("resorch.agent_loop")
    fake_agent_loop.run_agent_loop = _fake_run_agent_loop  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "resorch.agent_loop", fake_agent_loop)
    monkeypatch.setattr("resorch.playbook_extractor.extract_and_save", _fake_extract_and_save)

    out = run_portfolio_cycle(ledger=ledger, max_projects=1, steps_per_project=3, dry_run=False)
    assert out["projects_executed"] == 1
    assert out["projects_completed"] == 1
    assert out["playbook_extractions"] == 1
    assert extracted == ["p1"]
    assert get_project(ledger, "p1")["stage"] == "done"


def test_portfolio_triggers_playbook_extraction_on_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = make_tmp_repo(tmp_path)
    impl = create_project(
        ledger=ledger,
        project_id="p_impl",
        title="Implementation Project",
        domain="bio",
        stage="implementation",
        git_init=False,
    )
    done = create_project(
        ledger=ledger,
        project_id="p_done",
        title="Already Done Project",
        domain="bio",
        stage="done",
        git_init=False,
    )
    done_workspace = Path(done["repo_path"])
    (done_workspace / "notes" / "analysis_digest.md").write_text(
        "# Analysis Digest\n\n## Latest\n- completed.\n",
        encoding="utf-8",
    )
    (done_workspace / "results" / "scoreboard.json").write_text(
        '{"primary_metric":{"name":"score","direction":"maximize","current":1.0,"baseline":0.5}}\n',
        encoding="utf-8",
    )

    transitioned: List[str] = []
    extracted: List[str] = []

    def _fake_run_agent_loop(**kwargs: Any) -> Dict[str, Any]:
        project_id = kwargs["project_id"]
        transitioned.append(project_id)
        # Simulate the agent finishing implementation work before portfolio
        # runs its completion hook.
        ledger.update_project_stage(project_id, "done")
        return {"project_id": project_id, "steps": [], "stopped_reason": "done"}

    def _fake_extract_and_save(ledger: Any, project_id: str, mode: str = "compact") -> Dict[str, Any]:
        extracted.append(project_id)
        return {"entry_id": "pb_mock", "yaml_path": "playbook/extracted/mock.yaml", "ledger_stored": True}

    fake_agent_loop = ModuleType("resorch.agent_loop")
    fake_agent_loop.run_agent_loop = _fake_run_agent_loop  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "resorch.agent_loop", fake_agent_loop)
    monkeypatch.setattr("resorch.playbook_extractor.extract_and_save", _fake_extract_and_save)

    out = run_portfolio_cycle(ledger=ledger, max_projects=2, steps_per_project=3, dry_run=False)

    assert out["projects_evaluated"] == 1
    assert out["projects_selected"] == 1
    assert out["projects_executed"] == 1
    assert out["projects_completed"] == 1
    assert out["playbook_extractions"] == 1
    assert transitioned == ["p_impl"]
    assert extracted == ["p_impl"]
    assert get_project(ledger, "p_impl")["stage"] == "done"
    assert get_project(ledger, str(impl["id"]))["stage"] == "done"
    assert get_project(ledger, "p_done")["stage"] == "done"


def test_portfolio_cycle_logs_agent_loop_unavailable_when_import_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = make_tmp_repo(tmp_path)
    create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)
    monkeypatch.setitem(sys.modules, "resorch.agent_loop", None)

    out = run_portfolio_cycle(ledger=ledger, max_projects=1, steps_per_project=2, dry_run=False)
    assert out["projects_selected"] == 1
    assert out["projects_executed"] == 0
    assert any(e.get("event") == "agent_loop_unavailable" for e in out["decision_log"])


def test_portfolio_cli_parser_cycle_subcommand() -> None:
    args = build_parser().parse_args(
        [
            "portfolio",
            "cycle",
            "--max-projects",
            "4",
            "--steps-per-project",
            "9",
            "--dry-run",
        ]
    )
    assert args._handler == "portfolio_cycle"
    assert args.max_projects == 4
    assert args.steps_per_project == 9
    assert args.dry_run is True
