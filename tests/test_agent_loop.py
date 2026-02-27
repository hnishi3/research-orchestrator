from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from resorch.agent_loop import STAGE_ORDER, AgentLoopConfig, _is_stage_backward, load_agent_loop_config, run_agent_loop
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project, get_project


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def _write_scoreboard(workspace: Path, mean: float = 0.5) -> None:
    """Write a minimal scoreboard so that metric watchdogs do not fire."""
    results = workspace / "results"
    results.mkdir(parents=True, exist_ok=True)
    sb = {"primary_metric": {"current": {"mean": mean}}}
    (results / "scoreboard.json").write_text(
        json.dumps(sb, indent=2) + "\n", encoding="utf-8",
    )


def test_agent_loop_runs_one_step_and_creates_review_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )
    _write_scoreboard(Path(project["repo_path"]))

    def fake_autopilot(**kwargs):  # noqa: ANN003
        step = int(kwargs.get("iteration", 0))
        return {
            "plan": {
                "plan_id": "plan1",
                "project_id": project["id"],
                "iteration": step,
                "objective": kwargs.get("objective", ""),
                "self_confidence": 0.5,
                "evidence_strength": 0.5,
                "actions": [],
                "should_stop": True,
                "stop_reason": "done",
            },
            "plan_artifact_path": "notes/autopilot/plan.json",
            "started_at": "2026-01-31T00:00:00Z",
            "project_stage": {"before": "intake", "after": "intake", "changed": False},
            "tasks_created": [],
            "tasks_ran": [],
            "git_change_summary": {"is_git": False, "changed_lines": 0, "changed_files": 0, "changed_paths": []},
            "failure_streaks": {"any": 0, "same_task": 0},
            "stalled_jobs": [],
            "ready_stage_transitions": [],
            "review_recommendation": {"level": "soft", "reasons": ["test"], "targets": ["notes/problem.md"]},
        }

    created_jobs: List[Dict[str, Any]] = []

    def fake_create_job(*, ledger, project_id, provider, kind, spec):  # noqa: ANN001
        created_jobs.append({"provider": provider, "kind": kind, "spec": spec})
        return {"id": "job1", "provider": provider, "kind": kind, "spec": spec}

    def fake_run_job(*, ledger, job_id):  # noqa: ANN001
        return {"id": job_id, "status": "succeeded"}

    monkeypatch.setattr("resorch.agent_loop.run_autopilot_iteration", fake_autopilot)
    monkeypatch.setattr("resorch.agent_loop.create_job", fake_create_job)
    monkeypatch.setattr("resorch.agent_loop.run_job", fake_run_job)

    out = run_agent_loop(
        ledger=ledger,
        project_id=project["id"],
        objective="Do something",
        max_steps=5,
        dry_run=False,
        config_path=None,
    )
    assert len(out["steps"]) == 1
    assert created_jobs and created_jobs[0]["kind"] == "review"

    ws = Path(project["repo_path"])
    # Step files now live under a timestamped run directory.
    step_matches = list((ws / "runs" / "agent").glob("run-*/step_000/step.json"))
    assert len(step_matches) == 1
    step_path = step_matches[0]
    loaded = json.loads(step_path.read_text(encoding="utf-8"))
    assert loaded["step"] == 0


def test_agent_loop_lightweight_retry_schedules_rerun_and_skips_planner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    repo_root = ledger.paths.root
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "review_policy.yaml").write_text(
        "\n".join(
            [
                "policy_version: 1",
                "lightweight_retry:",
                "  enabled: true",
                "  max_consecutive: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="intake", git_init=False)
    _write_scoreboard(Path(project["repo_path"]))

    calls: List[Dict[str, Any]] = []

    def fake_autopilot(**kwargs):  # noqa: ANN003
        calls.append(dict(kwargs))
        step = int(kwargs.get("iteration", 0))
        rerun = bool(kwargs.get("rerun_mode", False))
        actions = [
            {"title": "code", "task_type": "codex_exec", "spec": {"prompt": "x"}},
            {"title": "run", "task_type": "shell_exec", "spec": {"cd": ".", "command": "echo ok", "shell": True}},
        ]
        return {
            "plan": {
                "plan_id": f"plan{step}",
                "project_id": project["id"],
                "iteration": step,
                "objective": kwargs.get("objective", ""),
                "self_confidence": 0.5,
                "evidence_strength": 0.5,
                "actions": actions,
                "should_stop": bool(rerun),
                "stop_reason": "done" if rerun else None,
            },
            "plan_artifact_path": "notes/autopilot/plan.json",
            "started_at": "2026-01-31T00:00:00Z",
            "project_stage": {"before": "intake", "after": "intake", "changed": False},
            "tasks_created": [],
            "tasks_ran": [],
            "git_change_summary": {"is_git": False, "changed_lines": 0, "changed_files": 0, "changed_paths": []},
            "failure_streaks": {"any": 0, "same_task": 0},
            "stalled_jobs": [],
            "ready_stage_transitions": [],
            "review_recommendation": {
                "level": "none" if rerun else "soft",
                "reasons": ["test"],
                "targets": ["notes/problem.md"],
            },
        }

    def fake_create_job(*, ledger, project_id, provider, kind, spec):  # noqa: ANN001
        return {"id": "job1", "provider": provider, "kind": kind, "spec": spec}

    def fake_run_job(*, ledger, job_id):  # noqa: ANN001
        return {
            "id": job_id,
            "status": "succeeded",
            "result": {"review_result": {"overall": "ok", "recommendation": "minor", "findings": [{"severity": "minor", "category": "paths"}]}},
        }

    monkeypatch.setattr("resorch.agent_loop.run_autopilot_iteration", fake_autopilot)
    monkeypatch.setattr("resorch.agent_loop.create_job", fake_create_job)
    monkeypatch.setattr("resorch.agent_loop.run_job", fake_run_job)
    monkeypatch.setattr("resorch.agent_loop.list_tasks", lambda *_args, **_kwargs: [])

    out = run_agent_loop(
        ledger=ledger,
        project_id=project["id"],
        objective="Do something",
        max_steps=3,
        dry_run=False,
        config_path=None,
    )

    assert len(out["steps"]) == 2
    assert len(calls) == 2
    assert calls[0].get("rerun_mode") is False
    assert calls[1].get("rerun_mode") is True
    assert isinstance(calls[1].get("rerun_actions"), list)
    assert len(calls[1]["rerun_actions"]) == 1
    assert calls[1]["rerun_actions"][0]["task_type"] == "shell_exec"


def test_agent_loop_lightweight_retry_not_triggered_on_major(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    repo_root = ledger.paths.root
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "review_policy.yaml").write_text(
        "\n".join(
            [
                "policy_version: 1",
                "lightweight_retry:",
                "  enabled: true",
                "  max_consecutive: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="intake", git_init=False)

    calls: List[Dict[str, Any]] = []

    def fake_autopilot(**kwargs):  # noqa: ANN003
        calls.append(dict(kwargs))
        step = int(kwargs.get("iteration", 0))
        return {
            "plan": {"plan_id": f"plan{step}", "project_id": project["id"], "iteration": step, "objective": "", "self_confidence": 0.5, "evidence_strength": 0.5, "actions": [], "should_stop": True, "stop_reason": "done"},
            "plan_artifact_path": "notes/autopilot/plan.json",
            "started_at": "2026-01-31T00:00:00Z",
            "project_stage": {"before": "intake", "after": "intake", "changed": False},
            "tasks_created": [],
            "tasks_ran": [],
            "git_change_summary": {"is_git": False, "changed_lines": 0, "changed_files": 0, "changed_paths": []},
            "failure_streaks": {"any": 0, "same_task": 0},
            "stalled_jobs": [],
            "ready_stage_transitions": [],
            "review_recommendation": {"level": "soft", "reasons": ["test"], "targets": ["notes/problem.md"]},
        }

    def fake_create_job(*, ledger, project_id, provider, kind, spec):  # noqa: ANN001
        return {"id": "job1", "provider": provider, "kind": kind, "spec": spec}

    def fake_run_job(*, ledger, job_id):  # noqa: ANN001
        return {
            "id": job_id,
            "status": "succeeded",
            "result": {"review_result": {"overall": "ok", "recommendation": "major", "findings": [{"severity": "major", "category": "paths"}]}},
        }

    monkeypatch.setattr("resorch.agent_loop.run_autopilot_iteration", fake_autopilot)
    monkeypatch.setattr("resorch.agent_loop.create_job", fake_create_job)
    monkeypatch.setattr("resorch.agent_loop.run_job", fake_run_job)
    monkeypatch.setattr("resorch.agent_loop.list_tasks", lambda *_args, **_kwargs: [])

    out = run_agent_loop(
        ledger=ledger,
        project_id=project["id"],
        objective="Do something",
        max_steps=3,
        dry_run=False,
        config_path=None,
    )
    assert len(out["steps"]) == 1
    assert len(calls) == 1
    assert calls[0].get("rerun_mode") is False


def test_agent_loop_lightweight_retry_not_triggered_on_novelty_category(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    repo_root = ledger.paths.root
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "review_policy.yaml").write_text(
        "\n".join(
            [
                "policy_version: 1",
                "lightweight_retry:",
                "  enabled: true",
                "  max_consecutive: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="intake", git_init=False)

    calls: List[Dict[str, Any]] = []

    def fake_autopilot(**kwargs):  # noqa: ANN003
        calls.append(dict(kwargs))
        step = int(kwargs.get("iteration", 0))
        return {
            "plan": {"plan_id": f"plan{step}", "project_id": project["id"], "iteration": step, "objective": "", "self_confidence": 0.5, "evidence_strength": 0.5, "actions": [], "should_stop": True, "stop_reason": "done"},
            "plan_artifact_path": "notes/autopilot/plan.json",
            "started_at": "2026-01-31T00:00:00Z",
            "project_stage": {"before": "intake", "after": "intake", "changed": False},
            "tasks_created": [],
            "tasks_ran": [],
            "git_change_summary": {"is_git": False, "changed_lines": 0, "changed_files": 0, "changed_paths": []},
            "failure_streaks": {"any": 0, "same_task": 0},
            "stalled_jobs": [],
            "ready_stage_transitions": [],
            "review_recommendation": {"level": "soft", "reasons": ["test"], "targets": ["notes/problem.md"]},
        }

    def fake_create_job(*, ledger, project_id, provider, kind, spec):  # noqa: ANN001
        return {"id": "job1", "provider": provider, "kind": kind, "spec": spec}

    def fake_run_job(*, ledger, job_id):  # noqa: ANN001
        return {
            "id": job_id,
            "status": "succeeded",
            "result": {"review_result": {"overall": "ok", "recommendation": "minor", "findings": [{"severity": "minor", "category": "novelty"}]}},
        }

    monkeypatch.setattr("resorch.agent_loop.run_autopilot_iteration", fake_autopilot)
    monkeypatch.setattr("resorch.agent_loop.create_job", fake_create_job)
    monkeypatch.setattr("resorch.agent_loop.run_job", fake_run_job)
    monkeypatch.setattr("resorch.agent_loop.list_tasks", lambda *_args, **_kwargs: [])

    out = run_agent_loop(
        ledger=ledger,
        project_id=project["id"],
        objective="Do something",
        max_steps=3,
        dry_run=False,
        config_path=None,
    )
    assert len(out["steps"]) == 1
    assert len(calls) == 1
    assert calls[0].get("rerun_mode") is False


def test_agent_loop_lightweight_retry_max_consecutive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    repo_root = ledger.paths.root
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "review_policy.yaml").write_text(
        "\n".join(
            [
                "policy_version: 1",
                "lightweight_retry:",
                "  enabled: true",
                "  max_consecutive: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="intake", git_init=False)
    _write_scoreboard(Path(project["repo_path"]))

    calls: List[Dict[str, Any]] = []

    def fake_autopilot(**kwargs):  # noqa: ANN003
        calls.append(dict(kwargs))
        step = int(kwargs.get("iteration", 0))
        actions = [
            {"title": "run", "task_type": "shell_exec", "spec": {"cd": ".", "command": "echo ok", "shell": True}},
        ]
        return {
            "plan": {
                "plan_id": f"plan{step}",
                "project_id": project["id"],
                "iteration": step,
                "objective": "",
                "self_confidence": 0.5,
                "evidence_strength": 0.5,
                "actions": actions,
                "should_stop": False,
            },
            "plan_artifact_path": "notes/autopilot/plan.json",
            "started_at": "2026-01-31T00:00:00Z",
            "project_stage": {"before": "intake", "after": "intake", "changed": False},
            "tasks_created": [],
            "tasks_ran": [],
            "git_change_summary": {"is_git": False, "changed_lines": 0, "changed_files": 0, "changed_paths": []},
            "failure_streaks": {"any": 0, "same_task": 0},
            "stalled_jobs": [],
            "ready_stage_transitions": [],
            "review_recommendation": {"level": "soft", "reasons": ["test"], "targets": ["notes/problem.md"]},
        }

    def fake_create_job(*, ledger, project_id, provider, kind, spec):  # noqa: ANN001
        return {"id": f"job{len(calls)}", "provider": provider, "kind": kind, "spec": spec}

    def fake_run_job(*, ledger, job_id):  # noqa: ANN001
        return {
            "id": job_id,
            "status": "succeeded",
            "result": {"review_result": {"overall": "ok", "recommendation": "minor", "findings": [{"severity": "minor", "category": "paths"}]}},
        }

    monkeypatch.setattr("resorch.agent_loop.run_autopilot_iteration", fake_autopilot)
    monkeypatch.setattr("resorch.agent_loop.create_job", fake_create_job)
    monkeypatch.setattr("resorch.agent_loop.run_job", fake_run_job)
    monkeypatch.setattr("resorch.agent_loop.list_tasks", lambda *_args, **_kwargs: [])

    out = run_agent_loop(
        ledger=ledger,
        project_id=project["id"],
        objective="Do something",
        max_steps=4,
        dry_run=False,
        config_path=None,
    )
    assert len(out["steps"]) == 4
    rerun_flags = [bool(c.get("rerun_mode", False)) for c in calls]
    assert rerun_flags[:4] == [False, True, True, False]


def test_agent_loop_flags_codex_stall_after_consecutive_blocked_iterations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="intake", git_init=False)
    _write_scoreboard(Path(project["repo_path"]))

    def fake_autopilot(**kwargs):  # noqa: ANN003
        step = int(kwargs.get("iteration", 0))
        should_stop = step >= 2
        return {
            "plan": {
                "plan_id": f"plan{step}",
                "project_id": project["id"],
                "iteration": step,
                "objective": kwargs.get("objective", ""),
                "self_confidence": 0.5,
                "evidence_strength": 0.5,
                "actions": [{"task_type": "codex_exec", "spec": {"prompt": "x"}}],
                "should_stop": should_stop,
                "stop_reason": "done" if should_stop else None,
            },
            "plan_artifact_path": "notes/autopilot/plan.json",
            "started_at": "2026-01-31T00:00:00Z",
            "project_stage": {"before": "intake", "after": "intake", "changed": False},
            "tasks_created": [],
            "tasks_ran": [
                {"id": f"t{step}", "type": "codex_exec", "status": "blocked"},
            ],
            "git_change_summary": {"is_git": False, "changed_lines": 0, "changed_files": 0, "changed_paths": []},
            "failure_streaks": {"any": 1, "same_task": 1},
            "stalled_jobs": [],
            "ready_stage_transitions": [],
            "review_recommendation": {"level": "none", "reasons": [], "targets": []},
        }

    monkeypatch.setattr("resorch.agent_loop.run_autopilot_iteration", fake_autopilot)

    out = run_agent_loop(
        ledger=ledger,
        project_id=project["id"],
        objective="Detect Codex stall",
        max_steps=5,
        dry_run=False,
        config_path=None,
    )

    assert len(out["steps"]) == 3
    assert out["steps"][0]["autopilot"].get("codex_stall_detected") is False
    assert out["steps"][1]["autopilot"].get("codex_stall_detected") is True
    assert out["steps"][2]["autopilot"].get("codex_stall_detected") is True
    assert out["steps"][2]["autopilot"].get("consecutive_codex_blocked") == 3

    assert any(e.get("event") == "codex_stall_warning" for e in out["steps"][1]["decision_log"])
    assert any(e.get("event") == "codex_stall_error" for e in out["steps"][2]["decision_log"])

    expected_note = (
        "STALL DETECTED: 3 consecutive iterations with all Codex tasks blocked. "
        "Recommend stopping and diagnosing Codex CLI."
    )
    assert expected_note in out["steps"][2]["notes"]


def test_stall_detector_fires_on_consecutive_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="intake", git_init=False)
    _write_scoreboard(Path(project["repo_path"]))

    def fake_autopilot(**kwargs):  # noqa: ANN003
        step = int(kwargs.get("iteration", 0))
        return {
            "plan": {
                "plan_id": f"plan{step}",
                "project_id": project["id"],
                "iteration": step,
                "objective": kwargs.get("objective", ""),
                "self_confidence": 0.5,
                "evidence_strength": 0.5,
                "actions": [{"task_type": "codex_exec", "spec": {"prompt": "x"}}],
                "should_stop": step >= 2,
                "stop_reason": "done" if step >= 2 else None,
            },
            "plan_artifact_path": "notes/autopilot/plan.json",
            "started_at": "2026-01-31T00:00:00Z",
            "project_stage": {"before": "intake", "after": "intake", "changed": False},
            "tasks_created": [],
            "tasks_ran": [{"id": f"t{step}", "type": "codex_exec", "status": "blocked"}],
            "git_change_summary": {"is_git": False, "changed_lines": 0, "changed_files": 0, "changed_paths": []},
            "failure_streaks": {"any": 1, "same_task": 1},
            "stalled_jobs": [],
            "ready_stage_transitions": [],
            "review_recommendation": {"level": "none", "reasons": [], "targets": []},
        }

    monkeypatch.setattr("resorch.agent_loop.run_autopilot_iteration", fake_autopilot)
    out = run_agent_loop(
        ledger=ledger,
        project_id=project["id"],
        objective="Detect Codex stall",
        max_steps=5,
        dry_run=False,
        config_path=None,
    )

    assert len(out["steps"]) == 3
    assert out["steps"][-1]["autopilot"].get("codex_stall_detected") is True
    assert out["steps"][-1]["autopilot"].get("consecutive_codex_blocked") == 3


# ---------------------------------------------------------------------------
# Stage auto-update tests
# ---------------------------------------------------------------------------


def _write_auto_stage_policy(repo_root: Path, *, enabled: bool = True) -> None:
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "review_policy.yaml").write_text(
        "\n".join(
            [
                "policy_version: 1",
                "auto_stage_update:",
                f"  enabled: {'true' if enabled else 'false'}",
                "  apply_on: [accept, minor]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _fake_autopilot_with_stage(project_id: str, *, next_stage: str, current_stage: str = "implementation"):
    def fake(**kwargs):  # noqa: ANN003
        step = int(kwargs.get("iteration", 0))
        return {
            "plan": {
                "plan_id": f"plan{step}",
                "project_id": project_id,
                "iteration": step,
                "objective": kwargs.get("objective", ""),
                "self_confidence": 0.5,
                "evidence_strength": 0.5,
                "actions": [],
                "should_stop": True,
                "stop_reason": "done",
            },
            "plan_artifact_path": "notes/autopilot/plan.json",
            "started_at": "2026-01-31T00:00:00Z",
            "project_stage": {"before": current_stage, "after": current_stage, "changed": False, "requested": next_stage},
            "tasks_created": [],
            "tasks_ran": [],
            "git_change_summary": {"is_git": False, "changed_lines": 0, "changed_files": 0, "changed_paths": []},
            "failure_streaks": {"any": 0, "same_task": 0},
            "stalled_jobs": [],
            "ready_stage_transitions": [],
            "review_recommendation": {"level": "hard", "reasons": ["stage_transition"], "targets": ["notes/problem.md"]},
        }

    return fake


def test_auto_stage_forward_review_accept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    _write_auto_stage_policy(ledger.paths.root)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="implementation", git_init=False)
    _write_scoreboard(Path(project["repo_path"]))

    monkeypatch.setattr(
        "resorch.agent_loop.run_autopilot_iteration",
        _fake_autopilot_with_stage(project["id"], next_stage="analysis", current_stage="implementation"),
    )
    monkeypatch.setattr("resorch.agent_loop.create_job", lambda **kw: {"id": "job1", **kw})
    monkeypatch.setattr(
        "resorch.agent_loop.run_job",
        lambda **kw: {
            "id": kw["job_id"],
            "status": "succeeded",
            "result": {"review_result": {"overall": "ok", "recommendation": "accept", "findings": []}},
        },
    )

    out = run_agent_loop(ledger=ledger, project_id="p1", objective="test", max_steps=1, dry_run=False)
    step = out["steps"][0]
    assert step.get("stage_update", {}).get("applied") is True
    assert step["stage_update"]["direction"] == "forward"
    assert step["stage_update"]["to"] == "analysis"
    assert get_project(ledger, "p1")["stage"] == "analysis"


def test_auto_stage_forward_review_reject(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    _write_auto_stage_policy(ledger.paths.root)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="implementation", git_init=False)
    _write_scoreboard(Path(project["repo_path"]))

    monkeypatch.setattr(
        "resorch.agent_loop.run_autopilot_iteration",
        _fake_autopilot_with_stage(project["id"], next_stage="analysis", current_stage="implementation"),
    )
    monkeypatch.setattr("resorch.agent_loop.create_job", lambda **kw: {"id": "job1", **kw})
    monkeypatch.setattr(
        "resorch.agent_loop.run_job",
        lambda **kw: {
            "id": kw["job_id"],
            "status": "succeeded",
            "result": {"review_result": {"overall": "issues", "recommendation": "reject", "findings": [{"severity": "blocker", "category": "method"}]}},
        },
    )

    out = run_agent_loop(ledger=ledger, project_id="p1", objective="test", max_steps=1, dry_run=False)
    step = out["steps"][0]
    assert step.get("stage_update", {}).get("applied") is False
    assert step["stage_update"]["direction"] == "forward"
    assert get_project(ledger, "p1")["stage"] == "implementation"


def test_auto_stage_backward_applied_unconditionally(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    _write_auto_stage_policy(ledger.paths.root)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)
    _write_scoreboard(Path(project["repo_path"]))

    monkeypatch.setattr(
        "resorch.agent_loop.run_autopilot_iteration",
        _fake_autopilot_with_stage(project["id"], next_stage="method", current_stage="analysis"),
    )
    monkeypatch.setattr("resorch.agent_loop.create_job", lambda **kw: {"id": "job1", **kw})
    # Review rejects, but backward should still apply.
    monkeypatch.setattr(
        "resorch.agent_loop.run_job",
        lambda **kw: {
            "id": kw["job_id"],
            "status": "succeeded",
            "result": {"review_result": {"overall": "issues", "recommendation": "reject", "findings": []}},
        },
    )

    out = run_agent_loop(ledger=ledger, project_id="p1", objective="test", max_steps=1, dry_run=False)
    step = out["steps"][0]
    assert step.get("stage_update", {}).get("applied") is True
    assert step["stage_update"]["direction"] == "backward"
    assert get_project(ledger, "p1")["stage"] == "method"


def test_auto_stage_disabled_no_change(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    _write_auto_stage_policy(ledger.paths.root, enabled=False)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="implementation", git_init=False)
    _write_scoreboard(Path(project["repo_path"]))

    monkeypatch.setattr(
        "resorch.agent_loop.run_autopilot_iteration",
        _fake_autopilot_with_stage(project["id"], next_stage="analysis", current_stage="implementation"),
    )
    monkeypatch.setattr("resorch.agent_loop.create_job", lambda **kw: {"id": "job1", **kw})
    monkeypatch.setattr(
        "resorch.agent_loop.run_job",
        lambda **kw: {
            "id": kw["job_id"],
            "status": "succeeded",
            "result": {"review_result": {"overall": "ok", "recommendation": "accept", "findings": []}},
        },
    )

    out = run_agent_loop(ledger=ledger, project_id="p1", objective="test", max_steps=1, dry_run=False)
    step = out["steps"][0]
    assert step.get("stage_update") is None
    assert get_project(ledger, "p1")["stage"] == "implementation"


def test_is_stage_backward() -> None:
    assert _is_stage_backward("analysis", "method") is True
    assert _is_stage_backward("implementation", "analysis") is False
    assert _is_stage_backward("writing", "intake") is True
    assert _is_stage_backward("intake", "intake") is False
    # Unknown stage → not backward (safe side)
    assert _is_stage_backward("implementation", "unknown_stage") is False
    assert _is_stage_backward("unknown_stage", "intake") is False


# ---------------------------------------------------------------------------
# load_agent_loop_config auto-discovery tests
# ---------------------------------------------------------------------------


def _write_agent_loop_yaml(path: Path, provider: str = "openai", model: str = "gpt-5.2-pro") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"planner:\n  provider: {provider}\n  model: {model}\n",
        encoding="utf-8",
    )


class TestLoadAgentLoopConfig:
    """Tests for load_agent_loop_config workspace fallback."""

    def test_workspace_config_takes_priority(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        workspace = repo_root / "workspaces" / "proj1"
        _write_agent_loop_yaml(repo_root / "configs" / "agent_loop.yaml", provider="claude_code_cli", model="opus")
        _write_agent_loop_yaml(workspace / "configs" / "agent_loop.yaml", provider="openai", model="gpt-5.2-pro")
        cfg = load_agent_loop_config(repo_root, workspace=workspace)
        assert cfg.planner_provider == "openai"
        assert cfg.planner_model == "gpt-5.2-pro"

    def test_falls_back_to_global(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        workspace = repo_root / "workspaces" / "proj1"
        workspace.mkdir(parents=True)
        _write_agent_loop_yaml(repo_root / "configs" / "agent_loop.yaml", provider="claude_code_cli", model="opus")
        cfg = load_agent_loop_config(repo_root, workspace=workspace)
        assert cfg.planner_provider == "claude_code_cli"
        assert cfg.planner_model == "opus"

    def test_falls_back_to_default_when_no_yaml(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        workspace = repo_root / "workspaces" / "proj1"
        workspace.mkdir(parents=True)
        cfg = load_agent_loop_config(repo_root, workspace=workspace)
        # _default_config() uses env vars or hardcoded defaults
        assert isinstance(cfg, AgentLoopConfig)

    def test_explicit_path_overrides_auto_discovery(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        workspace = repo_root / "workspaces" / "proj1"
        _write_agent_loop_yaml(workspace / "configs" / "agent_loop.yaml", provider="openai", model="gpt-5.2-pro")
        explicit = tmp_path / "custom.yaml"
        _write_agent_loop_yaml(explicit, provider="claude_code_cli", model="sonnet")
        cfg = load_agent_loop_config(repo_root, workspace=workspace, explicit_path=str(explicit))
        assert cfg.planner_provider == "claude_code_cli"
        assert cfg.planner_model == "sonnet"

    def test_explicit_path_missing_falls_back_to_default(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        cfg = load_agent_loop_config(repo_root, explicit_path="/nonexistent/path.yaml")
        assert isinstance(cfg, AgentLoopConfig)

    def test_no_workspace_uses_global(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        _write_agent_loop_yaml(repo_root / "configs" / "agent_loop.yaml", provider="claude_code_cli", model="opus")
        cfg = load_agent_loop_config(repo_root, workspace=None)
        assert cfg.planner_provider == "claude_code_cli"
        assert cfg.planner_model == "opus"
