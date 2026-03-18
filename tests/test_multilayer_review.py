from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from resorch.agent_loop import run_agent_loop
from resorch.autopilot import run_autopilot_iteration
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.tasks import create_task


def _make_tmp_repo(tmp_path: Path) -> Tuple[Ledger, Dict[str, Any]]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)

    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="analysis",
        git_init=False,
    )
    return ledger, project


def test_pre_exec_review_runs_between_codex_and_shell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger, project = _make_tmp_repo(tmp_path)

    def fake_generate_plan_openai(**_kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return (
            {
                "plan_id": "plan1",
                "project_id": project["id"],
                "iteration": 0,
                "objective": "test",
                "self_confidence": 0.5,
                "evidence_strength": 0.5,
                "actions": [
                    {"title": "code", "task_type": "codex_exec", "spec": {"prompt": "x"}},
                    {"title": "run", "task_type": "shell_exec", "spec": {"command": "echo ok"}},
                ],
                "should_stop": False,
            },
            {},
        )

    events: List[str] = []

    def fake_run_task(*, ledger: Ledger, project: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        events.append(f"task:{task['type']}")
        return {"task": {**task, "status": "success"}}

    def fake_create_job(*, ledger: Ledger, project_id: str, provider: str, kind: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": "job1", "provider": provider, "kind": kind, "spec": spec, "status": "created"}

    def fake_run_job(*, ledger: Ledger, job_id: str) -> Dict[str, Any]:
        events.append("job:code_review")
        return {
            "id": job_id,
            "provider": "claude_code_cli",
            "kind": "code_review",
            "status": "succeeded",
            "result": {"review_result": {"recommendation": "accept"}, "ingested": {"stored_path": "reviews/code/RESP-pre_exec.json"}},
        }

    monkeypatch.setattr("resorch.autopilot.generate_plan_openai", fake_generate_plan_openai)
    monkeypatch.setattr("resorch.autopilot.run_task", fake_run_task)
    monkeypatch.setattr("resorch.autopilot.create_job", fake_create_job)
    monkeypatch.setattr("resorch.autopilot.run_job", fake_run_job)
    monkeypatch.setattr("resorch.autopilot._list_git_changed_paths", lambda _ws: ["src/foo.py"])
    monkeypatch.setattr(
        "resorch.autopilot.load_review_policy",
        lambda _root, **kw: {"review_phases": {"code_review_gate": {"enabled": True, "provider": "claude_code_cli", "kind": "code_review"}}},
    )

    out = run_autopilot_iteration(
        ledger=ledger,
        project_id=project["id"],
        objective="obj",
        model="gpt-5.2-pro",
        iteration=0,
        dry_run=False,
        max_actions=2,
        background=False,
        config={"max_fix_tasks_per_review": 10, "planner_provider": "openai"},
    )

    assert events == ["task:codex_exec", "job:code_review", "task:shell_exec"]
    assert out["pre_exec_review"]["enabled"] is True
    assert out["pre_exec_review"]["jobs"] and out["pre_exec_review"]["jobs"][0]["recommendation"] == "accept"


def test_pre_exec_review_old_key_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Workspace configs still using old 'pre_exec' key should work via fallback."""
    ledger, project = _make_tmp_repo(tmp_path)

    def fake_generate_plan_openai(**_kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return (
            {
                "plan_id": "plan1",
                "project_id": project["id"],
                "iteration": 0,
                "objective": "test",
                "self_confidence": 0.5,
                "evidence_strength": 0.5,
                "actions": [
                    {"title": "code", "task_type": "codex_exec", "spec": {"prompt": "x"}},
                    {"title": "run", "task_type": "shell_exec", "spec": {"command": "echo ok"}},
                ],
                "should_stop": False,
            },
            {},
        )

    events: List[str] = []

    def fake_run_task(*, ledger: Ledger, project: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        events.append(f"task:{task['type']}")
        return {"task": {**task, "status": "success"}}

    def fake_create_job(*, ledger: Ledger, project_id: str, provider: str, kind: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": "job1", "provider": provider, "kind": kind, "spec": spec, "status": "created"}

    def fake_run_job(*, ledger: Ledger, job_id: str) -> Dict[str, Any]:
        events.append("job:code_review")
        return {
            "id": job_id,
            "provider": "claude_code_cli",
            "kind": "code_review",
            "status": "succeeded",
            "result": {"review_result": {"recommendation": "accept"}, "ingested": {"stored_path": "reviews/code/RESP.json"}},
        }

    monkeypatch.setattr("resorch.autopilot.generate_plan_openai", fake_generate_plan_openai)
    monkeypatch.setattr("resorch.autopilot.run_task", fake_run_task)
    monkeypatch.setattr("resorch.autopilot.create_job", fake_create_job)
    monkeypatch.setattr("resorch.autopilot.run_job", fake_run_job)
    monkeypatch.setattr("resorch.autopilot._list_git_changed_paths", lambda _ws: ["src/foo.py"])
    # Use OLD key name "pre_exec" — should still work via fallback.
    monkeypatch.setattr(
        "resorch.autopilot.load_review_policy",
        lambda _root, **kw: {"review_phases": {"pre_exec": {"enabled": True, "provider": "claude_code_cli", "kind": "code_review"}}},
    )

    out = run_autopilot_iteration(
        ledger=ledger,
        project_id=project["id"],
        objective="obj",
        model="gpt-5.2-pro",
        iteration=0,
        dry_run=False,
        max_actions=2,
        background=False,
        config={"max_fix_tasks_per_review": 10, "planner_provider": "openai"},
    )

    assert events == ["task:codex_exec", "job:code_review", "task:shell_exec"]
    assert out["pre_exec_review"]["enabled"] is True


def test_pre_exec_review_reject_runs_fix_then_retries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger, project = _make_tmp_repo(tmp_path)

    def fake_generate_plan_openai(**_kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return (
            {
                "plan_id": "plan1",
                "project_id": project["id"],
                "iteration": 0,
                "objective": "test",
                "self_confidence": 0.5,
                "evidence_strength": 0.5,
                "actions": [
                    {"title": "code", "task_type": "codex_exec", "spec": {"prompt": "x"}},
                    {"title": "run", "task_type": "shell_exec", "spec": {"command": "echo ok"}},
                ],
                "should_stop": False,
            },
            {},
        )

    events: List[str] = []
    calls = {"review": 0}

    def fake_run_task(*, ledger: Ledger, project: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        events.append(f"task:{task['type']}")
        return {"task": {**task, "status": "success"}}

    def fake_create_job(*, ledger: Ledger, project_id: str, provider: str, kind: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": f"job{calls['review']+1}", "provider": provider, "kind": kind, "spec": spec, "status": "created"}

    def fake_run_job(*, ledger: Ledger, job_id: str) -> Dict[str, Any]:
        calls["review"] += 1
        events.append("job:code_review")
        if calls["review"] == 1:
            # Create a review_fix task that autopilot should run.
            create_task(
                ledger=ledger,
                project_id=project["id"],
                task_type="review_fix",
                spec={"stage": "pre_exec", "severity": "blocker", "category": "paths", "message": "fix it", "target_paths": ["src/foo.py"]},
            )
            rec = "reject"
        else:
            rec = "accept"
        return {
            "id": job_id,
            "provider": "claude_code_cli",
            "kind": "code_review",
            "status": "succeeded",
            "result": {"review_result": {"recommendation": rec}, "ingested": {"stored_path": "reviews/code/RESP-pre_exec.json"}},
        }

    monkeypatch.setattr("resorch.autopilot.generate_plan_openai", fake_generate_plan_openai)
    monkeypatch.setattr("resorch.autopilot.run_task", fake_run_task)
    monkeypatch.setattr("resorch.autopilot.create_job", fake_create_job)
    monkeypatch.setattr("resorch.autopilot.run_job", fake_run_job)
    monkeypatch.setattr("resorch.autopilot._list_git_changed_paths", lambda _ws: ["src/foo.py"])
    monkeypatch.setattr(
        "resorch.autopilot.load_review_policy",
        lambda _root, **kw: {"review_phases": {"code_review_gate": {"enabled": True, "provider": "claude_code_cli", "kind": "code_review", "max_fix_retries": 1}}},
    )

    out = run_autopilot_iteration(
        ledger=ledger,
        project_id=project["id"],
        objective="obj",
        model="gpt-5.2-pro",
        iteration=0,
        dry_run=False,
        max_actions=2,
        background=False,
        config={"max_fix_tasks_per_review": 10, "planner_provider": "openai"},
    )

    assert events == ["task:codex_exec", "job:code_review", "task:review_fix", "job:code_review", "task:shell_exec"]
    assert len(out["pre_exec_review"]["jobs"]) == 2
    assert out["pre_exec_review"]["fix_tasks_ran"]
    assert out["pre_exec_review"]["final_recommendation"] == "accept"


def test_post_exec_hard_dual_review_creates_two_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger, project = _make_tmp_repo(tmp_path)

    # Enable dual_on_hard with different providers so we can assert 2 jobs.
    (Path(ledger.paths.root) / "configs" / "review_policy.yaml").write_text(
        "\n".join(
            [
                "reviewers:",
                "  primary:",
                "    provider: openai",
                "    model: gpt-5.2-pro",
                "  escalation:",
                "    provider: claude_code_cli",
                "    model: opus",
                "review_phases:",
                "  post_exec:",
                "    dual_on_hard: true",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_autopilot(**_kwargs: Any) -> Dict[str, Any]:
        return {
            "plan": {
                "plan_id": "plan1",
                "project_id": project["id"],
                "iteration": 0,
                "objective": "x",
                "self_confidence": 0.5,
                "evidence_strength": 0.5,
                "actions": [],
                "should_stop": True,
                "stop_reason": "done",
            },
            "plan_artifact_path": "notes/autopilot/plan.json",
            "planner_meta": {},
            "started_at": "2026-02-01T00:00:00Z",
            "project_stage": {"before": "analysis", "after": "analysis", "changed": False, "requested": None},
            "tasks_created": [],
            "tasks_ran": [],
            "git_change_summary": {"is_git": False, "changed_lines": 0, "changed_files": 0, "changed_paths": []},
            "failure_streaks": {"any": 0, "same_task": 0},
            "stalled_jobs": [],
            "ready_stage_transitions": [],
            "review_recommendation": {"level": "hard", "reasons": ["test"], "targets": ["notes/problem.md"]},
        }

    created_jobs: List[Dict[str, Any]] = []

    def fake_create_job(*, ledger: Ledger, project_id: str, provider: str, kind: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        created_jobs.append({"provider": provider, "kind": kind, "spec": spec})
        return {"id": f"job{len(created_jobs)}", "provider": provider, "kind": kind, "spec": spec}

    def fake_run_job(*, ledger: Ledger, job_id: str) -> Dict[str, Any]:
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
    assert len(created_jobs) == 2
    assert {created_jobs[0]["provider"], created_jobs[1]["provider"]} == {"openai", "claude_code_cli"}
    step = out["steps"][0]
    assert step["review_job"] is not None
    assert step["escalation_review_job"] is not None
