"""Tests for PDCA cycle fixes (Issues 1–5 from pdca_findings_v2.md)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from resorch.agent_loop import run_agent_loop
from resorch.autopilot import recommend_review_from_policy, _ensure_git_baseline
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project, get_project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def _write_scoreboard(workspace: Path, mean: float = 0.5) -> None:
    results = workspace / "results"
    results.mkdir(parents=True, exist_ok=True)
    sb = {"primary_metric": {"current": {"mean": mean}}}
    (results / "scoreboard.json").write_text(
        json.dumps(sb, indent=2) + "\n", encoding="utf-8",
    )


def _base_review_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Minimal kwargs for recommend_review_from_policy with all defaults."""
    defaults: Dict[str, Any] = {
        "policy": {},
        "plan_self_confidence": 0.5,
        "plan_evidence_strength": 0.5,
        "git_changed_lines": 10,
        "git_changed_files": 1,
        "git_changed_paths": [],
        "failure_streak_any": 0,
        "failure_streak_same_task": 0,
        "ready_stage_transitions": [],
        "claim_created": False,
        "external_fetch_detected": False,
        "stalled_jobs": [],
        "default_targets": ["notes/problem.md"],
        "plan_suggested": None,
        "stage_transition_requested": False,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Issue 1: Interpretation Challenger trigger — tested at integration level
#   (The challenger module tests already cover the function itself.
#    Here we verify the trigger condition in autopilot.)
# ---------------------------------------------------------------------------


def test_challenger_trigger_fires_on_codex_exec_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Issue 1: Challenger should fire when codex_exec tasks succeed, not only shell_exec."""
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=True,
    )
    workspace = Path(project["repo_path"])
    _write_scoreboard(workspace)

    # Write a minimal scoreboard so challenger can find it.
    (workspace / "results" / "scoreboard.json").write_text(
        json.dumps({"primary_metric": {"current": {"mean": 0.5}}}) + "\n",
        encoding="utf-8",
    )

    challenger_called = {"count": 0}

    def fake_maybe_challenge(**_kwargs: Any) -> Any:
        from resorch.interpretation_challenger import ChallengerResult, ChallengerCheck
        challenger_called["count"] += 1
        return ChallengerResult(
            overall_concern_level="low",
            flags=[],
            checks=[ChallengerCheck(item="test", status="ok", reason="ok")],
        )

    def fake_generate_plan_openai(**_kwargs: Any):
        return (
            {
                "plan_id": "plan1", "project_id": project["id"], "iteration": 0,
                "objective": "test", "self_confidence": 0.5, "evidence_strength": 0.5,
                "actions": [
                    {"title": "compute", "task_type": "codex_exec", "spec": {"prompt": "x"}},
                ],
                "should_stop": True, "stop_reason": "done",
            },
            {},
        )

    def fake_run_task(*, ledger: Ledger, project: Dict, task: Dict) -> Dict:
        return {"task": {**task, "status": "success"}}

    monkeypatch.setattr("resorch.autopilot.generate_plan_openai", fake_generate_plan_openai)
    monkeypatch.setattr("resorch.autopilot.run_task", fake_run_task)
    monkeypatch.setattr("resorch.autopilot._list_git_changed_paths", lambda _ws: [])
    monkeypatch.setattr(
        "resorch.autopilot.load_review_policy",
        lambda _root, **kw: {"interpretation_challenger": {"enabled": True, "provider": "claude_code_cli", "model": "sonnet"}},
    )
    monkeypatch.setattr(
        "resorch.interpretation_challenger.maybe_challenge_interpretation_from_workspace",
        fake_maybe_challenge,
    )

    from resorch.autopilot import run_autopilot_iteration
    out = run_autopilot_iteration(
        ledger=ledger, project_id="p1", objective="test", model="gpt-5.2-pro",
        iteration=0, dry_run=False, max_actions=1, background=False,
        config={"planner_provider": "openai"},
    )

    # The key assertion: challenger should have been called (not skipped).
    assert challenger_called["count"] == 1
    ic = out.get("interpretation_challenger") or {}
    assert ic.get("skipped_reason") is None
    assert ic.get("overall_concern_level") == "low"


# ---------------------------------------------------------------------------
# Issue 2: Escalation with same provider
# ---------------------------------------------------------------------------


def test_escalation_fires_with_same_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Issue 2: Escalation should fire even when primary and escalation use same provider."""
    ledger = _make_tmp_repo(tmp_path)
    (ledger.paths.root / "configs").mkdir(parents=True, exist_ok=True)
    (ledger.paths.root / "configs" / "review_policy.yaml").write_text(
        "\n".join([
            "reviewers:",
            "  primary:",
            "    provider: openai",
            "    model: gpt-5.2-pro",
            "  escalation:",
            "    provider: openai",  # Same provider!
            "    model: gpt-5.2-pro",
            "review_phases:",
            "  post_exec:",
            "    dual_on_hard: true",
            "",
        ]),
        encoding="utf-8",
    )

    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)
    _write_scoreboard(Path(project["repo_path"]))

    def fake_autopilot(**kwargs: Any) -> Dict[str, Any]:
        return {
            "plan": {
                "plan_id": "plan1", "project_id": project["id"], "iteration": 0,
                "objective": "", "self_confidence": 0.5, "evidence_strength": 0.5,
                "actions": [], "should_stop": True, "stop_reason": "done",
            },
            "plan_artifact_path": "notes/autopilot/plan.json",
            "planner_meta": {},
            "started_at": "2026-02-01T00:00:00Z",
            "project_stage": {"before": "analysis", "after": "analysis", "changed": False, "requested": None},
            "tasks_created": [], "tasks_ran": [],
            "git_change_summary": {"is_git": False, "changed_lines": 0, "changed_files": 0, "changed_paths": []},
            "failure_streaks": {"any": 0, "same_task": 0},
            "stalled_jobs": [], "ready_stage_transitions": [],
            "review_recommendation": {"level": "hard", "reasons": ["test"], "targets": ["notes/problem.md"]},
        }

    created_jobs: List[Dict[str, Any]] = []

    def fake_create_job(*, ledger: Any, project_id: str, provider: str, kind: str, spec: Dict) -> Dict:
        created_jobs.append({"provider": provider, "kind": kind})
        return {"id": f"job{len(created_jobs)}", "provider": provider, "kind": kind, "spec": spec}

    def fake_run_job(*, ledger: Any, job_id: str) -> Dict:
        return {"id": job_id, "status": "succeeded"}

    monkeypatch.setattr("resorch.agent_loop.run_autopilot_iteration", fake_autopilot)
    monkeypatch.setattr("resorch.agent_loop.create_job", fake_create_job)
    monkeypatch.setattr("resorch.agent_loop.run_job", fake_run_job)

    out = run_agent_loop(ledger=ledger, project_id="p1", objective="test", max_steps=1, dry_run=False)

    # Both primary and escalation review jobs should be created (same provider).
    assert len(created_jobs) == 2
    step = out["steps"][0]
    assert step.get("review_job") is not None
    assert step.get("escalation_review_job") is not None


# ---------------------------------------------------------------------------
# Issue 3b: stage_transition_requested parameter
# ---------------------------------------------------------------------------


def test_stage_transition_not_hard_when_not_requested() -> None:
    """Issue 3b: on_stage_transition should NOT trigger hard review when
    transitions are ready but Planner didn't request a stage change."""
    rec = recommend_review_from_policy(**_base_review_kwargs(
        policy={"hard_gates": {"on_stage_transition": True}},
        ready_stage_transitions=[{"name": "intake_to_analysis", "decision": "auto_pass"}],
        stage_transition_requested=False,
    ))
    assert rec.level != "hard"


def test_stage_transition_hard_when_requested() -> None:
    """Issue 3b: on_stage_transition SHOULD trigger hard when Planner requested."""
    rec = recommend_review_from_policy(**_base_review_kwargs(
        policy={"hard_gates": {"on_stage_transition": True}},
        ready_stage_transitions=[{"name": "intake_to_analysis", "decision": "auto_pass"}],
        stage_transition_requested=True,
    ))
    assert rec.level == "hard"
    assert any("stage_transition" in r for r in rec.reasons)


# ---------------------------------------------------------------------------
# Issue 3a: _ensure_git_baseline
# ---------------------------------------------------------------------------


def test_ensure_git_baseline_commits_untracked(tmp_path: Path) -> None:
    """Issue 3a: After baseline commit, untracked files become tracked."""
    import subprocess
    workspace = tmp_path / "ws"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=str(workspace), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(workspace), check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(workspace), check=True, capture_output=True,
    )

    # Create some files.
    (workspace / "results").mkdir()
    (workspace / "results" / "scoreboard.json").write_text("{}", encoding="utf-8")
    (workspace / "manuscript.md").write_text("# Draft", encoding="utf-8")

    # Before baseline: untracked files exist.
    proc = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=str(workspace), capture_output=True, text=True,
    )
    assert "manuscript.md" in proc.stdout

    # Run baseline commit.
    _ensure_git_baseline(workspace, iteration=0)

    # After baseline: no untracked files.
    proc2 = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=str(workspace), capture_output=True, text=True,
    )
    assert proc2.stdout.strip() == ""

    # git log shows the baseline commit.
    proc3 = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=str(workspace), capture_output=True, text=True,
    )
    assert "baseline-iter-000" in proc3.stdout


# ---------------------------------------------------------------------------
# Issue 5: Watchdog exempt stages
# ---------------------------------------------------------------------------


def test_unchanged_metric_watchdog_exempt_writing_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Issue 5: When stage is 'writing', unchanged metric watchdog should NOT stop."""
    ledger = _make_tmp_repo(tmp_path)
    repo_root = ledger.paths.root
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "pivot_policy.yaml").write_text(
        "\n".join([
            "metric_watchdog:",
            "  unchanged_metric_force_stop_after: 1",
            "",
        ]),
        encoding="utf-8",
    )

    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="writing", git_init=False)
    workspace = Path(project["repo_path"])
    _write_scoreboard(workspace, mean=0.175)

    iteration_count = {"n": 0}

    def fake_autopilot(**kwargs: Any) -> Dict[str, Any]:
        iteration_count["n"] += 1
        step = int(kwargs.get("iteration", 0))
        return {
            "plan": {
                "plan_id": f"plan{step}", "project_id": project["id"], "iteration": step,
                "objective": "", "self_confidence": 0.5, "evidence_strength": 0.5,
                "actions": [], "should_stop": step >= 2, "stop_reason": "done" if step >= 2 else None,
            },
            "plan_artifact_path": "notes/autopilot/plan.json",
            "started_at": "2026-02-01T00:00:00Z",
            "project_stage": {"before": "writing", "after": "writing", "changed": False, "requested": None},
            "tasks_created": [], "tasks_ran": [],
            "git_change_summary": {"is_git": False, "changed_lines": 10, "changed_files": 1, "changed_paths": []},
            "failure_streaks": {"any": 0, "same_task": 0},
            "stalled_jobs": [], "ready_stage_transitions": [],
            "review_recommendation": {"level": "none", "reasons": [], "targets": []},
        }

    monkeypatch.setattr("resorch.agent_loop.run_autopilot_iteration", fake_autopilot)

    out = run_agent_loop(ledger=ledger, project_id="p1", objective="test", max_steps=3, dry_run=False)

    # With unchanged_metric_force_stop_after=1, the old code would stop after step 2
    # (streak=1 after step 1). With the exempt_stages fix, the 'writing' stage should
    # be exempt and the loop should continue until should_stop=True at step 3.
    assert out.get("stopped_reason") != "unchanged_metric_force_stop"
    assert iteration_count["n"] == 3


def test_unchanged_metric_watchdog_stops_non_exempt_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Issue 5 control test: non-exempt stage SHOULD still be stopped by watchdog."""
    ledger = _make_tmp_repo(tmp_path)
    repo_root = ledger.paths.root
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "pivot_policy.yaml").write_text(
        "\n".join([
            "metric_watchdog:",
            "  unchanged_metric_force_stop_after: 1",
            "",
        ]),
        encoding="utf-8",
    )

    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)
    workspace = Path(project["repo_path"])
    _write_scoreboard(workspace, mean=0.175)

    iteration_count = {"n": 0}

    def fake_autopilot(**kwargs: Any) -> Dict[str, Any]:
        iteration_count["n"] += 1
        step = int(kwargs.get("iteration", 0))
        return {
            "plan": {
                "plan_id": f"plan{step}", "project_id": project["id"], "iteration": step,
                "objective": "", "self_confidence": 0.5, "evidence_strength": 0.5,
                "actions": [], "should_stop": False,
            },
            "plan_artifact_path": "notes/autopilot/plan.json",
            "started_at": "2026-02-01T00:00:00Z",
            "project_stage": {"before": "analysis", "after": "analysis", "changed": False, "requested": None},
            "tasks_created": [], "tasks_ran": [],
            "git_change_summary": {"is_git": False, "changed_lines": 10, "changed_files": 1, "changed_paths": []},
            "failure_streaks": {"any": 0, "same_task": 0},
            "stalled_jobs": [], "ready_stage_transitions": [],
            "review_recommendation": {"level": "none", "reasons": [], "targets": []},
        }

    monkeypatch.setattr("resorch.agent_loop.run_autopilot_iteration", fake_autopilot)

    out = run_agent_loop(ledger=ledger, project_id="p1", objective="test", max_steps=5, dry_run=False)

    # 'analysis' is NOT exempt → watchdog should stop after streak reaches threshold.
    assert "unchanged_metric" in (out.get("stopped_reason") or "")
    assert iteration_count["n"] == 2  # step 0 sets baseline, step 1 unchanged → stop before step 2
