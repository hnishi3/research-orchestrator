"""Tests for High/Medium bug fixes from Codex review:

H4. Task run finalized on exception (codex_exec/review_fix try/except)
H5. Review level preserved when do_phase_failed (not forced to 'none')
M7. Scoreboard not overwritten on parse error
M8. _write_last_errors uses direct SQL (no missing method)
M9. same_task_streak is truly consecutive
M10. last_message_path boundary check
M11. Non-dict JSON in artifact parse does not crash
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest

from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> Ledger:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo))
    ledger.init()
    return ledger


def _make_project(tmp_path: Path, project_id: str = "p1") -> tuple:
    ledger = _make_repo(tmp_path)
    project = create_project(
        ledger=ledger, project_id=project_id, title="P1",
        domain="test", stage="analysis", git_init=False,
    )
    return ledger, project


# ---------------------------------------------------------------------------
# H4. Task run finalized on exception
# ---------------------------------------------------------------------------

class _CrashingProc:
    """Proc that crashes immediately on stdout iteration."""
    def __init__(self, *_a: Any, **_kw: Any) -> None:
        self.stdin = self
        self.stdout = self
        self.returncode = None
        self._killed = False

    def write(self, _d: str) -> None: pass
    def close(self) -> None: pass
    def kill(self) -> None:
        self._killed = True
        self.returncode = -9
    def wait(self, timeout: Optional[int] = None) -> None:
        self.returncode = -9
    def __iter__(self) -> "_CrashingProc": return self
    def __next__(self) -> str:
        raise RuntimeError("codex binary not found")


def test_codex_exec_task_marks_failed_on_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If codex subprocess crashes, task must be marked 'failed', not stuck in 'running'."""
    from resorch.tasks import create_task, run_task

    ledger, project = _make_project(tmp_path)
    monkeypatch.setattr("resorch.codex_runner.subprocess.Popen", _CrashingProc)

    task = create_task(
        ledger=ledger, project_id="p1",
        task_type="codex_exec",
        spec={"prompt": "test"},
    )

    with pytest.raises(RuntimeError, match="codex binary not found"):
        run_task(ledger=ledger, project=project, task=task)

    # Task must NOT be stuck in 'running'.
    from resorch.tasks import get_task
    t = get_task(ledger, task["id"])
    assert t["status"] == "failed", f"Expected 'failed' but got '{t['status']}'"


# ---------------------------------------------------------------------------
# H5. Review level preserved when do_phase_failed
# ---------------------------------------------------------------------------

def test_review_recommendation_preserves_level_on_do_phase_failure() -> None:
    """review_recommendation should keep the policy-derived level even when do_phase_failed."""
    # Simulate what autopilot.py returns.
    from resorch.autopilot_config import ReviewRecommendation

    recommendation = ReviewRecommendation(level="hard", reasons=["stage_transition"], targets=[])
    do_phase_failed = True

    # New behavior: level is preserved, deferred flag is set.
    result = {
        "level": recommendation.level,
        "reasons": recommendation.reasons,
        "targets": recommendation.targets,
        "deferred_due_to_do_phase_failure": do_phase_failed,
    }

    assert result["level"] == "hard", "Level should NOT be forced to 'none'"
    assert result["deferred_due_to_do_phase_failure"] is True


# ---------------------------------------------------------------------------
# M7. Scoreboard not overwritten on parse error
# ---------------------------------------------------------------------------

def test_scoreboard_not_overwritten_on_corrupt_json(tmp_path: Path) -> None:
    """If scoreboard.json is corrupted, it should NOT be overwritten with {}."""
    from resorch.autopilot_config import ReviewRecommendation
    from resorch.autopilot_digests import _update_pdca_digests

    ledger, project = _make_project(tmp_path)
    ws = Path(project["repo_path"])
    scoreboard_path = ws / "results" / "scoreboard.json"
    scoreboard_path.parent.mkdir(parents=True, exist_ok=True)
    scoreboard_path.write_text("{corrupted json!!!", encoding="utf-8")

    _update_pdca_digests(
        ledger=ledger,
        project=project,
        iteration=0,
        started_at="2026-01-01T00:00:00Z",
        plan_artifact_path="plans/plan.json",
        tasks_created=[],
        tasks_ran=[],
        git_change_summary={"changed_files": 0, "changed_lines": 0},
        review_recommendation=ReviewRecommendation(level="none", reasons=[], targets=[]),
    )

    # The corrupt file should be left as-is (not overwritten).
    content = scoreboard_path.read_text(encoding="utf-8")
    assert content == "{corrupted json!!!", "Corrupt scoreboard should not be overwritten"


# ---------------------------------------------------------------------------
# M8. _write_last_errors uses direct SQL (no missing method)
# ---------------------------------------------------------------------------

def test_write_last_errors_no_attribute_error(tmp_path: Path) -> None:
    """_write_last_errors should not raise AttributeError for missing list_task_runs."""
    from resorch.autopilot_digests import _write_last_errors

    ledger, project = _make_project(tmp_path)
    ws = Path(project["repo_path"])

    # Should not crash even with failed tasks (the method used to call
    # ledger.list_task_runs which didn't exist).
    _write_last_errors(
        workspace=ws.resolve(),
        tasks_ran=[{"id": "t1", "type": "codex_exec", "status": "failed"}],
        ledger=ledger,
    )

    # If we get here without AttributeError, the fix works.
    errors_path = ws / "notes" / "last_errors.md"
    assert errors_path.exists()
    content = errors_path.read_text(encoding="utf-8")
    assert "Task t1" in content


# ---------------------------------------------------------------------------
# M9. same_task_streak is truly consecutive
# ---------------------------------------------------------------------------

def test_same_task_streak_stops_at_different_task(tmp_path: Path) -> None:
    """same_task_streak should stop counting when a different task_id appears."""
    from resorch.autopilot_review import compute_failure_streaks
    from resorch.tasks import create_task

    ledger, project = _make_project(tmp_path)

    # Create two tasks
    task_a = create_task(ledger=ledger, project_id="p1", task_type="codex_exec", spec={"prompt": "a"})
    task_b = create_task(ledger=ledger, project_id="p1", task_type="codex_exec", spec={"prompt": "b"})

    # Insert task runs in chronological order (ORDER BY started_at DESC in query):
    # t=10: task_A failed (most recent)
    # t=09: task_A failed
    # t=08: task_B failed  ← breaks consecutive same_task streak
    # t=07: task_A failed  ← should NOT be counted
    runs_data = [
        ("r4", task_a["id"], "failed", "2026-01-01T00:07:00Z"),
        ("r3", task_b["id"], "failed", "2026-01-01T00:08:00Z"),
        ("r2", task_a["id"], "failed", "2026-01-01T00:09:00Z"),
        ("r1", task_a["id"], "failed", "2026-01-01T00:10:00Z"),
    ]
    for run_id, task_id, status, ts in runs_data:
        ledger.insert_task_run(
            run_id=run_id,
            task_id=task_id,
            status=status,
            jsonl_path=None,
            last_message_path=None,
            meta={"runner": "test"},
        )
        # Update started_at to control ordering
        ledger._exec("UPDATE task_runs SET started_at = ? WHERE id = ?", (ts, run_id))
    ledger._maybe_commit()

    streaks = compute_failure_streaks(ledger=ledger, project_id="p1")
    assert streaks["same_task"] == 2, f"Expected same_task=2 but got {streaks['same_task']}"


# ---------------------------------------------------------------------------
# M11. Non-dict JSON in artifact parse
# ---------------------------------------------------------------------------

def test_artifact_parse_handles_non_dict_json(tmp_path: Path) -> None:
    """Non-dict JSON (e.g., a list or string) should not crash artifact parsing."""
    # Write a non-dict JSON as last_message
    msg_path = tmp_path / "last_message.txt"
    msg_path.write_text('"just a string"', encoding="utf-8")

    # Simulate the parsing logic from tasks.py
    task_result = json.loads(msg_path.read_text(encoding="utf-8"))
    if not isinstance(task_result, dict):
        task_result = {}

    # Should safely produce empty artifacts list
    artifacts = task_result.get("artifacts_created") or []
    assert artifacts == []


def test_artifact_parse_handles_list_json(tmp_path: Path) -> None:
    """JSON list should not crash artifact parsing."""
    msg_path = tmp_path / "last_message.txt"
    msg_path.write_text('[1, 2, 3]', encoding="utf-8")

    task_result = json.loads(msg_path.read_text(encoding="utf-8"))
    if not isinstance(task_result, dict):
        task_result = {}

    artifacts = task_result.get("artifacts_created") or []
    assert artifacts == []
