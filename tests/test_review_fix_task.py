from __future__ import annotations

import json
from pathlib import Path

import pytest

from resorch.codex_runner import CodexRunResult
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.tasks import create_task, run_task


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_review_fix_task_runs_codex(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )

    task = create_task(
        ledger=ledger,
        project_id=project["id"],
        task_type="review_fix",
        spec={
            "stage": "intake",
            "severity": "minor",
            "category": "writing",
            "message": "Tighten the research question.",
            "suggested_fix": "Add a clear one-sentence claim.",
            "target_paths": ["notes/problem.md"],
        },
    )

    def fake_run_codex_exec_jsonl(**kwargs):  # noqa: ANN003
        last_message_path = kwargs["last_message_path"]
        last_message_path.write_text(
            json.dumps({"status": "success", "summary": "ok", "artifacts_created": []}),
            encoding="utf-8",
        )
        return CodexRunResult(
            returncode=0,
            jsonl_path=kwargs["jsonl_path"],
            last_message_path=last_message_path,
            stderr_path=kwargs["stderr_path"],
        )

    monkeypatch.setattr("resorch.tasks.run_codex_exec_jsonl", fake_run_codex_exec_jsonl)

    out = run_task(ledger=ledger, project=project, task=task)
    assert out["task"]["status"] == "success"

    ws = Path(project["repo_path"])
    prompt_path = ws / "runs" / "review_fix" / task["id"] / "prompt.md"
    assert prompt_path.exists()

