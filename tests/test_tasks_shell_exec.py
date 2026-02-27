from __future__ import annotations

from pathlib import Path

from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.tasks import create_task, run_task


def test_shell_exec_task_run(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")

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

    task = create_task(
        ledger=ledger,
        project_id=project["id"],
        task_type="shell_exec",
        spec={"command": ["python", "-c", "print('hi')"]},
    )
    result = run_task(ledger=ledger, project=project, task=task)

    assert result["task"]["status"] == "success"
    assert result["run"]["exit_code"] == 0
    stdout_path = Path(result["run"]["meta"]["stdout_path"])
    assert stdout_path.exists()
    assert "hi" in stdout_path.read_text(encoding="utf-8")


def test_shell_exec_task_timeout(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")

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

    task = create_task(
        ledger=ledger,
        project_id=project["id"],
        task_type="shell_exec",
        spec={
            "command": ["python", "-c", "import time; time.sleep(10)"],
            "timeout_sec": 1,
        },
    )
    result = run_task(ledger=ledger, project=project, task=task)

    assert result["task"]["status"] == "failed"
    assert result["run"]["meta"].get("timed_out") is True
    assert result["run"]["meta"]["timeout_sec"] == 1
    assert result["run"]["exit_code"] == -1

