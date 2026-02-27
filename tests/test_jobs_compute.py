from __future__ import annotations

import time
from pathlib import Path

import pytest

from resorch.jobs import create_job, poll_job, run_job
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def _write_constraints(workspace: Path, *, backend: str, max_parallel: int = 1) -> None:
    if backend == "local":
        txt = "\n".join(
            [
                "compute:",
                "  backend: local",
                "  local:",
                f"    max_parallel: {int(max_parallel)}",
                "",
            ]
        )
    else:
        txt = "\n".join(
            [
                "compute:",
                "  backend: slurm",
                "",
            ]
        )
    (workspace / "constraints.yaml").write_text(txt, encoding="utf-8")


def _poll_until_terminal(ledger: Ledger, job_id: str, *, timeout_sec: float = 5.0) -> dict:
    deadline = time.time() + timeout_sec
    last = None
    while time.time() < deadline:
        last = poll_job(ledger=ledger, job_id=job_id)
        if last["status"] in {"completed_external", "failed_external"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job did not reach terminal state in {timeout_sec}s (last={last})")


def test_compute_local_submit_and_poll_creates_ingest_task(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="intake", git_init=False)
    ws = Path(project["repo_path"]).resolve()
    _write_constraints(ws, backend="local", max_parallel=2)

    job = create_job(
        ledger=ledger,
        project_id=project["id"],
        provider="compute",
        kind="compute",
        spec={"cd": ".", "command": "python -c \"print(123)\""},
    )
    job = run_job(ledger=ledger, job_id=job["id"])
    assert job["status"] in {"running_external", "submitted_external"}

    job = _poll_until_terminal(ledger, job["id"])
    assert job["status"] == "completed_external"
    assert isinstance(job.get("result"), dict)
    assert job["result"].get("ingest_task_id")

    job_dir = ws / "jobs" / job["id"]
    assert (job_dir / "job.json").exists()
    assert (job_dir / "run_local.sh").exists()


def test_compute_local_max_parallel_queues_second_job(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="intake", git_init=False)
    ws = Path(project["repo_path"]).resolve()
    _write_constraints(ws, backend="local", max_parallel=1)

    job1 = create_job(
        ledger=ledger,
        project_id=project["id"],
        provider="compute",
        kind="compute",
        spec={"cd": ".", "command": "python -c \"import time; time.sleep(0.4); print('job1')\""},
    )
    job2 = create_job(
        ledger=ledger,
        project_id=project["id"],
        provider="compute",
        kind="compute",
        spec={"cd": ".", "command": "python -c \"print('job2')\""},
    )

    job1 = run_job(ledger=ledger, job_id=job1["id"])
    assert job1["status"] in {"running_external", "submitted_external"}

    job2 = run_job(ledger=ledger, job_id=job2["id"])
    assert job2["status"] == "submitted_external"
    assert job2.get("remote_id") in {None, ""}

    job1 = _poll_until_terminal(ledger, job1["id"], timeout_sec=5.0)
    assert job1["status"] == "completed_external"

    job2 = _poll_until_terminal(ledger, job2["id"], timeout_sec=5.0)
    assert job2["status"] == "completed_external"


def test_compute_slurm_submit_and_poll_with_mocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="intake", git_init=False)
    ws = Path(project["repo_path"]).resolve()
    _write_constraints(ws, backend="slurm")

    def fake_run(cmd, cwd=None, check=False, text=False, capture_output=False):  # noqa: ANN001
        exe = cmd[0]
        if exe == "sbatch":
            return subprocess.CompletedProcess(cmd, 0, stdout="12345\n", stderr="")
        if exe == "sacct":
            return subprocess.CompletedProcess(cmd, 0, stdout="12345|COMPLETED|0:0\n", stderr="")
        if exe == "squeue":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected")

    import subprocess  # noqa: WPS433

    monkeypatch.setattr("resorch.jobs.subprocess.run", fake_run)

    job = create_job(
        ledger=ledger,
        project_id=project["id"],
        provider="compute",
        kind="compute",
        spec={"cd": ".", "command": "echo hi"},
    )
    job = run_job(ledger=ledger, job_id=job["id"])
    assert job["status"] == "submitted_external"
    assert job.get("remote_id") == "12345"

    job = poll_job(ledger=ledger, job_id=job["id"])
    assert job["status"] == "completed_external"
    assert isinstance(job.get("result"), dict)
    assert job["result"].get("ingest_task_id")

