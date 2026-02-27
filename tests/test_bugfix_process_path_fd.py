"""Tests for batch-3 fixes:

A. codex_runner: try/finally guarantees proc cleanup on exception
B. codex_runner: proc.wait() has timeout (no infinite block)
C. shell_exec: cd parameter validated to stay within workspace
D. artifacts: mkstemp fd closed immediately, no leak
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest

from resorch.artifacts import put_artifact
from resorch.codex_runner import run_codex_exec_jsonl
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


# ---------------------------------------------------------------------------
# A. codex_runner: proc cleanup on exception (try/finally)
# ---------------------------------------------------------------------------

class _FakeProc:
    """Simulates a Popen whose stdout iteration raises."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.stdin = self  # self-referencing for .write/.close
        self.stdout = self
        self.returncode = 0
        self._waited = False
        self._killed = False
        self._stdout_closed = False
        self._iter_count = 0

    def write(self, _data: str) -> None:
        pass

    def close(self) -> None:
        self._stdout_closed = True

    def kill(self) -> None:
        self._killed = True

    def wait(self, timeout: Optional[int] = None) -> None:
        self._waited = True
        self.returncode = 0

    def __iter__(self) -> "_FakeProc":
        return self

    def __next__(self) -> str:
        self._iter_count += 1
        if self._iter_count == 1:
            return '{"type": "message"}\n'
        # Simulate crash during stdout iteration
        raise IOError("simulated read error")


def test_codex_runner_cleans_up_on_iteration_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If an exception occurs during stdout iteration, proc.wait() must still be called."""
    fake_proc = _FakeProc()

    def fake_popen(*_args: Any, **_kwargs: Any) -> _FakeProc:
        return fake_proc

    monkeypatch.setattr("resorch.codex_runner.subprocess.Popen", fake_popen)

    jsonl = tmp_path / "out.jsonl"
    last_msg = tmp_path / "last_msg.txt"
    stderr = tmp_path / "stderr.txt"

    with pytest.raises(IOError, match="simulated read error"):
        run_codex_exec_jsonl(
            prompt="test",
            cd=tmp_path,
            sandbox="read-only",
            model=None,
            config_overrides=[],
            jsonl_path=jsonl,
            last_message_path=last_msg,
            stderr_path=stderr,
            on_event=None,
        )

    # Key assertion: proc.wait() was called despite the exception.
    assert fake_proc._waited, "proc.wait() must be called even when stdout iteration fails"
    assert fake_proc._stdout_closed, "proc.stdout.close() must be called in finally block"


# ---------------------------------------------------------------------------
# B. codex_runner: proc.wait() has timeout
# ---------------------------------------------------------------------------

class _HangingProc:
    """Simulates a Popen whose wait() hangs until killed."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.stdin = self
        self.stdout = self
        self.returncode = None
        self._killed = False

    def write(self, _data: str) -> None:
        pass

    def close(self) -> None:
        pass

    def kill(self) -> None:
        self._killed = True
        self.returncode = -9

    def wait(self, timeout: Optional[int] = None) -> None:
        if not self._killed and timeout is not None:
            raise subprocess.TimeoutExpired(cmd="codex", timeout=timeout)
        # After kill, wait succeeds.
        self.returncode = -9

    def __iter__(self) -> "_HangingProc":
        return self

    def __next__(self) -> str:
        raise StopIteration


def test_codex_runner_kills_on_wait_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If proc.wait(timeout=...) times out, the process should be killed."""
    fake_proc = _HangingProc()

    def fake_popen(*_args: Any, **_kwargs: Any) -> _HangingProc:
        return fake_proc

    monkeypatch.setattr("resorch.codex_runner.subprocess.Popen", fake_popen)

    jsonl = tmp_path / "out.jsonl"
    last_msg = tmp_path / "last_msg.txt"
    stderr = tmp_path / "stderr.txt"

    # Should NOT raise — the finally block catches TimeoutExpired, kills, and re-waits.
    result = run_codex_exec_jsonl(
        prompt="test",
        cd=tmp_path,
        sandbox="read-only",
        model=None,
        config_overrides=[],
        jsonl_path=jsonl,
        last_message_path=last_msg,
        stderr_path=stderr,
        on_event=None,
    )

    assert fake_proc._killed, "Process must be killed when wait() times out"
    assert result.returncode == -9


# ---------------------------------------------------------------------------
# C. shell_exec: cd path traversal blocked
# ---------------------------------------------------------------------------

def test_shell_exec_blocks_path_traversal(tmp_path: Path) -> None:
    """shell_exec must reject cd values that escape the workspace."""
    from resorch.tasks import create_task, run_task

    ledger = _make_repo(tmp_path)
    project = create_project(
        ledger=ledger, project_id="p1", title="P1",
        domain="test", stage="analysis", git_init=False,
    )

    task = create_task(
        ledger=ledger,
        project_id="p1",
        task_type="shell_exec",
        spec={"command": "echo hi", "cd": "../../etc"},
    )

    with pytest.raises(SystemExit, match="shell_exec cd must be within workspace"):
        run_task(ledger=ledger, project=project, task=task)


def test_shell_exec_allows_subdirectory(tmp_path: Path) -> None:
    """shell_exec should allow cd to a valid subdirectory within workspace."""
    from resorch.tasks import create_task, run_task

    ledger = _make_repo(tmp_path)
    project = create_project(
        ledger=ledger, project_id="p1", title="P1",
        domain="test", stage="analysis", git_init=False,
    )
    ws = Path(project["repo_path"])
    (ws / "subdir").mkdir(parents=True, exist_ok=True)

    task = create_task(
        ledger=ledger,
        project_id="p1",
        task_type="shell_exec",
        spec={"command": "echo ok", "cd": "subdir"},
    )

    result = run_task(ledger=ledger, project=project, task=task)
    assert result["task"]["status"] == "success"


# ---------------------------------------------------------------------------
# D. artifacts: mkstemp fd closed immediately (no leak)
# ---------------------------------------------------------------------------

def test_put_artifact_no_fd_leak_on_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If write_text fails, the mkstemp fd should already be closed (no leak)."""
    ledger = _make_repo(tmp_path)
    project = create_project(
        ledger=ledger, project_id="p1", title="P1",
        domain="test", stage="intake", git_init=False,
    )
    ws = Path(project["repo_path"])
    (ws / "data").mkdir(parents=True, exist_ok=True)

    # Monkey-patch Path.write_text to fail after fd is (supposedly) closed.
    original_write_text = Path.write_text
    call_count = {"n": 0}

    def failing_write(self: Path, *args: Any, **kwargs: Any) -> None:
        # Only fail on the temp file write (not on other write_text calls)
        if ".tmp" in str(self):
            call_count["n"] += 1
            raise OSError("simulated write failure")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write)

    with pytest.raises(OSError, match="simulated write failure"):
        put_artifact(
            ledger=ledger, project=project,
            relative_path="data/test.txt", content="hello",
            mode="overwrite", kind="test",
        )

    assert call_count["n"] >= 1, "write_text should have been called on the temp file"

    # No leftover temp files.
    tmp_files = list((ws / "data").glob("*.tmp"))
    assert tmp_files == [], f"Leftover temp files: {tmp_files}"


def test_put_artifact_atomic_overwrite_still_works(tmp_path: Path) -> None:
    """Basic smoke test: atomic overwrite should still produce correct content."""
    ledger = _make_repo(tmp_path)
    project = create_project(
        ledger=ledger, project_id="p1", title="P1",
        domain="test", stage="intake", git_init=False,
    )
    ws = Path(project["repo_path"])

    put_artifact(
        ledger=ledger, project=project,
        relative_path="file.txt", content="first",
        mode="overwrite", kind="test",
    )
    assert (ws / "file.txt").read_text(encoding="utf-8") == "first"

    put_artifact(
        ledger=ledger, project=project,
        relative_path="file.txt", content="second",
        mode="overwrite", kind="test",
    )
    assert (ws / "file.txt").read_text(encoding="utf-8") == "second"
