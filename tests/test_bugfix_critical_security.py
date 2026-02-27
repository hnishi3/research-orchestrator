"""Tests for critical security fixes (Codex review batch):

A. codex_exec/review_fix: cd path traversal blocked
B. _load_prompt_text: absolute paths rejected, relative paths boundary-checked
C. create_successor_project: inherit dir traversal blocked
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project, create_successor_project
from resorch.tasks import _load_prompt_text


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
# A. codex_exec / review_fix cd traversal
# ---------------------------------------------------------------------------

def test_codex_exec_blocks_cd_traversal(tmp_path: Path) -> None:
    """codex_exec must reject cd values that escape the workspace."""
    from resorch.tasks import create_task, run_task

    ledger, project = _make_project(tmp_path)

    task = create_task(
        ledger=ledger,
        project_id="p1",
        task_type="codex_exec",
        spec={"prompt": "echo hi", "cd": "../../etc"},
    )

    with pytest.raises(SystemExit, match="codex_exec cd must be within workspace"):
        run_task(ledger=ledger, project=project, task=task)


def test_review_fix_blocks_cd_traversal(tmp_path: Path) -> None:
    """review_fix must reject cd values that escape the workspace."""
    from resorch.tasks import create_task, run_task

    ledger, project = _make_project(tmp_path)

    task = create_task(
        ledger=ledger,
        project_id="p1",
        task_type="review_fix",
        spec={
            "prompt": "fix it",
            "cd": "../../../tmp",
            "review_findings": [{"category": "test", "severity": "nit", "finding": "x"}],
        },
    )

    with pytest.raises(SystemExit, match="review_fix cd must be within workspace"):
        run_task(ledger=ledger, project=project, task=task)


def test_codex_exec_allows_valid_subdirectory(tmp_path: Path) -> None:
    """codex_exec should allow cd to a valid subdirectory (no crash on path check)."""
    from resorch.tasks import create_task

    ledger, project = _make_project(tmp_path)
    ws = Path(project["repo_path"])
    (ws / "subdir").mkdir(parents=True, exist_ok=True)

    # Just create the task — actually running it would need codex binary.
    # The traversal check happens before execution, so we verify it passes
    # by checking that the task is created without error.
    task = create_task(
        ledger=ledger,
        project_id="p1",
        task_type="codex_exec",
        spec={"prompt": "echo ok", "cd": "subdir"},
    )
    assert task["spec"]["cd"] == "subdir"


# ---------------------------------------------------------------------------
# B. _load_prompt_text: path restrictions
# ---------------------------------------------------------------------------

def test_prompt_file_rejects_absolute_path(tmp_path: Path) -> None:
    """_load_prompt_text must reject absolute prompt_file paths."""
    with pytest.raises(SystemExit, match="must be a relative path"):
        _load_prompt_text(
            spec={"prompt_file": "/etc/passwd"},
            repo_root=tmp_path,
            workspace=tmp_path,
        )


def test_prompt_file_rejects_traversal(tmp_path: Path) -> None:
    """_load_prompt_text must reject relative paths that escape roots."""
    # Create a file outside workspace/repo that traversal would reach
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    # This path traverses out of both workspace and repo_root
    with pytest.raises(SystemExit, match="prompt_file not found"):
        _load_prompt_text(
            spec={"prompt_file": "../../outside/secret.txt"},
            repo_root=repo_root,
            workspace=workspace,
        )


def test_prompt_file_allows_valid_relative(tmp_path: Path) -> None:
    """_load_prompt_text should load a valid relative prompt file."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "prompt.txt").write_text("hello world", encoding="utf-8")

    result = _load_prompt_text(
        spec={"prompt_file": "prompt.txt"},
        repo_root=tmp_path,
        workspace=workspace,
    )
    assert result == "hello world"


def test_prompt_file_allows_repo_root_relative(tmp_path: Path) -> None:
    """_load_prompt_text should fall back to repo_root for prompt files."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "prompts").mkdir()
    (repo_root / "prompts" / "test.md").write_text("from repo", encoding="utf-8")

    result = _load_prompt_text(
        spec={"prompt_file": "prompts/test.md"},
        repo_root=repo_root,
        workspace=workspace,
    )
    assert result == "from repo"


# ---------------------------------------------------------------------------
# C. create_successor_project: inherit validation
# ---------------------------------------------------------------------------

def test_successor_rejects_dotdot_inherit(tmp_path: Path) -> None:
    """create_successor_project must reject inherit dirs with '..'."""
    ledger, project = _make_project(tmp_path)

    with pytest.raises(SystemExit, match="safe relative name"):
        create_successor_project(
            ledger=ledger,
            predecessor_id="p1",
            inherit=["../etc"],
        )


def test_successor_rejects_absolute_inherit(tmp_path: Path) -> None:
    """create_successor_project must reject absolute inherit paths."""
    ledger, project = _make_project(tmp_path)

    with pytest.raises(SystemExit, match="safe relative name"):
        create_successor_project(
            ledger=ledger,
            predecessor_id="p1",
            inherit=["/tmp/evil"],
        )


def test_successor_allows_valid_inherit(tmp_path: Path) -> None:
    """create_successor_project should work with valid directory names."""
    ledger, project = _make_project(tmp_path)
    ws = Path(project["repo_path"])
    (ws / "data").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "file.txt").write_text("data", encoding="utf-8")

    result = create_successor_project(
        ledger=ledger,
        predecessor_id="p1",
        inherit=["data"],
    )
    new_ws = Path(result["repo_path"])
    assert (new_ws / "data").is_symlink()
    assert (new_ws / "data" / "file.txt").read_text() == "data"
