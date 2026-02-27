from __future__ import annotations

from pathlib import Path

from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project, create_successor_project, get_project


def _make_ledger(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_create_project_with_idea_id_stores_meta(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)

    project = create_project(
        ledger=ledger,
        project_id="p-idea",
        title="Project With Idea",
        domain="test",
        stage="intake",
        git_init=False,
        idea_id="idea-123",
    )

    assert project["meta"]["idea_id"] == "idea-123"


def test_create_project_without_idea_id_is_backward_compatible(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)

    project = create_project(
        ledger=ledger,
        project_id="p-no-idea",
        title="Project Without Idea",
        domain="test",
        stage="intake",
        git_init=False,
    )

    assert project["id"] == "p-no-idea"
    assert "idea_id" not in project["meta"]


def test_get_project_returns_idea_id_in_meta(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    create_project(
        ledger=ledger,
        project_id="p-get-idea",
        title="Project Get Idea",
        domain="test",
        stage="intake",
        git_init=False,
        idea_id="idea-xyz",
    )

    fetched = get_project(ledger, "p-get-idea")

    assert fetched["meta"]["idea_id"] == "idea-xyz"


def test_create_successor_project_propagates_idea_id(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    create_project(
        ledger=ledger,
        project_id="pred",
        title="Predecessor",
        domain="test",
        stage="analysis",
        git_init=False,
        idea_id="idea-parent",
    )

    successor = create_successor_project(
        ledger=ledger,
        predecessor_id="pred",
        project_id="succ",
        git_init=False,
    )

    assert successor["meta"]["idea_id"] == "idea-parent"

