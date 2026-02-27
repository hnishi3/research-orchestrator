from __future__ import annotations

from pathlib import Path

from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project, get_project, list_projects


def test_ledger_init_and_project_crud(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")

    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()

    project = create_project(
        ledger=ledger,
        project_id="demo",
        title="Demo",
        domain="test",
        stage="intake",
        git_init=False,
    )
    assert project["id"] == "demo"
    assert (Path(project["repo_path"]) / "notes" / "problem.md").exists()
    assert (Path(project["repo_path"]) / "notes" / "method.md").exists()
    assert (Path(project["repo_path"]) / "notes" / "analysis_digest.md").exists()
    assert (Path(project["repo_path"]) / "results" / "scoreboard.json").exists()

    fetched = get_project(ledger, "demo")
    assert fetched["title"] == "Demo"

    projects = list_projects(ledger)
    assert [p["id"] for p in projects] == ["demo"]


def test_create_project_stores_idea_id_in_meta(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")

    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()

    project = create_project(
        ledger=ledger,
        project_id="demo-idea",
        title="Demo Idea",
        domain="test",
        stage="intake",
        git_init=False,
        idea_id="idea-123",
    )
    assert project["meta"]["idea_id"] == "idea-123"
