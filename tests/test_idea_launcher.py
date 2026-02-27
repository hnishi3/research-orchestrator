from __future__ import annotations

import json
from pathlib import Path

import pytest

from resorch.cli import build_parser
from resorch.idea_launcher import commit_and_launch
from resorch.ideas import import_ideas_jsonl
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project, get_project


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")

    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def _seed_imported_idea(ledger: Ledger, idea_id: str = "idea-launch-1") -> dict:
    source_project = create_project(
        ledger=ledger,
        project_id="intake",
        title="Intake",
        domain="ml",
        stage="intake",
        git_init=False,
    )

    workspace = Path(source_project["repo_path"])
    idea = {
        "id": idea_id,
        "status": "candidate",
        "title": "Launchable Idea",
        "domain": "ml",
        "description": "Evaluate the commit-and-launch flow.",
        "objectives": ["Create project", "Keep idea lineage"],
        "success_criteria": ["topic_brief.md exists", "notes/problem.md is populated"],
        "one_sentence_claim": "Commit-and-launch reduces handoff friction.",
        "evaluation_plan": {
            "datasets": ["demo-dataset"],
            "metrics": ["accuracy"],
            "baselines": ["manual workflow"],
            "ablations": ["without lineage"],
        },
    }

    ideas_path = workspace / "ideas" / "ideas.jsonl"
    ideas_path.parent.mkdir(parents=True, exist_ok=True)
    ideas_path.write_text(json.dumps(idea) + "\n", encoding="utf-8")

    import_ideas_jsonl(ledger=ledger, project_id=source_project["id"], input_path="ideas/ideas.jsonl")
    return source_project


def test_commit_and_launch_creates_project(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    _seed_imported_idea(ledger)

    result = commit_and_launch(
        ledger=ledger,
        repo_paths=ledger.paths,
        idea_id="idea-launch-1",
        dry_run=True,
    )

    project = get_project(ledger, result["project_id"])
    assert Path(result["workspace_path"]).exists()
    assert project["meta"]["idea_id"] == "idea-launch-1"
    assert result["idea_id"] == "idea-launch-1"


def test_commit_and_launch_sets_idea_status(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    _seed_imported_idea(ledger)

    commit_and_launch(
        ledger=ledger,
        repo_paths=ledger.paths,
        idea_id="idea-launch-1",
        dry_run=True,
    )

    assert ledger.get_idea("idea-launch-1")["status"] == "selected"


def test_commit_and_launch_creates_topic_brief(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    _seed_imported_idea(ledger)

    result = commit_and_launch(
        ledger=ledger,
        repo_paths=ledger.paths,
        idea_id="idea-launch-1",
        dry_run=True,
    )

    topic_brief_path = Path(result["topic_brief_path"])
    assert topic_brief_path.exists()
    assert topic_brief_path.name == "topic_brief.md"


def test_commit_and_launch_creates_problem_md(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    _seed_imported_idea(ledger)

    result = commit_and_launch(
        ledger=ledger,
        repo_paths=ledger.paths,
        idea_id="idea-launch-1",
        dry_run=True,
    )

    problem_path = Path(result["workspace_path"]) / "notes" / "problem.md"
    text = problem_path.read_text(encoding="utf-8")

    assert problem_path.exists()
    assert "Launchable Idea" in text
    assert "Evaluate the commit-and-launch flow." in text


def test_commit_and_launch_dry_run(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    _seed_imported_idea(ledger)

    result = commit_and_launch(
        ledger=ledger,
        repo_paths=ledger.paths,
        idea_id="idea-launch-1",
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert "set_idea_in_progress" not in result["steps_taken"]
    assert ledger.get_idea("idea-launch-1")["status"] == "selected"


def test_commit_and_launch_invalid_idea_fails(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)

    with pytest.raises(SystemExit, match="Idea not found"):
        commit_and_launch(
            ledger=ledger,
            repo_paths=ledger.paths,
            idea_id="missing-idea",
            dry_run=True,
        )


def test_commit_and_launch_cli_parser() -> None:
    args = build_parser().parse_args(
        [
            "idea",
            "commit-and-launch",
            "--idea-id",
            "idea-123",
            "--title",
            "Demo Title",
            "--domain",
            "ml",
            "--objective",
            "Run the experiment",
            "--dry-run",
            "--max-steps",
            "5",
        ]
    )

    assert args._handler == "idea_commit_and_launch"
    assert args.idea_id == "idea-123"
    assert args.title == "Demo Title"
    assert args.domain == "ml"
    assert args.objective == "Run the experiment"
    assert args.dry_run is True
    assert args.max_steps == 5
