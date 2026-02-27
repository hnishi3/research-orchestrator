"""Tests for the project succession / continuation feature."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import (
    _generate_predecessor_summary,
    create_project,
    create_successor_project,
    get_project,
)


def _make_ledger(tmp_path: Path) -> Ledger:
    """Create an initialized Ledger in *tmp_path*."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def _make_predecessor(ledger: Ledger, *, idea_id: Optional[str] = None) -> dict:
    """Create a predecessor project with sample artifacts."""
    pred = create_project(
        ledger=ledger,
        project_id="pred-alpha",
        title="Predecessor Alpha",
        domain="bio",
        stage="writing",
        git_init=False,
        idea_id=idea_id,
    )
    ws = Path(pred["repo_path"])

    # Add some data and src dirs with content
    (ws / "data").mkdir(exist_ok=True)
    (ws / "data" / "features.parquet").write_text("fake", encoding="utf-8")
    (ws / "src" / "01_analysis.py").write_text("print('hello')", encoding="utf-8")

    # Populate notes with realistic content
    (ws / "notes" / "problem.md").write_text(
        "# Problem\n\n- Question: Does X predict Y?\n- Hypothesis: Yes\n",
        encoding="utf-8",
    )
    (ws / "notes" / "method.md").write_text(
        "# Method\n\nRandom Forest + localization features.\n",
        encoding="utf-8",
    )
    (ws / "notes" / "analysis_digest.md").write_text(
        "# Analysis Digest\n\n## Latest\n- AUROC = 0.823\n",
        encoding="utf-8",
    )

    # Scoreboard
    sb = {
        "schema_version": 2,
        "primary_metric": {
            "name": "micro_avg_auroc",
            "direction": "maximize",
            "current": 0.823,
            "best": 0.823,
            "baseline": 0.75,
        },
        "runs": [{"id": "run-1"}, {"id": "run-2"}],
    }
    (ws / "results" / "scoreboard.json").write_text(
        json.dumps(sb, indent=2), encoding="utf-8"
    )
    return pred


# ---- _generate_predecessor_summary ----


class TestGeneratePredecessorSummary:
    def test_includes_problem(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        pred = _make_predecessor(ledger)
        summary = _generate_predecessor_summary(Path(pred["repo_path"]))
        assert "# Predecessor Project Summary" in summary
        assert "Does X predict Y" in summary

    def test_includes_scoreboard_metric(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        pred = _make_predecessor(ledger)
        summary = _generate_predecessor_summary(Path(pred["repo_path"]))
        assert "micro_avg_auroc" in summary
        assert "0.823" in summary

    def test_includes_analysis_digest(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        pred = _make_predecessor(ledger)
        summary = _generate_predecessor_summary(Path(pred["repo_path"]))
        assert "AUROC = 0.823" in summary

    def test_includes_method(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        pred = _make_predecessor(ledger)
        summary = _generate_predecessor_summary(Path(pred["repo_path"]))
        assert "Random Forest" in summary

    def test_empty_workspace(self, tmp_path: Path) -> None:
        ws = tmp_path / "empty_ws"
        ws.mkdir()
        (ws / "notes").mkdir()
        (ws / "results").mkdir()
        summary = _generate_predecessor_summary(ws)
        assert "# Predecessor Project Summary" in summary
        # Should not crash, just have header and timestamp
        assert "_Generated:" in summary


# ---- create_successor_project ----


class TestCreateSuccessorProject:
    def test_basic_successor(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        _make_predecessor(ledger)
        result = create_successor_project(
            ledger=ledger,
            predecessor_id="pred-alpha",
            project_id="succ-beta",
            title="Successor Beta",
            git_init=False,
        )
        assert result["id"] == "succ-beta"
        assert result["title"] == "Successor Beta"
        assert "predecessor" in result["meta"]
        assert result["meta"]["predecessor"]["id"] == "pred-alpha"

    def test_propagates_idea_id_from_predecessor(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        _make_predecessor(ledger, idea_id="idea-parent-1")
        result = create_successor_project(
            ledger=ledger,
            predecessor_id="pred-alpha",
            project_id="succ-idea-prop",
            git_init=False,
        )
        assert result["meta"]["idea_id"] == "idea-parent-1"

    def test_successor_idea_id_override(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        _make_predecessor(ledger, idea_id="idea-parent-1")
        result = create_successor_project(
            ledger=ledger,
            predecessor_id="pred-alpha",
            project_id="succ-idea-override",
            idea_id="idea-child-9",
            git_init=False,
        )
        assert result["meta"]["idea_id"] == "idea-child-9"

    def test_inherits_data_via_symlink(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        _make_predecessor(ledger)
        result = create_successor_project(
            ledger=ledger,
            predecessor_id="pred-alpha",
            project_id="succ-sym",
            git_init=False,
        )
        ws = Path(result["repo_path"])
        data_link = ws / "data"
        assert data_link.is_symlink()
        assert (data_link / "features.parquet").exists()

    def test_inherits_src_via_symlink(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        _make_predecessor(ledger)
        result = create_successor_project(
            ledger=ledger,
            predecessor_id="pred-alpha",
            project_id="succ-src",
            git_init=False,
        )
        ws = Path(result["repo_path"])
        src_link = ws / "src"
        assert src_link.is_symlink()
        assert (src_link / "01_analysis.py").exists()

    def test_predecessor_summary_created(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        _make_predecessor(ledger)
        result = create_successor_project(
            ledger=ledger,
            predecessor_id="pred-alpha",
            project_id="succ-sum",
            git_init=False,
        )
        ws = Path(result["repo_path"])
        summary = ws / "notes" / "predecessor_summary.md"
        assert summary.exists()
        content = summary.read_text(encoding="utf-8")
        assert "Predecessor Project Summary" in content
        assert "micro_avg_auroc" in content

    def test_default_title_from_predecessor(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        _make_predecessor(ledger)
        result = create_successor_project(
            ledger=ledger,
            predecessor_id="pred-alpha",
            project_id="succ-default",
            git_init=False,
        )
        assert "continuation" in result["title"].lower()

    def test_custom_inherit_dirs(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        _make_predecessor(ledger)
        result = create_successor_project(
            ledger=ledger,
            predecessor_id="pred-alpha",
            project_id="succ-custom",
            inherit=["data"],  # Only data, not src
            git_init=False,
        )
        ws = Path(result["repo_path"])
        assert (ws / "data").is_symlink()
        # src should NOT be symlinked — should be a regular dir from template
        assert not (ws / "src").is_symlink()
        assert result["inherited_dirs"] == ["data"]

    def test_empty_inherit_list(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        _make_predecessor(ledger)
        result = create_successor_project(
            ledger=ledger,
            predecessor_id="pred-alpha",
            project_id="succ-noinherit",
            inherit=[],
            git_init=False,
        )
        ws = Path(result["repo_path"])
        assert not (ws / "data").is_symlink()
        assert not (ws / "src").is_symlink()
        assert result["inherited_dirs"] == []

    def test_predecessor_not_found(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        with pytest.raises(SystemExit, match="not found"):
            create_successor_project(
                ledger=ledger,
                predecessor_id="nonexistent",
                project_id="succ-fail",
                git_init=False,
            )

    def test_meta_recorded_in_ledger(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        _make_predecessor(ledger)
        create_successor_project(
            ledger=ledger,
            predecessor_id="pred-alpha",
            project_id="succ-meta",
            git_init=False,
        )
        # Re-fetch from ledger to confirm persistence
        fetched = get_project(ledger, "succ-meta")
        assert fetched["meta"]["predecessor"]["id"] == "pred-alpha"
        assert fetched["meta"]["predecessor"]["title"] == "Predecessor Alpha"
        assert "data" in fetched["meta"]["predecessor"]["inherited_dirs"]

    def test_standard_template_files_present(self, tmp_path: Path) -> None:
        ledger = _make_ledger(tmp_path)
        _make_predecessor(ledger)
        result = create_successor_project(
            ledger=ledger,
            predecessor_id="pred-alpha",
            project_id="succ-tmpl",
            git_init=False,
        )
        ws = Path(result["repo_path"])
        # Template files should exist (not overwritten by symlinks)
        assert (ws / "notes" / "problem.md").exists()
        assert (ws / "notes" / "method.md").exists()
        assert (ws / "notes" / "analysis_digest.md").exists()
        assert (ws / "results" / "scoreboard.json").exists()
        assert (ws / "paper" / "manuscript.md").exists()
