from __future__ import annotations

from pathlib import Path

import pytest

from resorch.ledger import Ledger
from resorch.paths import RepoPaths


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_ledger_transaction_commits_on_success(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    with ledger.transaction():
        ledger.insert_project(
            project_id="p1",
            title="P1",
            domain="",
            stage="intake",
            repo_path=str(tmp_path / "ws"),
            meta={},
        )
    assert ledger.get_project("p1")["id"] == "p1"


def test_ledger_transaction_rolls_back_on_error(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    try:
        with ledger.transaction():
            ledger.insert_project(
                project_id="p1",
                title="P1",
                domain="",
                stage="intake",
                repo_path=str(tmp_path / "ws"),
                meta={},
            )
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    with pytest.raises(SystemExit):
        ledger.get_project("p1")


def test_ledger_transaction_allows_nesting(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    with ledger.transaction():
        ledger.insert_project(
            project_id="p1",
            title="P1",
            domain="",
            stage="intake",
            repo_path=str(tmp_path / "ws"),
            meta={},
        )
        with ledger.transaction():
            ledger.update_project_stage("p1", "analysis")

    assert ledger.get_project("p1")["stage"] == "analysis"

