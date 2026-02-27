from __future__ import annotations

import json
from pathlib import Path

import pytest

from resorch.db_inventory import ensure_databases
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


def test_db_ensure_reports_missing_and_writes_artifacts(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="intake", git_init=False)
    ws = Path(project["repo_path"]).resolve()

    db_root = tmp_path / "db_root"
    db_root.mkdir()
    (db_root / "uniprot.fa").write_text(">x\nAAAA\n", encoding="utf-8")

    (ws / "constraints.yaml").write_text(
        "\n".join(
            [
                "databases:",
                f"  root: {db_root}",
                "  items:",
                "    - name: uniprot",
                "      path: uniprot.fa",
                "      license: CC-BY",
                "      version: v1",
                "      auto_download: false",
                "    - name: missing",
                "      path: missing.db",
                "      auto_download: false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    out = ensure_databases(ledger=ledger, project_id=project["id"])
    assert out["found"] == 1
    assert out["missing"] == 1

    report_path = (ws / out["report_path"]).resolve()
    script_path = (ws / out["script_path"]).resolve()
    assert report_path.exists()
    assert script_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report["found"]) == 1
    assert len(report["missing"]) == 1
    assert report["missing"][0]["name"] == "missing"


def test_db_ensure_requires_root_for_relative_paths(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="intake", git_init=False)
    ws = Path(project["repo_path"]).resolve()

    (ws / "constraints.yaml").write_text(
        "\n".join(
            [
                "databases:",
                "  root: \"\"",
                "  items:",
                "    - name: uniprot",
                "      path: uniprot.fa",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        ensure_databases(ledger=ledger, project_id=project["id"])

