from __future__ import annotations

import json
from pathlib import Path

import pytest

from resorch.artifacts import list_artifacts
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.summary_ingest import ingest_summary


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_summary_ingest_updates_scoreboard_and_digest(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)
    ws = Path(project["repo_path"]).resolve()

    summary = {
        "primary_metric": {"name": "acc", "direction": "maximize", "value": 0.7, "baseline": 0.6},
        "metrics": {"tm_score": 0.8, "rmsd": 2.1},
        "notes": "ok",
    }
    (ws / "results" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out = ingest_summary(ledger=ledger, project_id=project["id"], summary_path="results/summary.json")
    assert out["summary_path"] == "results/summary.json"

    scoreboard = json.loads((ws / "results" / "scoreboard.json").read_text(encoding="utf-8"))
    assert scoreboard["primary_metric"]["current"]["mean"] == 0.7
    assert scoreboard["primary_metric"]["best"] == 0.7
    assert scoreboard["primary_metric"]["baseline"]["mean"] == 0.6
    assert scoreboard["primary_metric"]["delta_vs_baseline"] == pytest.approx(0.1)
    assert scoreboard["metrics"]["tm_score"] == 0.8
    assert scoreboard["runs"][-1]["source"] == "summary_ingest"

    digest = (ws / "notes" / "analysis_digest.md").read_text(encoding="utf-8")
    assert "Pipeline Summary Ingest" in digest

    arts = list_artifacts(ledger, project_id=project["id"], prefix="", limit=200)
    assert any(a["kind"] == "pipeline_summary_json" and a["path"] == "results/summary.json" for a in arts)
