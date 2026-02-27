from __future__ import annotations

import json
from pathlib import Path

from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.smoke_tests import ingest_smoke_test_result, list_smoke_tests
from resorch.stage_gates import compute_gate_env


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_smoke_test_ingest_and_list(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )
    ws = Path(project["repo_path"])

    ledger.upsert_idea(
        idea_id="idea1",
        project_id=project["id"],
        status="candidate",
        score_total=None,
        data={"id": "idea1", "status": "candidate"},
    )

    raw = {
        "idea_id": "idea1",
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:10:00Z",
        "verdict": "pass",
        "checkpoints": [{"name": "train_1_epoch", "status": "pass"}],
        "metrics": [{"name": "loss", "value": 1.0}],
    }
    (ws / "tmp_smoke.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    out = ingest_smoke_test_result(ledger=ledger, project_id=project["id"], result_path="tmp_smoke.json")
    assert out["smoke_test"]["idea_id"] == "idea1"
    assert out["smoke_test"]["verdict"] == "pass"
    assert out["artifact"] is not None
    assert out["idea"]["status"] == "smoke_passed"

    stored = Path(out["stored_path"])
    assert stored.exists()
    assert stored.name == "result.json"
    assert stored.parent.name == "idea1"

    rows = list_smoke_tests(ledger=ledger, project_id=project["id"], limit=10)
    assert len(rows) == 1
    assert rows[0]["idea_id"] == "idea1"
    assert rows[0]["result"]["verdict"] == "pass"

    env = compute_gate_env(ledger=ledger, project_id=project["id"])
    assert env["smoke_test"]["verdict"] == "pass"
