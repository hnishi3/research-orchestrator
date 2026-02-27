from __future__ import annotations

import json
from pathlib import Path

from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.reviews import ingest_review_result, write_review_request


def test_review_request_and_ingest(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")

    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()

    project = create_project(
        ledger=ledger,
        project_id="p2",
        title="P2",
        domain="",
        stage="intake",
        git_init=False,
    )

    req = write_review_request(
        ledger=ledger,
        project=project,
        stage="intake",
        mode="balanced",
        targets=["notes/problem.md"],
        questions=["Is the question clear?"],
        rubric=None,
        time_budget_minutes=10,
    )
    assert Path(req["packet_path"]).exists()
    assert Path(req["request_json_path"]).exists()

    # Create a review result JSON outside the workspace; ingest should copy into workspace/reviews/.
    result_path = tmp_path / "review_result.json"
    result_path.write_text(
        json.dumps(
            {
                "project_id": project["id"],
                "stage": "intake",
                "reviewer": "tester",
                "recommendation": "minor",
                "findings": [
                    {
                        "severity": "minor",
                        "category": "writing",
                        "message": "Add a one-sentence claim.",
                        "target_paths": ["notes/problem.md"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ingested = ingest_review_result(ledger=ledger, result_path=result_path)
    stored = Path(ingested["stored_path"])
    assert stored.exists()
    assert stored.parent.name == "reviews"
    assert (stored.parent / "last_review_summary.md").exists()
    assert len(ingested["tasks_created"]) == 1
    assert ingested["tasks_created"][0]["type"] == "review_fix"


def test_review_request_embeds_rubric(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (repo_root / "rubric.txt").write_text("REDTEAM PROMPT\n", encoding="utf-8")

    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()

    project = create_project(
        ledger=ledger,
        project_id="p3",
        title="P3",
        domain="",
        stage="intake",
        git_init=False,
    )

    req = write_review_request(
        ledger=ledger,
        project=project,
        stage="redteam",
        mode="balanced",
        targets=["topic_brief.md"],
        questions=["List 3 blockers."],
        rubric="rubric.txt",
        time_budget_minutes=None,
    )
    packet_txt = Path(req["packet_path"]).read_text(encoding="utf-8")
    assert "## Rubric / Prompt (embedded)" in packet_txt
    assert "REDTEAM PROMPT" in packet_txt
