from __future__ import annotations

import json
from pathlib import Path

from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.reviews import _update_finding_recurrence, ingest_review_result, write_review_request


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


def test_ingest_no_duplicate_when_result_already_in_reviews_dir(tmp_path: Path) -> None:
    """Regression: when result_path is already inside reviews/, ingest must NOT
    create a second copy with a different UUID.  (Bug: 28/56 RESP files were
    duplicates in metallome-map-v4 because _run_*_job wrote RESP first, then
    ingest_review_result copied it with a new UUID.)"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")

    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()

    project = create_project(
        ledger=ledger,
        project_id="dup-test",
        title="Dup Test",
        domain="test",
        stage="intake",
        git_init=False,
    )

    reviews_dir = Path(project["repo_path"]) / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)

    # Simulate what _run_*_job does: write RESP directly into reviews_dir.
    result_data = {
        "project_id": project["id"],
        "stage": "intake",
        "reviewer": "claude_code",
        "recommendation": "minor",
        "findings": [
            {
                "severity": "minor",
                "category": "methodology",
                "message": "Consider additional baseline.",
                "target_paths": ["notes/method.md"],
            }
        ],
    }
    pre_existing = reviews_dir / "RESP-intake-20260301-abc123-claude_code.json"
    pre_existing.write_text(json.dumps(result_data), encoding="utf-8")

    files_before = set(reviews_dir.iterdir())

    ingested = ingest_review_result(ledger=ledger, result_path=pre_existing)

    files_after = set(reviews_dir.iterdir())
    # The only new file should be last_review_summary.md — NOT a second RESP.
    new_files = files_after - files_before
    resp_copies = [f for f in new_files if f.name.startswith("RESP-")]
    assert resp_copies == [], f"Duplicate RESP created: {[f.name for f in resp_copies]}"

    # The stored path should point to the original file, not a copy.
    assert Path(ingested["stored_path"]).resolve() == pre_existing.resolve()


def _make_review_result(project_id: str, stage: str, findings: list) -> dict:
    return {
        "project_id": project_id,
        "stage": stage,
        "reviewer": "claude_code",
        "recommendation": "major" if any(f.get("severity") == "major" for f in findings) else "minor",
        "findings": findings,
    }


def test_resolvability_appears_in_summary(tmp_path: Path) -> None:
    """Resolvability counts and limitation/pivot sections appear in summary."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")

    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    project = create_project(
        ledger=ledger, project_id="resolv-test", title="T", domain="test",
        stage="analysis", git_init=False,
    )

    reviews_dir = Path(project["repo_path"]) / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)

    resp = reviews_dir / "RESP-analysis-20260302-aaa111-claude_code.json"
    resp.write_text(json.dumps(_make_review_result(project["id"], "analysis", [
        {
            "severity": "major", "category": "method",
            "message": "Annotation circularity remains.", "target_paths": ["notes/method.md"],
            "resolvability": "inherent_limitation",
            "suggested_fix": "Acknowledge in Discussion as a limitation.",
        },
        {
            "severity": "major", "category": "analysis",
            "message": "Missing PLM baseline.", "target_paths": ["src/"],
            "resolvability": "fixable",
            "suggested_fix": "Run ESM-2 embedding baseline.",
        },
        {
            "severity": "major", "category": "method",
            "message": "Temporal holdout reverses delta.", "target_paths": ["notes/method.md"],
            "resolvability": "requires_pivot",
            "suggested_fix": "Consider alternative evaluation strategy.",
        },
    ])), encoding="utf-8")

    ingested = ingest_review_result(ledger=ledger, result_path=resp)
    summary = (reviews_dir / "last_review_summary.md").read_text(encoding="utf-8")

    assert "inherent_limitation: 1" in summary
    assert "fixable: 1" in summary
    assert "requires_pivot: 1" in summary
    assert "Inherent Limitations" in summary
    assert "Annotation circularity" in summary
    assert "Requires Pivot" in summary
    assert "Temporal holdout" in summary


def test_finding_recurrence_tracker(tmp_path: Path) -> None:
    """finding_recurrence.md is generated and flags 4+ occurrences."""
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()

    # Write 5 RESP files, each with a major "method" finding.
    for i in range(5):
        data = {
            "project_id": "test",
            "stage": "analysis",
            "reviewer": "claude_code",
            "recommendation": "major",
            "findings": [
                {
                    "severity": "major", "category": "method",
                    "message": f"Annotation circularity iteration {i}",
                    "target_paths": ["notes/method.md"],
                    "resolvability": "fixable" if i < 3 else "inherent_limitation",
                },
            ],
        }
        (reviews_dir / f"RESP-analysis-2026030{i}-uid{i:03d}-claude_code.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

    _update_finding_recurrence(reviews_dir)

    tracker = (reviews_dir / "finding_recurrence.md").read_text(encoding="utf-8")
    assert "method (5 occurrences)" in tracker
    assert "WARNING" in tracker
    assert "inherent_limitation" in tracker


def test_finding_recurrence_empty(tmp_path: Path) -> None:
    """No RESP files → tracker says no findings."""
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()

    _update_finding_recurrence(reviews_dir)

    tracker = (reviews_dir / "finding_recurrence.md").read_text(encoding="utf-8")
    assert "No major/blocker findings" in tracker


def test_schema_accepts_resolvability() -> None:
    """review_result.schema.json validates findings with resolvability field."""
    import jsonschema

    schema_path = Path(__file__).resolve().parent.parent / "review" / "review_result.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    doc = {
        "project_id": "test",
        "stage": "analysis",
        "reviewer": "claude_code",
        "recommendation": "major",
        "findings": [
            {
                "severity": "major", "category": "method",
                "message": "Some issue.", "target_paths": ["x.py"],
                "resolvability": "fixable",
            },
            {
                "severity": "major", "category": "analysis",
                "message": "Limitation.", "target_paths": ["y.py"],
                "resolvability": "inherent_limitation",
            },
            {
                "severity": "minor", "category": "writing",
                "message": "Typo.", "target_paths": ["z.md"],
                "resolvability": None,
            },
        ],
    }
    jsonschema.validate(doc, schema)  # should not raise
