from __future__ import annotations

from pathlib import Path

from resorch.artifacts import list_artifacts
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.smoke_tests import ingest_smoke_test_result
from resorch.topic_brief import write_topic_brief


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_write_topic_brief(tmp_path: Path) -> None:
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
        score_total=3.2,
        data={
            "id": "idea1",
            "status": "candidate",
            "title": "Demo Idea",
            "one_sentence_claim": "We can measure X.",
            "contribution_type": "analysis",
            "target_venues": ["arXiv"],
            "novelty_statement": "Different from prior work by Y.",
            "evaluation_plan": {"datasets": ["d1"], "metrics": ["m1"], "baselines": ["b1"], "ablations": ["a1"]},
            "feasibility": {"estimated_gpu_hours": 1, "estimated_calendar_days": 1, "blocking_dependencies": []},
            "risks": {"ethics": "low", "license": "low", "safety": "low", "reproducibility": "high"},
            "evidence": [],
        },
    )

    # Include smoke test for the idea so the brief can summarize it.
    (ws / "smoke.json").write_text(
        "\n".join(
            [
                "{",
                '  "idea_id": "idea1",',
                '  "started_at": "2026-01-01T00:00:00Z",',
                '  "completed_at": "2026-01-01T00:05:00Z",',
                '  "verdict": "pass",',
                '  "metrics": [{"name":"loss","value":1.0}],',
                '  "checkpoints": [{"name":"train","status":"pass"}]',
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ingest_smoke_test_result(ledger=ledger, project_id=project["id"], result_path="smoke.json")

    out = write_topic_brief(
        ledger=ledger,
        project_id=project["id"],
        idea_id="idea1",
        output_path="topic_brief.md",
        register_as_artifact=True,
        set_selected=True,
    )
    brief_path = Path(out["output_path"])
    assert brief_path.exists()
    txt = brief_path.read_text(encoding="utf-8")
    assert "idea_id: `idea1`" in txt
    assert "We can measure X." in txt
    assert "verdict: `pass`" in txt

    arts = list_artifacts(ledger, project_id=project["id"], prefix=None, limit=200)
    assert any(a["path"] == "topic_brief.md" for a in arts)

    idea_row = ledger.get_idea("idea1")
    assert idea_row["status"] == "selected"

