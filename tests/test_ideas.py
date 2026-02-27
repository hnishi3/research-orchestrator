from __future__ import annotations

import json
from pathlib import Path

from resorch.artifacts import list_artifacts
from resorch.ideas import dedupe_ideas_jsonl, import_ideas_jsonl, list_ideas, score_ideas
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (repo_root / "rubrics").mkdir()
    (repo_root / "rubrics" / "idea_score_rubric.yaml").write_text(
        "\n".join(
            [
                "weights:",
                "  novelty: 0.25",
                "  feasibility: 0.25",
                "  impact: 0.20",
                "  clarity: 0.15",
                "  reusability: 0.10",
                "  risk_penalty: 0.05",
                "",
            ]
        ),
        encoding="utf-8",
    )
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_import_and_score_ideas(tmp_path: Path) -> None:
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
    ideas_path = ws / "ideas" / "ideas.jsonl"
    ideas_path.parent.mkdir(parents=True, exist_ok=True)
    ideas = [
        {"id": "idea1", "title": "Idea 1", "status": "candidate", "scores": {"novelty": 3, "feasibility": 4}},
        {"id": "idea2", "title": "Idea 2", "status": "candidate"},
    ]
    ideas_path.write_text("\n".join(json.dumps(i) for i in ideas) + "\n", encoding="utf-8")

    out = import_ideas_jsonl(ledger=ledger, project_id=project["id"], input_path="ideas/ideas.jsonl")
    assert out["imported"] == 2

    listed = list_ideas(ledger=ledger, project_id=project["id"], limit=10)
    assert {i["id"] for i in listed} == {"idea1", "idea2"}

    scored = score_ideas(
        ledger=ledger,
        project_id=project["id"],
        rubric_path="rubrics/idea_score_rubric.yaml",
        output_path="ideas/ranked.jsonl",
    )
    assert scored["count"] == 2
    ranked_path = ws / "ideas" / "ranked.jsonl"
    assert ranked_path.exists()

    arts = list_artifacts(ledger, project_id=project["id"], prefix="ideas/", limit=50)
    assert any(a["path"] == "ideas/ideas.jsonl" for a in arts)
    assert any(a["path"] == "ideas/ranked.jsonl" for a in arts)


def test_score_ideas_claude_provider(tmp_path: Path, monkeypatch) -> None:
    """When provider='claude', LLM scores should be applied to ideas."""
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
    ideas_path = ws / "ideas" / "ideas.jsonl"
    ideas_path.parent.mkdir(parents=True, exist_ok=True)
    ideas = [
        {"id": "idea1", "title": "Idea 1", "status": "candidate"},
        {"id": "idea2", "title": "Idea 2", "status": "candidate"},
    ]
    ideas_path.write_text("\n".join(json.dumps(i) for i in ideas) + "\n", encoding="utf-8")
    from resorch.ideas import import_ideas_jsonl
    import_ideas_jsonl(ledger=ledger, project_id="p1", input_path="ideas/ideas.jsonl")

    fake_scores = {"novelty": 4.0, "feasibility": 3.5, "impact": 5.0, "clarity": 4.0, "reusability": 2.0, "risk_penalty": 1.0}

    import resorch.ideas as ideas_mod
    monkeypatch.setattr(ideas_mod, "_score_idea_claude", lambda rec, ws_dir: dict(fake_scores))

    scored = score_ideas(
        ledger=ledger,
        project_id="p1",
        rubric_path="rubrics/idea_score_rubric.yaml",
        output_path="ideas/ranked_claude.jsonl",
        provider="claude",
    )
    assert scored["count"] == 2

    ranked_path = ws / "ideas" / "ranked_claude.jsonl"
    lines = ranked_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        rec = json.loads(line)
        assert rec["scores"]["novelty"] == 4.0
        assert rec["scores"]["impact"] == 5.0
        assert "total" in rec["scores"]


def test_score_ideas_claude_failopen(tmp_path: Path, monkeypatch) -> None:
    """When claude scoring raises, fallback to 2.5 (fail-open)."""
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
    ideas_path = ws / "ideas" / "ideas.jsonl"
    ideas_path.parent.mkdir(parents=True, exist_ok=True)
    ideas = [{"id": "idea1", "title": "Idea 1", "status": "candidate"}]
    ideas_path.write_text(json.dumps(ideas[0]) + "\n", encoding="utf-8")
    from resorch.ideas import import_ideas_jsonl
    import_ideas_jsonl(ledger=ledger, project_id="p1", input_path="ideas/ideas.jsonl")

    import resorch.ideas as ideas_mod
    monkeypatch.setattr(ideas_mod, "_score_idea_claude", lambda rec, ws_dir: (_ for _ in ()).throw(RuntimeError("LLM down")))

    scored = score_ideas(
        ledger=ledger,
        project_id="p1",
        rubric_path="rubrics/idea_score_rubric.yaml",
        output_path="ideas/ranked_fail.jsonl",
        provider="claude",
    )
    assert scored["count"] == 1
    ranked_path = ws / "ideas" / "ranked_fail.jsonl"
    rec = json.loads(ranked_path.read_text(encoding="utf-8").strip())
    for axis in ["novelty", "feasibility", "impact", "clarity", "reusability", "risk_penalty"]:
        assert rec["scores"][axis] == 2.5


def test_dedupe_ideas_jsonl(tmp_path: Path) -> None:
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
    ideas_path = ws / "ideas" / "ideas.jsonl"
    ideas_path.parent.mkdir(parents=True, exist_ok=True)
    ideas = [
        {"id": "idea1", "title": "Same", "one_sentence_claim": "Claim", "status": "candidate"},
        {"id": "idea1_dup", "title": "Same", "one_sentence_claim": "Claim", "status": "candidate"},
        {"id": "idea2", "title": "Other", "one_sentence_claim": "Different", "status": "candidate"},
    ]
    ideas_path.write_text("\n".join(json.dumps(i) for i in ideas) + "\n", encoding="utf-8")

    out = dedupe_ideas_jsonl(
        ledger=ledger,
        project_id=project["id"],
        input_path="ideas/ideas.jsonl",
        output_path="ideas/deduped.jsonl",
        threshold=0.95,
    )
    assert out["before"] == 3
    assert out["after"] == 2

    deduped_path = ws / "ideas" / "deduped.jsonl"
    mapping_path = Path(out["mapping_path"])
    assert deduped_path.exists()
    assert mapping_path.exists()

    arts = list_artifacts(ledger, project_id=project["id"], prefix="ideas/", limit=100)
    assert any(a["path"] == "ideas/deduped.jsonl" for a in arts)
