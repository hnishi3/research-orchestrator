from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import make_tmp_repo
from resorch.cli import build_parser
from resorch.projects import create_project
from resorch.topic_engine_loop import run_topic_engine


def _write_rubric(repo_root: Path) -> None:
    rubric = repo_root / "rubrics" / "idea_score_rubric.yaml"
    rubric.parent.mkdir(parents=True, exist_ok=True)
    rubric.write_text(
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


def _seed_idea(
    *,
    ledger: Any,
    project_id: str,
    idea_id: str,
    status: str = "candidate",
    score_total: float | None = None,
    scores: dict[str, float] | None = None,
) -> None:
    payload = {
        "id": idea_id,
        "title": f"title-{idea_id}",
        "status": status,
        "scores": scores or {},
    }
    ledger.upsert_idea(
        idea_id=idea_id,
        project_id=project_id,
        status=status,
        score_total=score_total,
        data=payload,
    )


def test_topic_engine_cli_parser() -> None:
    args = build_parser().parse_args(["topic", "engine", "--project", "p1", "--rounds", "4", "--dry-run"])
    assert args._handler == "topic_engine"
    assert args.project_id == "p1"
    assert args.rounds == 4
    assert args.dry_run is True


def test_topic_engine_dry_run_empty_project_returns_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = make_tmp_repo(tmp_path)
    create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )

    def _should_not_be_called(**_kwargs: Any) -> Any:
        raise AssertionError("create_task should not run in dry-run mode")

    monkeypatch.setattr("resorch.topic_engine_loop.create_task_fn", _should_not_be_called)

    out = run_topic_engine(ledger=ledger, project_id="p1", rounds=3, dry_run=True)
    assert out["stopped_reason"] == "max_rounds"
    assert out["cycles_run"] == 0
    assert out["ideas_generated"] == 0
    assert out["selected_idea_id"] is None


def test_topic_engine_dry_run_scores_and_activates_existing_ideas(tmp_path: Path) -> None:
    ledger = make_tmp_repo(tmp_path)
    _write_rubric(ledger.paths.root)
    create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="analysis",
        git_init=False,
    )
    _seed_idea(
        ledger=ledger,
        project_id="p1",
        idea_id="idea-a",
        scores={"novelty": 5.0, "feasibility": 4.0, "impact": 5.0},
    )
    _seed_idea(
        ledger=ledger,
        project_id="p1",
        idea_id="idea-b",
        scores={"novelty": 3.0, "feasibility": 3.0, "impact": 3.0},
    )
    _seed_idea(
        ledger=ledger,
        project_id="p1",
        idea_id="idea-c",
        scores={"novelty": 1.0, "feasibility": 1.0, "impact": 1.0},
    )

    out = run_topic_engine(ledger=ledger, project_id="p1", rounds=1, dry_run=True, top_k=2)
    assert out["cycles_run"] == 1
    assert out["ideas_generated"] == 0
    assert out["ideas_scored"] == 3
    assert out["selected_idea_id"] is None

    assert ledger.get_idea("idea-a")["status"] == "active"
    assert ledger.get_idea("idea-b")["status"] == "active"


def test_topic_engine_round_limit_respected_in_dry_run(tmp_path: Path) -> None:
    ledger = make_tmp_repo(tmp_path)
    create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="analysis",
        git_init=False,
    )
    _seed_idea(ledger=ledger, project_id="p1", idea_id="idea-only")

    out = run_topic_engine(ledger=ledger, project_id="p1", rounds=2, dry_run=True)
    assert out["cycles_run"] == 2
    assert out["stopped_reason"] == "max_rounds"
    assert out["selected_idea_id"] is None


def test_topic_engine_selects_smoke_passed_in_first_cycle(tmp_path: Path) -> None:
    ledger = make_tmp_repo(tmp_path)
    _write_rubric(ledger.paths.root)
    create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="analysis",
        git_init=False,
    )
    _seed_idea(
        ledger=ledger,
        project_id="p1",
        idea_id="idea-low",
        status="smoke_passed",
        score_total=1.0,
        scores={"novelty": 1.0, "feasibility": 1.0, "impact": 1.0, "clarity": 1.0, "reusability": 1.0},
    )
    _seed_idea(
        ledger=ledger,
        project_id="p1",
        idea_id="idea-high",
        status="smoke_passed",
        score_total=9.0,
        scores={"novelty": 5.0, "feasibility": 5.0, "impact": 5.0, "clarity": 5.0, "reusability": 5.0},
    )

    out = run_topic_engine(ledger=ledger, project_id="p1", rounds=3, dry_run=True)
    assert out["cycles_run"] == 1
    assert out["stopped_reason"] == "selected_found"
    assert out["selected_idea_id"] == "idea-high"
    assert out["topic_brief_path"] is not None
    assert Path(str(out["topic_brief_path"])).exists()
    assert ledger.get_idea("idea-high")["status"] == "selected"


def test_topic_engine_survives_score_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = make_tmp_repo(tmp_path)
    create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="analysis",
        git_init=False,
    )
    _seed_idea(ledger=ledger, project_id="p1", idea_id="idea-1")

    def _fail_score(**_kwargs: Any) -> Any:
        raise RuntimeError("score failed")

    monkeypatch.setattr("resorch.topic_engine_loop.score_ideas_fn", _fail_score)

    out = run_topic_engine(ledger=ledger, project_id="p1", rounds=1, dry_run=True)
    assert out["cycles_run"] == 1
    assert out["stopped_reason"] == "max_rounds"
    assert out["ideas_scored"] == 0


def test_topic_engine_survives_dedupe_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = make_tmp_repo(tmp_path)
    _write_rubric(ledger.paths.root)
    create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="analysis",
        git_init=False,
    )

    def _fake_generate(**kwargs: Any) -> str:
        project = kwargs["project"]
        workspace = Path(project["repo_path"])
        rel = "ideas/generated/fake.jsonl"
        out = workspace / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "\n".join(
                [
                    json.dumps({"id": "gen-1", "title": "Gen 1", "status": "candidate"}),
                    json.dumps({"id": "gen-2", "title": "Gen 2", "status": "candidate"}),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return rel

    def _fail_dedupe(**_kwargs: Any) -> Any:
        raise RuntimeError("dedupe failed")

    monkeypatch.setattr("resorch.topic_engine_loop._generate_ideas_via_codex", _fake_generate)
    monkeypatch.setattr("resorch.topic_engine_loop.dedupe_ideas_jsonl_fn", _fail_dedupe)

    out = run_topic_engine(ledger=ledger, project_id="p1", rounds=1, dry_run=False)
    assert out["cycles_run"] == 1
    assert out["stopped_reason"] == "max_rounds"
    assert out["ideas_generated"] == 2
    assert out["ideas_imported"] == 2
    assert ledger.get_idea("gen-1")["id"] == "gen-1"
    assert ledger.get_idea("gen-2")["id"] == "gen-2"
