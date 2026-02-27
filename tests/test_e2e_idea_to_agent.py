from __future__ import annotations

import json
from pathlib import Path

import pytest

from resorch.autopilot import _build_planner_prompt, _default_planner_context_files, _load_context_files
from resorch.idea_launcher import commit_and_launch
from resorch.ideas import get_idea, import_ideas_jsonl, score_ideas, set_idea_status
from resorch.ledger import Ledger
from resorch.manuscript_checker import check_manuscript_consistency, write_consistency_report
from resorch.paths import RepoPaths
from resorch.projects import create_project, get_project
from resorch.smoke_tests import ingest_smoke_test_result
from resorch.topic_brief import write_topic_brief
from resorch.verification_checklist import generate_verification_checklist, write_checklist


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")

    (repo_root / "schemas").mkdir(parents=True, exist_ok=True)
    (repo_root / "schemas" / "autopilot_plan.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "iteration": {"type": "integer"},
                    "actions": {"type": "array"},
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (repo_root / "rubrics").mkdir(parents=True, exist_ok=True)
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


def _write_ideas_jsonl(workspace: Path, ideas: list[dict], rel_path: str = "ideas/ideas.jsonl") -> str:
    path = workspace / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in ideas) + "\n", encoding="utf-8")
    return rel_path


def _write_smoke_result(workspace: Path, payload: dict, rel_path: str = "tmp_smoke.json") -> str:
    path = workspace / rel_path
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rel_path


def _checklist_items_by_id(checklist) -> dict[str, object]:
    return {item.id: item for item in checklist.items}


def _consistency_checks_by_id(report) -> dict[str, object]:
    return {check.check_id: check for check in report.checks}


def test_e2e_idea_import_to_project_lineage(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    intake = create_project(
        ledger=ledger,
        project_id="intake",
        title="Intake",
        domain="ml",
        stage="intake",
        git_init=False,
    )
    intake_ws = Path(intake["repo_path"])

    _write_ideas_jsonl(
        intake_ws,
        [
            {
                "id": "idea-lineage-1",
                "status": "candidate",
                "title": "Lineage idea",
                "one_sentence_claim": "Lineage claim",
            }
        ],
    )
    out = import_ideas_jsonl(ledger=ledger, project_id=intake["id"], input_path="ideas/ideas.jsonl")
    assert out["imported"] == 1

    project = create_project(
        ledger=ledger,
        project_id="agent-lineage",
        title="Agent Lineage",
        domain="ml",
        stage="analysis",
        git_init=False,
        idea_id="idea-lineage-1",
    )

    assert project["meta"]["idea_id"] == "idea-lineage-1"
    fetched = get_project(ledger, project["id"])
    assert fetched["meta"]["idea_id"] == "idea-lineage-1"


def test_e2e_idea_score_updates_idea_record(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="score-proj",
        title="Scoring",
        domain="ml",
        stage="intake",
        git_init=False,
    )
    ws = Path(project["repo_path"])

    _write_ideas_jsonl(
        ws,
        [
            {
                "id": "idea-score-1",
                "status": "candidate",
                "title": "Scored idea",
                "scores": {"novelty": 4.0, "feasibility": 3.0, "impact": 5.0},
            }
        ],
    )
    import_ideas_jsonl(ledger=ledger, project_id=project["id"], input_path="ideas/ideas.jsonl")

    result = score_ideas(
        ledger=ledger,
        project_id=project["id"],
        rubric_path="rubrics/idea_score_rubric.yaml",
        output_path="ideas/ranked.jsonl",
    )
    assert result["count"] == 1

    idea = get_idea(ledger=ledger, idea_id="idea-score-1")
    assert isinstance(idea["data"].get("scores"), dict)
    assert isinstance(idea["data"]["scores"].get("total"), float)
    assert idea["score_total"] == pytest.approx(float(idea["data"]["scores"]["total"]))


def test_e2e_smoke_ingest_updates_idea_status(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="smoke-proj",
        title="Smoke",
        domain="ml",
        stage="analysis",
        git_init=False,
    )
    ws = Path(project["repo_path"])

    _write_ideas_jsonl(
        ws,
        [{"id": "idea-smoke-1", "status": "candidate", "title": "Smoke idea"}],
    )
    import_ideas_jsonl(ledger=ledger, project_id=project["id"], input_path="ideas/ideas.jsonl")

    smoke_rel = _write_smoke_result(
        ws,
        {
            "idea_id": "idea-smoke-1",
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:05:00Z",
            "verdict": "pass",
            "metrics": [{"name": "loss", "value": 1.0}],
        },
    )
    ingest_out = ingest_smoke_test_result(ledger=ledger, project_id=project["id"], result_path=smoke_rel)
    assert ingest_out["smoke_test"]["verdict"] == "pass"

    idea = get_idea(ledger=ledger, idea_id="idea-smoke-1")
    assert idea["status"] == "smoke_passed"


def test_e2e_topic_brief_reads_idea_and_smoke(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="brief-proj",
        title="Topic Brief",
        domain="ml",
        stage="analysis",
        git_init=False,
    )
    ws = Path(project["repo_path"])

    claim_text = "E2E_TOPIC_BRIEF_CLAIM"
    _write_ideas_jsonl(
        ws,
        [
            {
                "id": "idea-brief-1",
                "status": "candidate",
                "title": "Brief idea",
                "one_sentence_claim": claim_text,
                "contribution_type": "method",
                "evaluation_plan": {
                    "datasets": ["dataset-a"],
                    "metrics": ["acc"],
                    "baselines": ["baseline-a"],
                    "ablations": ["ablation-a"],
                },
            }
        ],
    )
    import_ideas_jsonl(ledger=ledger, project_id=project["id"], input_path="ideas/ideas.jsonl")

    smoke_rel = _write_smoke_result(
        ws,
        {
            "idea_id": "idea-brief-1",
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:03:00Z",
            "verdict": "pass",
            "metrics": [{"name": "brief_metric", "value": 0.77}],
            "checkpoints": [{"name": "sanity", "status": "pass", "notes": "E2E_TOPIC_BRIEF_CHECKPOINT"}],
        },
    )
    ingest_smoke_test_result(ledger=ledger, project_id=project["id"], result_path=smoke_rel)

    out = write_topic_brief(
        ledger=ledger,
        project_id=project["id"],
        idea_id="idea-brief-1",
        output_path="topic_brief.md",
    )
    brief_path = Path(out["output_path"])
    assert brief_path.exists()

    brief_text = brief_path.read_text(encoding="utf-8")
    assert claim_text in brief_text
    assert "verdict: `pass`" in brief_text
    assert "brief_metric: 0.77" in brief_text
    assert "E2E_TOPIC_BRIEF_CHECKPOINT" in brief_text


def test_e2e_planner_context_includes_topic_brief(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="planner-context",
        title="Planner Context",
        domain="ml",
        stage="analysis",
        git_init=False,
    )
    ws = Path(project["repo_path"])
    marker = "E2E_TOPIC_BRIEF_CONTEXT_MARKER"
    (ws / "topic_brief.md").write_text(f"# Topic Brief\n\n{marker}\n", encoding="utf-8")

    rels = _default_planner_context_files(ws)
    assert "topic_brief.md" in rels

    loaded = _load_context_files(ws, rel_paths=rels, max_chars=10_000)
    assert any(rel == "topic_brief.md" and marker in txt for rel, txt in loaded)

    prompt, _schema, _validator = _build_planner_prompt(
        ledger=ledger,
        project_id=project["id"],
        objective="Use context files.",
        iteration=0,
        max_actions=3,
        context_files=rels,
    )
    assert marker in prompt


def test_e2e_full_pipeline_idea_to_planner_context(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)

    intake = create_project(
        ledger=ledger,
        project_id="pipeline-intake",
        title="Pipeline Intake",
        domain="ml",
        stage="intake",
        git_init=False,
    )
    intake_ws = Path(intake["repo_path"])

    idea_id = "idea-full-1"
    claim_marker = "E2E_FULL_PIPELINE_IDEA_CLAIM"
    _write_ideas_jsonl(
        intake_ws,
        [
            {
                "id": idea_id,
                "status": "candidate",
                "title": "Full pipeline idea",
                "one_sentence_claim": claim_marker,
                "contribution_type": "analysis",
                "evaluation_plan": {
                    "datasets": ["set-a"],
                    "metrics": ["metric-a"],
                    "baselines": ["base-a"],
                    "ablations": ["ablation-a"],
                },
            }
        ],
    )
    import_ideas_jsonl(ledger=ledger, project_id=intake["id"], input_path="ideas/ideas.jsonl")
    set_idea_status(ledger=ledger, idea_id=idea_id, status="active")

    project = create_project(
        ledger=ledger,
        project_id="pipeline-agent",
        title="Pipeline Agent",
        domain="ml",
        stage="analysis",
        git_init=False,
        idea_id=idea_id,
    )
    assert project["meta"]["idea_id"] == idea_id
    ws = Path(project["repo_path"])

    _write_ideas_jsonl(
        ws,
        [
            {
                "id": idea_id,
                "status": "active",
                "title": "Full pipeline idea",
                "one_sentence_claim": claim_marker,
                "contribution_type": "analysis",
                "evaluation_plan": {
                    "datasets": ["set-a"],
                    "metrics": ["metric-a"],
                    "baselines": ["base-a"],
                    "ablations": ["ablation-a"],
                },
                "scores": {"novelty": 4.0, "feasibility": 3.0, "impact": 4.0, "clarity": 3.5},
            }
        ],
    )
    import_ideas_jsonl(ledger=ledger, project_id=project["id"], input_path="ideas/ideas.jsonl")
    score_ideas(
        ledger=ledger,
        project_id=project["id"],
        rubric_path="rubrics/idea_score_rubric.yaml",
        output_path="ideas/ranked.jsonl",
    )
    scored_idea = get_idea(ledger=ledger, idea_id=idea_id)
    assert isinstance(scored_idea["data"]["scores"]["total"], float)

    smoke_rel = _write_smoke_result(
        ws,
        {
            "idea_id": idea_id,
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:04:00Z",
            "verdict": "pass",
            "metrics": [{"name": "full_metric", "value": 0.88}],
            "checkpoints": [{"name": "full_sanity", "status": "pass", "notes": "E2E_FULL_PIPELINE_SMOKE_NOTE"}],
        },
    )
    ingest_smoke_test_result(ledger=ledger, project_id=project["id"], result_path=smoke_rel)

    write_topic_brief(ledger=ledger, project_id=project["id"], idea_id=idea_id, output_path="topic_brief.md")

    playbook_marker = "E2E_FULL_PIPELINE_PLAYBOOK_LESSON"
    ledger.upsert_playbook_entry(
        entry_id="ptn_full_pipeline",
        topic="ml:full-pipeline",
        rule={"summary": playbook_marker},
    )

    scoreboard_path = ws / "results" / "scoreboard.json"
    scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8"))
    scoreboard["notes"] = "E2E_FULL_PIPELINE_SCOREBOARD_MARKER"
    scoreboard["primary_metric"]["name"] = "full_metric"
    scoreboard["primary_metric"]["direction"] = "maximize"
    scoreboard["primary_metric"]["current"] = {"mean": 0.88, "n_runs": 1}
    scoreboard_path.write_text(json.dumps(scoreboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    prompt, _schema, _validator = _build_planner_prompt(
        ledger=ledger,
        project_id=project["id"],
        objective="Plan from complete context.",
        iteration=1,
        max_actions=4,
    )

    assert claim_marker in prompt
    assert "E2E_FULL_PIPELINE_SMOKE_NOTE" in prompt
    assert playbook_marker in prompt
    assert "E2E_FULL_PIPELINE_SCOREBOARD_MARKER" in prompt


def test_e2e_idea_import_malformed_jsonl(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="malformed-import-proj",
        title="Malformed Import",
        domain="ml",
        stage="intake",
        git_init=False,
    )
    ws = Path(project["repo_path"])
    malformed_path = ws / "ideas" / "malformed.jsonl"
    malformed_path.parent.mkdir(parents=True, exist_ok=True)
    malformed_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "idea-malformed-1", "status": "candidate", "title": "valid-1"}),
                '{"id":"idea-malformed-bad",',
                json.dumps({"id": "idea-malformed-2", "status": "candidate", "title": "valid-2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        out = import_ideas_jsonl(
            ledger=ledger,
            project_id=project["id"],
            input_path="ideas/malformed.jsonl",
        )
    except (json.JSONDecodeError, SystemExit) as exc:
        message = str(exc)
        assert (
            "json" in message.lower()
            or "line" in message.lower()
            or "expecting" in message.lower()
        ), message
        imported_first = get_idea(ledger=ledger, idea_id="idea-malformed-1")
        assert imported_first["id"] == "idea-malformed-1"
    else:
        assert out["imported"] == 2
        assert get_idea(ledger=ledger, idea_id="idea-malformed-1")["id"] == "idea-malformed-1"
        assert get_idea(ledger=ledger, idea_id="idea-malformed-2")["id"] == "idea-malformed-2"


def test_e2e_commit_launch_missing_idea(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    with pytest.raises(SystemExit, match="Idea not found: missing-idea-e2e"):
        commit_and_launch(
            ledger=ledger,
            repo_paths=ledger.paths,
            idea_id="missing-idea-e2e",
            dry_run=True,
        )


def test_e2e_commit_launch_already_in_progress(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="launch-status-proj",
        title="Launch Status",
        domain="ml",
        stage="intake",
        git_init=False,
    )
    ws = Path(project["repo_path"])
    _write_ideas_jsonl(
        ws,
        [
            {
                "id": "idea-in-progress-1",
                "status": "candidate",
                "title": "Already launched",
                "description": "Should not relaunch while in progress",
            }
        ],
    )
    import_ideas_jsonl(ledger=ledger, project_id=project["id"], input_path="ideas/ideas.jsonl")
    set_idea_status(ledger=ledger, idea_id="idea-in-progress-1", status="in_progress")

    with pytest.raises(SystemExit, match="not launchable.*in_progress"):
        commit_and_launch(
            ledger=ledger,
            repo_paths=ledger.paths,
            idea_id="idea-in-progress-1",
            dry_run=True,
        )


def test_e2e_topic_brief_without_smoke_data(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="brief-no-smoke-proj",
        title="Brief Without Smoke",
        domain="ml",
        stage="analysis",
        git_init=False,
    )
    ws = Path(project["repo_path"])
    claim_marker = "E2E_BRIEF_WITHOUT_SMOKE_CLAIM"
    _write_ideas_jsonl(
        ws,
        [
            {
                "id": "idea-brief-nosmoke-1",
                "status": "candidate",
                "title": "No smoke data idea",
                "one_sentence_claim": claim_marker,
            }
        ],
    )
    import_ideas_jsonl(ledger=ledger, project_id=project["id"], input_path="ideas/ideas.jsonl")

    out = write_topic_brief(
        ledger=ledger,
        project_id=project["id"],
        idea_id="idea-brief-nosmoke-1",
        output_path="topic_brief.md",
    )
    brief_path = Path(out["output_path"])
    brief_text = brief_path.read_text(encoding="utf-8")

    assert brief_path.exists()
    assert claim_marker in brief_text
    assert "(no smoke test results recorded)" in brief_text


def test_e2e_planner_context_missing_files(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="planner-missing-context",
        title="Planner Missing Context",
        domain="ml",
        stage="analysis",
        git_init=False,
    )
    ws = Path(project["repo_path"])

    (ws / "topic_brief.md").unlink(missing_ok=True)
    rels = _default_planner_context_files(ws)
    loaded = _load_context_files(ws, rel_paths=rels, max_chars=10_000)
    loaded_rels = {rel for rel, _txt in loaded}

    assert "notes/problem.md" in rels
    assert "results/scoreboard.json" in rels
    assert "notes/problem.md" in loaded_rels
    assert "results/scoreboard.json" in loaded_rels
    assert "topic_brief.md" not in loaded_rels


def test_e2e_scoreboard_empty_pipeline(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="empty-scoreboard-proj",
        title="Empty Scoreboard",
        domain="ml",
        stage="analysis",
        git_init=False,
    )
    ws = Path(project["repo_path"])
    (ws / "results" / "scoreboard.json").write_text("{}\n", encoding="utf-8")

    checklist = generate_verification_checklist(ws, project_id=project["id"], include_manuscript_checks=True)
    items = _checklist_items_by_id(checklist)

    assert len(checklist.items) == 17
    assert items["metric_baseline_stated"].auto_status == "fail"
    assert items["metric_current_vs_baseline"].auto_status == "fail"
    assert items["metric_reproducible"].auto_status in {"needs_human", "fail"}
    assert items["code_tests_pass"].auto_status in {"needs_human", "fail"}
    assert checklist.fail_count > 0
    assert checklist.needs_human_count > 0


def test_e2e_full_pipeline_with_verification(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    intake = create_project(
        ledger=ledger,
        project_id="verify-intake",
        title="Verification Intake",
        domain="ml",
        stage="intake",
        git_init=False,
    )
    intake_ws = Path(intake["repo_path"])

    idea_id = "idea-verify-full-1"
    _write_ideas_jsonl(
        intake_ws,
        [
            {
                "id": idea_id,
                "status": "candidate",
                "title": "Full verification pipeline idea",
                "description": "Run full e2e verification pipeline",
                "one_sentence_claim": "Full pipeline produces consistent artifacts.",
                "contribution_type": "analysis",
                "evaluation_plan": {
                    "datasets": ["dataset-a"],
                    "metrics": ["full_metric"],
                    "baselines": ["baseline-a"],
                    "ablations": ["ablation-a"],
                },
                "scores": {"novelty": 4.0, "feasibility": 3.5, "impact": 4.5, "clarity": 4.0},
            }
        ],
    )

    import_out = import_ideas_jsonl(ledger=ledger, project_id=intake["id"], input_path="ideas/ideas.jsonl")
    assert import_out["imported"] == 1

    score_out = score_ideas(
        ledger=ledger,
        project_id=intake["id"],
        rubric_path="rubrics/idea_score_rubric.yaml",
        output_path="ideas/ranked.jsonl",
    )
    assert score_out["count"] == 1

    smoke_rel = _write_smoke_result(
        intake_ws,
        {
            "idea_id": idea_id,
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:05:00Z",
            "verdict": "pass",
            "metrics": [{"name": "full_metric", "value": 0.88}],
            "checkpoints": [{"name": "sanity", "status": "pass", "notes": "E2E_FULL_VERIFY_SMOKE_NOTE"}],
        },
    )
    smoke_out = ingest_smoke_test_result(ledger=ledger, project_id=intake["id"], result_path=smoke_rel)
    assert smoke_out["smoke_test"]["verdict"] == "pass"

    brief_out = write_topic_brief(ledger=ledger, project_id=intake["id"], idea_id=idea_id, output_path="topic_brief.md")
    source_brief = Path(brief_out["output_path"])
    assert source_brief.exists()

    project = create_project(
        ledger=ledger,
        project_id="verify-agent",
        title="Verification Agent",
        domain="ml",
        stage="analysis",
        git_init=False,
        idea_id=idea_id,
    )
    ws = Path(project["repo_path"])
    (ws / "topic_brief.md").write_text(source_brief.read_text(encoding="utf-8"), encoding="utf-8")
    (ws / "src" / "main.py").write_text("def run() -> float:\n    return 0.88\n", encoding="utf-8")

    (ws / "notes" / "method.md").write_text(
        "\n".join(
            [
                "# Method",
                "## Metric Definitions",
                "- Metric: full_metric",
                "- Definition: end-to-end quality score",
                "- Direction: maximize",
                "- Baseline: 0.8",
                "## Data Sources",
                "- dataset-a",
                "## Preprocessing",
                "- normalization and filtering",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scoreboard_payload = {
        "schema_version": 2,
        "primary_metric": {
            "name": "full_metric",
            "direction": "maximize",
            "current": {"mean": 0.88, "n_runs": 3, "ci_95": [0.86, 0.9]},
            "best": {"mean": 0.88},
            "baseline": {"mean": 0.8},
            "delta_vs_baseline": 0.08,
        },
        "metrics": {
            "full_metric": 0.88,
            "test_fail_count": 0,
            "test_pass_count": 21,
            "test_pass_count_baseline": 20,
        },
        "runs": [
            {"metrics": {"full_metric": 0.84, "test_pass_count": 20}},
            {"metrics": {"full_metric": 0.86, "test_pass_count": 21}},
            {"metrics": {"full_metric": 0.88, "test_pass_count": 21}},
        ],
    }
    (ws / "results" / "scoreboard.json").write_text(
        json.dumps(scoreboard_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    (ws / "paper" / "manuscript.md").write_text(
        "\n".join(
            [
                "# Full Pipeline Paper",
                "",
                "## Results",
                "We achieved full_metric = 0.88 on dataset-a.",
                "Statistical significance was observed at p < 0.05 with Cohen's d = 0.6.",
                "",
                "## References",
                "1. Example Study. DOI:10.1234/example.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    checklist = generate_verification_checklist(ws, project_id=project["id"], include_manuscript_checks=True)
    checklist_path = write_checklist(ws, checklist)

    consistency_report = check_manuscript_consistency(ws)
    consistency_path = write_consistency_report(ws, consistency_report)

    checklist_items = _checklist_items_by_id(checklist)
    consistency_checks = _consistency_checks_by_id(consistency_report)

    assert Path(score_out["output_path"]).exists()
    assert Path(smoke_out["stored_path"]).exists()
    assert source_brief.exists()
    assert (ws / "topic_brief.md").exists()
    assert checklist_path.exists()
    assert consistency_path.exists()

    assert consistency_checks["scoreboard_exists"].passed is True
    assert consistency_checks["scoreboard_has_primary_metric"].passed is True
    assert consistency_checks["text_numbers_match_scoreboard"].passed is True
    assert checklist_items["metric_current_vs_baseline"].auto_status == "pass"
    assert checklist_items["metric_reproducible"].auto_status == "pass"
    assert checklist_items["manuscript_refs_have_dois"].auto_status == "pass"
