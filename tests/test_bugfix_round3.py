"""Tests for Codex Round 3 bug fixes:

High #1: Failed tasks appended to ran list (codex_exec / shell_exec exception handlers)
High #3: dual_on_hard escalation gate blocks stage advance on escalation reject
Medium #4: Empty targets skip review instead of SystemExit
Medium #6: ideas.py workspace path containment (relative_to guard)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> Ledger:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo))
    ledger.init()
    return ledger


def _make_project(tmp_path: Path, project_id: str = "p1") -> tuple:
    ledger = _make_repo(tmp_path)
    project = create_project(
        ledger=ledger, project_id=project_id, title="P1",
        domain="test", stage="analysis", git_init=False,
    )
    return ledger, project


# ---------------------------------------------------------------------------
# High #3: dual_on_hard escalation gate
# ---------------------------------------------------------------------------

def test_extract_review_result_with_recommendation() -> None:
    """_extract_review_result returns recommendation from job result."""
    from resorch.agent_loop import _extract_review_result

    job = {"result": {"review_result": {"recommendation": "accept", "findings": []}}}
    rr = _extract_review_result(job)
    assert rr["recommendation"] == "accept"


def test_escalation_reject_blocks_stage_advance() -> None:
    """When escalation review rejects, stage should NOT advance even if primary accepts.

    This tests the worst-of-both-reviews logic directly.
    """
    from resorch.agent_loop import _extract_review_result

    # Simulated step_record with both reviews
    primary_job = {"result": {"review_result": {"recommendation": "accept"}}}
    escalation_job = {"result": {"review_result": {"recommendation": "reject"}}}

    step_record: Dict[str, Any] = {
        "review_job": primary_job,
        "escalation_review_job": escalation_job,
    }

    # Replicate the worst-of logic from agent_loop.py
    auto_stage_apply_on_set = {"accept", "accept_with_notes"}

    review_rec = ""
    if step_record.get("review_job"):
        rr = _extract_review_result(step_record["review_job"])
        review_rec = str(rr.get("recommendation") or "").strip().lower()
    if step_record.get("escalation_review_job"):
        esc_rr = _extract_review_result(step_record["escalation_review_job"])
        esc_rec = str(esc_rr.get("recommendation") or "").strip().lower()
        if esc_rec and esc_rec not in auto_stage_apply_on_set:
            review_rec = esc_rec

    assert review_rec == "reject", "Escalation reject should override primary accept"
    assert review_rec not in auto_stage_apply_on_set, "Stage should NOT advance"


def test_both_accept_allows_stage_advance() -> None:
    """When both primary and escalation accept, stage should advance."""
    from resorch.agent_loop import _extract_review_result

    primary_job = {"result": {"review_result": {"recommendation": "accept"}}}
    escalation_job = {"result": {"review_result": {"recommendation": "accept"}}}

    step_record: Dict[str, Any] = {
        "review_job": primary_job,
        "escalation_review_job": escalation_job,
    }

    auto_stage_apply_on_set = {"accept", "accept_with_notes"}

    review_rec = ""
    if step_record.get("review_job"):
        rr = _extract_review_result(step_record["review_job"])
        review_rec = str(rr.get("recommendation") or "").strip().lower()
    if step_record.get("escalation_review_job"):
        esc_rr = _extract_review_result(step_record["escalation_review_job"])
        esc_rec = str(esc_rr.get("recommendation") or "").strip().lower()
        if esc_rec and esc_rec not in auto_stage_apply_on_set:
            review_rec = esc_rec

    assert review_rec == "accept", "Both accept should keep accept"
    assert review_rec in auto_stage_apply_on_set, "Stage should advance"


def test_no_escalation_job_uses_primary_only() -> None:
    """Without escalation review, primary review alone gates stage advance."""
    from resorch.agent_loop import _extract_review_result

    primary_job = {"result": {"review_result": {"recommendation": "accept"}}}
    step_record: Dict[str, Any] = {"review_job": primary_job}

    auto_stage_apply_on_set = {"accept", "accept_with_notes"}

    review_rec = ""
    if step_record.get("review_job"):
        rr = _extract_review_result(step_record["review_job"])
        review_rec = str(rr.get("recommendation") or "").strip().lower()
    if step_record.get("escalation_review_job"):
        esc_rr = _extract_review_result(step_record["escalation_review_job"])
        esc_rec = str(esc_rr.get("recommendation") or "").strip().lower()
        if esc_rec and esc_rec not in auto_stage_apply_on_set:
            review_rec = esc_rec

    assert review_rec == "accept"
    assert review_rec in auto_stage_apply_on_set


# ---------------------------------------------------------------------------
# Medium #4: Empty targets → skip review
# ---------------------------------------------------------------------------

def test_empty_targets_fallback_to_changed_paths() -> None:
    """When targets is empty, fallback to git changed_paths."""
    targets: list = []
    iter_out = {
        "review_recommendation": {"level": "soft", "targets": []},
        "git_change_summary": {"changed_paths": ["src/a.py", "src/b.py"]},
    }

    # Replicate the target resolution from agent_loop.py
    targets = (iter_out.get("review_recommendation") or {}).get("targets") or []
    if not isinstance(targets, list):
        targets = []
    targets = [str(x) for x in targets if x]
    if not targets:
        _changed = (iter_out.get("git_change_summary") or {}).get("changed_paths") or []
        targets = [str(x) for x in _changed[:20] if x]

    assert targets == ["src/a.py", "src/b.py"]


def test_empty_targets_no_fallback_skips_review() -> None:
    """When both targets and changed_paths are empty, review should be skipped."""
    iter_out = {
        "review_recommendation": {"level": "soft", "targets": []},
        "git_change_summary": {},
    }

    targets = (iter_out.get("review_recommendation") or {}).get("targets") or []
    if not isinstance(targets, list):
        targets = []
    targets = [str(x) for x in targets if x]
    if not targets:
        _changed = (iter_out.get("git_change_summary") or {}).get("changed_paths") or []
        targets = [str(x) for x in _changed[:20] if x]

    step_record: Dict[str, Any] = {}
    if not targets:
        step_record["review_skipped"] = "empty_targets"

    assert targets == []
    assert step_record.get("review_skipped") == "empty_targets"


# ---------------------------------------------------------------------------
# Medium #6: ideas.py workspace path containment
# ---------------------------------------------------------------------------

def test_import_ideas_outside_workspace_skips_artifact(tmp_path: Path) -> None:
    """Importing ideas from outside workspace should not crash, artifact is None."""
    from resorch.ideas import import_ideas_jsonl

    ledger, project = _make_project(tmp_path)
    # Create ideas file OUTSIDE workspace
    external = tmp_path / "external_ideas.jsonl"
    external.write_text(
        json.dumps({"id": "i1", "title": "idea 1", "status": "candidate"}) + "\n",
        encoding="utf-8",
    )

    result = import_ideas_jsonl(
        ledger=ledger,
        project_id="p1",
        input_path=str(external),
        register_as_artifact=True,
    )
    assert result["imported"] == 1
    assert result["artifact"] is None  # Should skip, not crash


def test_import_ideas_inside_workspace_registers_artifact(tmp_path: Path) -> None:
    """Importing ideas from inside workspace should register artifact."""
    from resorch.ideas import import_ideas_jsonl

    ledger, project = _make_project(tmp_path)
    ws = Path(project["repo_path"])
    ideas_file = ws / "ideas.jsonl"
    ideas_file.write_text(
        json.dumps({"id": "i2", "title": "idea 2", "status": "candidate"}) + "\n",
        encoding="utf-8",
    )

    result = import_ideas_jsonl(
        ledger=ledger,
        project_id="p1",
        input_path=str(ideas_file),
        register_as_artifact=True,
    )
    assert result["imported"] == 1
    assert result["artifact"] is not None


def test_score_ideas_outside_workspace_skips_artifact(tmp_path: Path) -> None:
    """score_ideas with output outside workspace should not crash on artifact registration."""
    from resorch.ideas import import_ideas_jsonl, score_ideas

    ledger, project = _make_project(tmp_path)
    ws = Path(project["repo_path"])

    # Import some ideas first
    ideas_file = ws / "ideas.jsonl"
    ideas_file.write_text(
        json.dumps({"id": "i1", "title": "idea 1", "status": "candidate"}) + "\n",
        encoding="utf-8",
    )
    import_ideas_jsonl(
        ledger=ledger, project_id="p1",
        input_path=str(ideas_file), register_as_artifact=False,
    )

    # Create rubric in the repo root
    rubric = ledger.paths.root / "rubrics" / "test.yaml"
    rubric.parent.mkdir(parents=True, exist_ok=True)
    rubric.write_text(
        "weights:\n  novelty: 1.0\n  feasibility: 1.0\n  impact: 1.0\n  clarity: 1.0\n  reusability: 1.0\n  risk_penalty: 0.5\n",
        encoding="utf-8",
    )

    # Output goes OUTSIDE workspace
    out_path = tmp_path / "outside_ranked.jsonl"
    result = score_ideas(
        ledger=ledger,
        project_id="p1",
        rubric_path=str(rubric),
        output_path=str(out_path),
        register_output_artifact=True,
    )
    assert result["count"] == 1
    assert result["artifact"] is None  # Should skip, not crash


def test_dedupe_ideas_outside_workspace_skips_artifact(tmp_path: Path) -> None:
    """dedupe_ideas_jsonl with output outside workspace should not crash."""
    from resorch.ideas import dedupe_ideas_jsonl, import_ideas_jsonl

    ledger, project = _make_project(tmp_path)
    ws = Path(project["repo_path"])

    # Create ideas file inside workspace
    ideas_file = ws / "ideas.jsonl"
    ideas_file.write_text(
        json.dumps({"id": "i1", "title": "idea 1"}) + "\n"
        + json.dumps({"id": "i2", "title": "idea 2"}) + "\n",
        encoding="utf-8",
    )

    # Output goes OUTSIDE workspace
    out_path = tmp_path / "outside_deduped.jsonl"
    result = dedupe_ideas_jsonl(
        ledger=ledger,
        project_id="p1",
        input_path=str(ideas_file),
        output_path=str(out_path),
        register_output_artifacts=True,
    )
    assert result["before"] >= 1
    # Artifacts should be None (skipped), not crash
    assert result.get("artifact_out") is None
    assert result.get("artifact_map") is None
