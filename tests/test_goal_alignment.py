from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from resorch.goal_alignment import check_goal_alignment


def test_goal_alignment_cli_aligned_true(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_cli(**_kwargs: Any) -> Dict[str, Any]:
        return {"structured_output": {"aligned": True, "drift_summary": None}}

    monkeypatch.setattr("resorch.goal_alignment.run_claude_code_print_json", fake_cli)

    out = check_goal_alignment(
        research_question="RQ",
        recent_objectives=["do X"],
        provider="claude_code_cli",
        model="haiku",
        workspace_dir=tmp_path,
    )
    assert out.aligned is True
    assert out.drift_summary is None
    assert out.method == "claude_code_cli"


def test_goal_alignment_cli_aligned_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_cli(**_kwargs: Any) -> Dict[str, Any]:
        return {"structured_output": {"aligned": False, "drift_summary": "drift"}}

    monkeypatch.setattr("resorch.goal_alignment.run_claude_code_print_json", fake_cli)

    out = check_goal_alignment(
        research_question="RQ",
        recent_objectives=["do X"],
        provider="claude_code_cli",
        model="haiku",
        workspace_dir=tmp_path,
    )
    assert out.aligned is False
    assert out.drift_summary == "drift"
    assert out.method == "claude_code_cli"


def test_goal_alignment_anthropic_aligned_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("resorch.goal_alignment._call_anthropic", lambda **_kwargs: {"aligned": True, "drift_summary": None})

    out = check_goal_alignment(
        research_question="RQ",
        recent_objectives=["do X"],
        provider="anthropic",
        model="claude-haiku-4-5",
        workspace_dir=None,
    )
    assert out.aligned is True
    assert out.method == "anthropic"


def test_goal_alignment_codex_aligned_true(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_codex(**_kwargs: Any) -> Dict[str, Any]:
        return {"structured_output": {"aligned": True, "drift_summary": None}}

    monkeypatch.setattr("resorch.goal_alignment.run_codex_exec_print_json", fake_codex)

    out = check_goal_alignment(
        research_question="RQ",
        recent_objectives=["do X"],
        provider="codex_cli",
        model="gpt-5.3-codex",
        workspace_dir=tmp_path,
    )
    assert out.aligned is True
    assert out.method == "codex_cli"


def test_goal_alignment_skips_when_research_question_empty() -> None:
    out = check_goal_alignment(
        research_question="  \n",
        recent_objectives=["do X"],
        provider="claude_code_cli",
        model="haiku",
        workspace_dir=None,
    )
    assert out.aligned is True
    assert out.method == "skipped"


def test_goal_alignment_fail_open_on_cli_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_cli(**_kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr("resorch.goal_alignment.run_claude_code_print_json", fake_cli)

    out = check_goal_alignment(
        research_question="RQ",
        recent_objectives=["do X"],
        provider="claude_code_cli",
        model="haiku",
        workspace_dir=tmp_path,
    )
    assert out.aligned is True
    assert out.method == "skipped"
