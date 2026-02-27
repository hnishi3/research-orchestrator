from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from resorch.interpretation_challenger import challenge_interpretation, maybe_challenge_interpretation_from_workspace


def test_interpretation_challenger_cli_low(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_cli(**_kwargs: Any) -> Dict[str, Any]:
        return {
            "structured_output": {
                "checks": [{"item": "statistical_reliability", "status": "ok", "reason": "ok"}],
                "flags": [],
                "overall_concern_level": "low",
            }
        }

    monkeypatch.setattr("resorch.interpretation_challenger.run_claude_code_print_json", fake_cli)

    out = challenge_interpretation(
        scoreboard_json="{}",
        analysis_digest="",
        problem_md="",
        provider="claude_code_cli",
        model="sonnet",
        workspace_dir=tmp_path,
    )
    assert out.overall_concern_level == "low"
    assert out.flags == []


def test_interpretation_challenger_cli_medium(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_cli(**_kwargs: Any) -> Dict[str, Any]:
        return {
            "structured_output": {
                "checks": [
                    {"item": "statistical_reliability", "status": "needs_review", "reason": "x"},
                    {"item": "baseline_strength", "status": "needs_review", "reason": "y"},
                ],
                "flags": ["statistical_reliability", "baseline_strength"],
                "overall_concern_level": "medium",
            }
        }

    monkeypatch.setattr("resorch.interpretation_challenger.run_claude_code_print_json", fake_cli)

    out = challenge_interpretation(
        scoreboard_json="{}",
        analysis_digest="",
        problem_md="",
        provider="claude_code_cli",
        model="sonnet",
        workspace_dir=tmp_path,
    )
    assert out.overall_concern_level == "medium"
    assert "baseline_strength" in out.flags


def test_interpretation_challenger_cli_high(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_cli(**_kwargs: Any) -> Dict[str, Any]:
        return {
            "structured_output": {
                "checks": [
                    {"item": "statistical_reliability", "status": "needs_review", "reason": "x"},
                    {"item": "baseline_strength", "status": "needs_review", "reason": "y"},
                    {"item": "alternative_explanation", "status": "needs_review", "reason": "z"},
                ],
                "flags": ["statistical_reliability", "baseline_strength", "alternative_explanation"],
                "overall_concern_level": "high",
            }
        }

    monkeypatch.setattr("resorch.interpretation_challenger.run_claude_code_print_json", fake_cli)

    out = challenge_interpretation(
        scoreboard_json="{}",
        analysis_digest="",
        problem_md="",
        provider="claude_code_cli",
        model="sonnet",
        workspace_dir=tmp_path,
    )
    assert out.overall_concern_level == "high"
    assert len(out.flags) == 3


def test_interpretation_challenger_anthropic_low(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "resorch.interpretation_challenger._call_anthropic",
        lambda **_kwargs: {
            "checks": [{"item": "statistical_reliability", "status": "ok", "reason": "ok"}],
            "flags": [],
            "overall_concern_level": "low",
        },
    )

    out = challenge_interpretation(
        scoreboard_json="{}",
        analysis_digest="",
        problem_md="",
        provider="anthropic",
        model="claude-sonnet-4-5",
        workspace_dir=tmp_path,
    )
    assert out.overall_concern_level == "low"


def test_interpretation_challenger_codex_low(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_codex(**_kwargs: Any) -> Dict[str, Any]:
        return {
            "structured_output": {
                "checks": [{"item": "statistical_reliability", "status": "ok", "reason": "ok"}],
                "flags": [],
                "overall_concern_level": "low",
            }
        }

    monkeypatch.setattr("resorch.interpretation_challenger.run_codex_exec_print_json", fake_codex)

    out = challenge_interpretation(
        scoreboard_json="{}",
        analysis_digest="",
        problem_md="",
        provider="codex_cli",
        model="gpt-5.3-codex",
        workspace_dir=tmp_path,
    )
    assert out.overall_concern_level == "low"
    assert out.flags == []


def test_interpretation_challenger_skips_if_no_scoreboard(tmp_path: Path) -> None:
    out = maybe_challenge_interpretation_from_workspace(workspace_dir=tmp_path, provider="claude_code_cli", model="sonnet")
    assert out is None


def test_interpretation_challenger_fail_open_on_cli_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_cli(**_kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr("resorch.interpretation_challenger.run_claude_code_print_json", fake_cli)

    out = challenge_interpretation(
        scoreboard_json="{}",
        analysis_digest="",
        problem_md="",
        provider="claude_code_cli",
        model="sonnet",
        workspace_dir=tmp_path,
    )
    assert out.overall_concern_level == "low"
