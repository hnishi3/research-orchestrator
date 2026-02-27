"""Tests for pre-exec review gate injection into codex_exec prompts."""
from __future__ import annotations

import pytest

from resorch.autopilot import (
    PRE_EXEC_REVIEW_INSTRUCTIONS,
    PRE_EXEC_REVIEW_INSTRUCTIONS_CODEX,
    _maybe_inject_pre_exec_review,
)


class TestMaybeInjectPreExecReview:
    """Unit tests for _maybe_inject_pre_exec_review()."""

    def test_injects_into_codex_exec_when_enabled(self) -> None:
        policy = {"pre_exec_review": {"enabled": True, "model": "haiku"}}
        spec = {"prompt": "Do the analysis"}
        result = _maybe_inject_pre_exec_review(spec=spec, task_type="codex_exec", policy=policy)
        assert "PRE-EXEC REVIEW GATE" in result["prompt"]
        assert "Do the analysis" in result["prompt"]
        assert "--model haiku" in result["prompt"]
        # Original spec must not be mutated.
        assert "PRE-EXEC REVIEW GATE" not in spec["prompt"]

    def test_noop_when_disabled(self) -> None:
        policy = {"pre_exec_review": {"enabled": False, "model": "haiku"}}
        spec = {"prompt": "Do the analysis"}
        result = _maybe_inject_pre_exec_review(spec=spec, task_type="codex_exec", policy=policy)
        assert result is spec

    def test_noop_when_missing_config(self) -> None:
        policy: dict = {}
        spec = {"prompt": "Do the analysis"}
        result = _maybe_inject_pre_exec_review(spec=spec, task_type="codex_exec", policy=policy)
        assert result is spec

    def test_noop_for_shell_exec(self) -> None:
        policy = {"pre_exec_review": {"enabled": True, "model": "haiku"}}
        spec = {"command": "echo ok"}
        result = _maybe_inject_pre_exec_review(spec=spec, task_type="shell_exec", policy=policy)
        assert result is spec

    def test_noop_when_no_prompt(self) -> None:
        policy = {"pre_exec_review": {"enabled": True, "model": "haiku"}}
        spec = {"prompt_file": "some/file.md"}
        result = _maybe_inject_pre_exec_review(spec=spec, task_type="codex_exec", policy=policy)
        assert result is spec

    def test_model_placeholder_filled(self) -> None:
        policy = {"pre_exec_review": {"enabled": True, "model": "sonnet"}}
        spec = {"prompt": "Analyze data"}
        result = _maybe_inject_pre_exec_review(spec=spec, task_type="codex_exec", policy=policy)
        assert "--model sonnet" in result["prompt"]
        assert "{model}" not in result["prompt"]

    def test_default_model_is_haiku(self) -> None:
        policy = {"pre_exec_review": {"enabled": True, "provider": "claude_code_cli"}}
        spec = {"prompt": "Analyze data"}
        result = _maybe_inject_pre_exec_review(spec=spec, task_type="codex_exec", policy=policy)
        assert "--model haiku" in result["prompt"]

    def test_noop_when_cfg_is_none(self) -> None:
        policy = {"pre_exec_review": None}
        spec = {"prompt": "Analyze data"}
        result = _maybe_inject_pre_exec_review(spec=spec, task_type="codex_exec", policy=policy)
        assert result is spec

    def test_noop_when_cfg_is_not_dict(self) -> None:
        policy = {"pre_exec_review": "invalid"}
        spec = {"prompt": "Analyze data"}
        result = _maybe_inject_pre_exec_review(spec=spec, task_type="codex_exec", policy=policy)
        assert result is spec

    def test_injects_codex_when_provider_codex(self) -> None:
        policy = {"pre_exec_review": {"enabled": True, "provider": "codex_cli", "model": "gpt-5.3-codex"}}
        spec = {"prompt": "Run evaluation"}
        result = _maybe_inject_pre_exec_review(spec=spec, task_type="codex_exec", policy=policy)
        assert "codex exec" in result["prompt"]
        assert "--model gpt-5.3-codex" in result["prompt"]
        assert "claude --print" not in result["prompt"]

    def test_codex_default_model(self) -> None:
        policy = {"pre_exec_review": {"enabled": True, "provider": "codex_cli"}}
        spec = {"prompt": "Run evaluation"}
        result = _maybe_inject_pre_exec_review(spec=spec, task_type="codex_exec", policy=policy)
        assert "--model gpt-5.3-codex" in result["prompt"]

    def test_codex_alias_model_is_normalized(self) -> None:
        policy = {"pre_exec_review": {"enabled": True, "provider": "codex_cli", "model": "sonnet"}}
        spec = {"prompt": "Run evaluation"}
        result = _maybe_inject_pre_exec_review(spec=spec, task_type="codex_exec", policy=policy)
        assert "--model gpt-5.3-codex" in result["prompt"]
        assert "--model sonnet" not in result["prompt"]

    def test_backward_compat_old_key(self) -> None:
        """Old configs using 'science_review' key should still work via fallback."""
        policy = {"science_review": {"enabled": True, "model": "haiku"}}
        spec = {"prompt": "Do the analysis"}
        result = _maybe_inject_pre_exec_review(spec=spec, task_type="codex_exec", policy=policy)
        assert "PRE-EXEC REVIEW GATE" in result["prompt"]
        assert "--model haiku" in result["prompt"]


class TestPreExecReviewInstructionsConstant:
    """Verify the PRE_EXEC_REVIEW_INSTRUCTIONS constant has expected content."""

    def test_contains_checklist_items(self) -> None:
        lower = PRE_EXEC_REVIEW_INSTRUCTIONS.lower()
        assert "label direction" in lower
        assert "train/test leakage" in lower
        assert "baseline fairness" in lower
        assert "statistical test" in lower
        assert "misleadingly positive" in lower

    def test_contains_pass_fail(self) -> None:
        assert "PASS" in PRE_EXEC_REVIEW_INSTRUCTIONS
        assert "FAIL" in PRE_EXEC_REVIEW_INSTRUCTIONS

    def test_contains_model_placeholder(self) -> None:
        assert "{model}" in PRE_EXEC_REVIEW_INSTRUCTIONS

    def test_contains_gate_markers(self) -> None:
        assert "--- PRE-EXEC REVIEW GATE ---" in PRE_EXEC_REVIEW_INSTRUCTIONS
        assert "--- END PRE-EXEC REVIEW GATE ---" in PRE_EXEC_REVIEW_INSTRUCTIONS

    def test_contains_claude_print_command(self) -> None:
        assert "claude --print" in PRE_EXEC_REVIEW_INSTRUCTIONS

    def test_format_with_model_succeeds(self) -> None:
        result = PRE_EXEC_REVIEW_INSTRUCTIONS.format(model="opus")
        assert "--model opus" in result
        assert "{model}" not in result

    def test_codex_constant_contains_codex_command(self) -> None:
        assert "codex exec" in PRE_EXEC_REVIEW_INSTRUCTIONS_CODEX

    def test_codex_format_with_model_flag_succeeds(self) -> None:
        result = PRE_EXEC_REVIEW_INSTRUCTIONS_CODEX.format(model_flag="--model gpt-5.3-codex ")
        assert "--model gpt-5.3-codex" in result
        assert "{model_flag}" not in result
