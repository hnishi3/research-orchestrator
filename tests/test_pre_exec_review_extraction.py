"""Tests for pre-exec review gate extraction from Codex JSONL output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from resorch.codex_runner import extract_pre_exec_review_results
from resorch.autopilot import summarize_pre_exec_reviews


def _write_jsonl(path: Path, events: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _make_command_event(*, command: str, output: str, status: str = "completed") -> Dict[str, Any]:
    return {
        "type": "item.completed",
        "item": {
            "id": "item_test",
            "type": "command_execution",
            "command": command,
            "aggregated_output": output,
            "exit_code": 0,
            "status": status,
        },
    }


class TestExtractPreExecReviewResults:
    def test_pass_result(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "output.jsonl"
        _write_jsonl(jsonl, [
            _make_command_event(
                command='claude --print --no-session-persistence --model sonnet -p "Read the script at /tmp/test.py. Check..."',
                output="PASS: accuracy is a correctly-oriented maximize metric.\n",
            ),
        ])
        results = extract_pre_exec_review_results(jsonl)
        assert len(results) == 1
        assert results[0]["verdict"] == "PASS"
        assert "accuracy" in results[0]["reason"]
        assert results[0]["script"] == "/tmp/test.py"

    def test_fail_result(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "output.jsonl"
        _write_jsonl(jsonl, [
            _make_command_event(
                command='claude --print --model sonnet -p "Read the script at src/eval.py. Check..."',
                output="FAIL: training data leaks into evaluation set.\n",
            ),
        ])
        results = extract_pre_exec_review_results(jsonl)
        assert len(results) == 1
        assert results[0]["verdict"] == "FAIL"
        assert "leaks" in results[0]["reason"]
        assert results[0]["script"] == "src/eval.py"

    def test_codex_result(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "output.jsonl"
        _write_jsonl(jsonl, [
            _make_command_event(
                command=(
                    'codex exec --sandbox read-only --skip-git-repo-check --cd . --model gpt-5.3-codex '
                    '"Read notes/problem.md and notes/method.md for context, then read the script at '
                    'scripts/eval.py. Your ENTIRE response must be a single line starting with PASS or FAIL."'
                ),
                output="PASS: no leakage and metric direction is correct.\n",
            ),
        ])
        results = extract_pre_exec_review_results(jsonl)
        assert len(results) == 1
        assert results[0]["verdict"] == "PASS"
        assert results[0]["script"] == "scripts/eval.py"

    def test_pass_fail_not_first_line(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "output.jsonl"
        _write_jsonl(jsonl, [
            _make_command_event(
                command='claude --print -p "Read the script at x.py. Check..."',
                output="warning: ignored\nFAIL: baseline is not comparable.\n",
            ),
        ])
        results = extract_pre_exec_review_results(jsonl)
        assert len(results) == 1
        assert results[0]["verdict"] == "FAIL"

    def test_multiple_reviews(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "output.jsonl"
        _write_jsonl(jsonl, [
            _make_command_event(
                command='claude --print -p "Read the script at a.py. Check..."',
                output="PASS: all checks pass.\n",
            ),
            _make_command_event(
                command='claude --print -p "Read the script at b.py. Check..."',
                output="FAIL: label direction is wrong.\n",
            ),
        ])
        results = extract_pre_exec_review_results(jsonl)
        assert len(results) == 2
        assert results[0]["verdict"] == "PASS"
        assert results[1]["verdict"] == "FAIL"

    def test_no_pre_exec_review(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "output.jsonl"
        _write_jsonl(jsonl, [
            _make_command_event(
                command="python /tmp/test.py",
                output="Accuracy: 0.800\n",
            ),
        ])
        results = extract_pre_exec_review_results(jsonl)
        assert results == []

    def test_empty_file(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "output.jsonl"
        jsonl.write_text("", encoding="utf-8")
        results = extract_pre_exec_review_results(jsonl)
        assert results == []

    def test_missing_file(self, tmp_path: Path) -> None:
        results = extract_pre_exec_review_results(tmp_path / "nonexistent.jsonl")
        assert results == []

    def test_non_command_events_ignored(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "output.jsonl"
        _write_jsonl(jsonl, [
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "hello"}},
            _make_command_event(
                command='claude --print -p "Read the script at x.py. Check..."',
                output="PASS: ok.\n",
            ),
        ])
        results = extract_pre_exec_review_results(jsonl)
        assert len(results) == 1

    def test_command_truncated_at_500(self, tmp_path: Path) -> None:
        long_cmd = "claude --print -p " + "x" * 600
        jsonl = tmp_path / "output.jsonl"
        _write_jsonl(jsonl, [
            _make_command_event(command=long_cmd, output="PASS: ok.\n"),
        ])
        results = extract_pre_exec_review_results(jsonl)
        assert len(results) == 1
        assert len(results[0]["command"]) == 500


class TestSummarizePreExecReviews:
    def test_empty_tasks(self) -> None:
        summary = summarize_pre_exec_reviews([])
        assert summary == {"total": 0, "pass": 0, "fail": 0, "reviews": []}

    def test_tasks_without_reviews(self) -> None:
        tasks = [{"id": "t1", "type": "codex_exec", "status": "success"}]
        summary = summarize_pre_exec_reviews(tasks)
        assert summary["total"] == 0

    def test_aggregation(self) -> None:
        tasks = [
            {
                "id": "t1",
                "pre_exec_review_results": [
                    {"script": "a.py", "verdict": "PASS", "reason": "ok", "command": "..."},
                ],
            },
            {
                "id": "t2",
                "pre_exec_review_results": [
                    {"script": "b.py", "verdict": "FAIL", "reason": "bad", "command": "..."},
                    {"script": "c.py", "verdict": "PASS", "reason": "ok2", "command": "..."},
                ],
            },
        ]
        summary = summarize_pre_exec_reviews(tasks)
        assert summary["total"] == 3
        assert summary["pass"] == 2
        assert summary["fail"] == 1
        assert len(summary["reviews"]) == 3
