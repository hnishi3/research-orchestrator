from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import resorch.jobs as jobs_mod
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.providers.claude_code_cli import run_claude_code_print_json


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (repo_root / "review").mkdir()
    (repo_root / "review" / "review_result.schema.json").write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"recommendation": {"type": "string"}, "findings": {"type": "array"}},
                "required": ["recommendation", "findings"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_claude_code_cli_runner_sanitizes_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "should_be_removed")

    captured = {"cmd": None, "env": None}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        out = {"type": "result", "structured_output": {"recommendation": "minor", "findings": []}}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(out), stderr="")

    monkeypatch.setattr("resorch.providers.claude_code_cli.subprocess.run", fake_run)

    got = run_claude_code_print_json(
        prompt="Return JSON.",
        system_prompt="You are a reviewer.",
        json_schema={"type": "object"},
        workspace_dir=ws,
    )
    assert got["type"] == "result"
    assert captured["env"] is not None
    assert "ANTHROPIC_API_KEY" not in captured["env"]


def test_claude_code_cli_review_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="pcc",
        title="PCC",
        domain="",
        stage="intake",
        git_init=False,
    )

    def fake_claude(*, prompt, system_prompt, json_schema, workspace_dir, config):  # noqa: ANN001
        assert "Review request JSON" in prompt
        assert system_prompt
        assert json_schema["type"] == "object"
        assert workspace_dir == Path(project["repo_path"]).resolve()
        out = {
            "type": "result",
            "structured_output": {
                "recommendation": "minor",
                "findings": [
                    {
                        "severity": "minor",
                        "category": "writing",
                        "message": "Tighten the research question.",
                        "target_paths": ["notes/problem.md"],
                    }
                ],
            },
        }
        return out

    monkeypatch.setattr(jobs_mod, "run_claude_code_print_json", fake_claude)

    job = jobs_mod.create_job(
        ledger=ledger,
        project_id=project["id"],
        provider="claude_code_cli",
        kind="review",
        spec={
            "stage": "intake",
            "targets": ["notes/problem.md"],
            "questions": ["Is the question clear?"],
            "reviewer": "claude_code",
            "fallback_provider": "none",
        },
    )

    out = jobs_mod.run_job(ledger=ledger, job_id=job["id"])
    assert out["status"] == "succeeded"
    # kind="review" (research review) → create_fix_tasks="none", so no tasks created.
    assert out["result"]["ingested"]["tasks_created"] == 0

    tasks = ledger.list_tasks(project_id=project["id"])
    assert len(tasks) == 0


def test_claude_code_cli_review_job_fallback_to_openai(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="pccfb",
        title="PCCFB",
        domain="",
        stage="intake",
        git_init=False,
    )

    monkeypatch.setenv("OPENAI_API_KEY", "ok")

    def fake_claude(*args, **kwargs):  # noqa: ANN001
        raise jobs_mod.ClaudeCodeCliError("boom")

    monkeypatch.setattr(jobs_mod, "run_claude_code_print_json", fake_claude)

    class DummyOpenAI:
        def responses_create(self, payload: dict) -> dict:
            return {
                "id": "resp_review",
                "status": "completed",
                "output": [
                    {
                        "type": "tool_call",
                        "name": "submit_review",
                        "arguments": {
                            "recommendation": "minor",
                            "findings": [
                                {
                                    "severity": "minor",
                                    "category": "writing",
                                    "message": "Fallback reviewer ran.",
                                    "target_paths": ["notes/problem.md"],
                                }
                            ],
                        },
                    }
                ],
            }

        def responses_get(self, response_id: str) -> dict:
            raise AssertionError("responses_get should not be called for completed responses")

    def fake_from_env(cls):  # noqa: ANN001
        return DummyOpenAI()

    monkeypatch.setattr(jobs_mod.OpenAIClient, "from_env", classmethod(fake_from_env))

    job = jobs_mod.create_job(
        ledger=ledger,
        project_id=project["id"],
        provider="claude_code_cli",
        kind="review",
        spec={
            "stage": "intake",
            "targets": ["notes/problem.md"],
            "questions": ["Is the question clear?"],
            "reviewer": "claude_code",
            "fallback_provider": "openai",
        },
    )

    out = jobs_mod.run_job(ledger=ledger, job_id=job["id"])
    assert out["status"] == "succeeded"
    assert out["result"]["fallback_provider"] == "openai"

    # kind="review" → no review_fix tasks created for research reviews.
    tasks = ledger.list_tasks(project_id=project["id"])
    assert len(tasks) == 0
