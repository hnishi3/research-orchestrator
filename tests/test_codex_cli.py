from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

import resorch.jobs as jobs_mod
from resorch.autopilot import run_autopilot_iteration
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.providers.codex_cli import CodexCliConfig, run_codex_exec_print_json


def _make_tmp_repo(tmp_path: Path) -> Tuple[Ledger, Dict[str, Any]]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "review").mkdir(parents=True, exist_ok=True)
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
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="analysis",
        git_init=False,
    )
    return ledger, project


def test_codex_cli_runner_reads_structured_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()

    captured: Dict[str, Any] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        schema_idx = cmd.index("--output-schema") + 1
        captured["schema_obj"] = json.loads(Path(cmd[schema_idx]).read_text(encoding="utf-8"))
        out_idx = cmd.index("--output-last-message") + 1
        Path(cmd[out_idx]).write_text('{"ok": true}', encoding="utf-8")
        stdout = "\n".join(
            [
                '{"type":"thread.started"}',
                '{"type":"turn.completed","usage":{"total_tokens":42}}',
            ]
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("resorch.providers.codex_cli.subprocess.run", fake_run)

    out = run_codex_exec_print_json(
        prompt="Return JSON.",
        json_schema={"type": "object"},
        workspace_dir=ws,
        config=CodexCliConfig(model="gpt-5.2"),
    )

    assert out["structured_output"]["ok"] is True
    assert out["usage"]["total_tokens"] == 42
    assert "--output-schema" in captured["cmd"]
    assert "--json" in captured["cmd"]
    assert "--ephemeral" in captured["cmd"]
    assert captured["schema_obj"]["additionalProperties"] is False
    assert captured["schema_obj"]["required"] == []


def test_codex_cli_runner_raises_on_turn_failed_event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        out_idx = cmd.index("--output-last-message") + 1
        Path(cmd[out_idx]).write_text("", encoding="utf-8")
        stdout = "\n".join(
            [
                '{"type":"thread.started"}',
                '{"type":"turn.failed","error":{"message":"schema invalid"}}',
            ]
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("resorch.providers.codex_cli.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="turn failure"):
        run_codex_exec_print_json(
            prompt="Return JSON.",
            json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
            workspace_dir=ws,
            config=CodexCliConfig(model="gpt-5.2"),
        )


def test_codex_cli_schema_normalizes_required_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    captured: Dict[str, Any] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        schema_idx = cmd.index("--output-schema") + 1
        captured["schema_obj"] = json.loads(Path(cmd[schema_idx]).read_text(encoding="utf-8"))
        out_idx = cmd.index("--output-last-message") + 1
        Path(cmd[out_idx]).write_text('{"ok": true}', encoding="utf-8")
        stdout = '{"type":"turn.completed","usage":{"total_tokens":1}}'
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("resorch.providers.codex_cli.subprocess.run", fake_run)

    run_codex_exec_print_json(
        prompt="Return JSON.",
        json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        workspace_dir=ws,
        config=CodexCliConfig(model="gpt-5.2"),
    )

    assert captured["schema_obj"]["required"] == ["ok"]


def test_codex_cli_review_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger, project = _make_tmp_repo(tmp_path)

    def fake_codex(*, prompt, json_schema, workspace_dir, config):  # noqa: ANN001
        assert "Review request JSON" in prompt
        assert json_schema["type"] == "object"
        assert workspace_dir == Path(project["repo_path"]).resolve()
        return {
            "structured_output": {
                "recommendation": "minor",
                "findings": [
                    {
                        "severity": "minor",
                        "category": "writing",
                        "message": "Looks fine with minor notes.",
                        "target_paths": ["notes/problem.md"],
                    }
                ],
            },
            "usage": {"total_tokens": 123},
            "events": [],
        }

    monkeypatch.setattr(jobs_mod, "run_codex_exec_print_json", fake_codex)

    job = jobs_mod.create_job(
        ledger=ledger,
        project_id=project["id"],
        provider="codex_cli",
        kind="review",
        spec={
            "stage": "analysis",
            "targets": ["notes/problem.md"],
            "questions": ["Is the question clear?"],
            "reviewer": "codex",
            "fallback_provider": "none",
        },
    )

    out = jobs_mod.run_job(ledger=ledger, job_id=job["id"])
    assert out["status"] == "succeeded"
    assert out["result"]["review_result"]["recommendation"] == "minor"
    assert out["result"]["ingested"]["tasks_created"] == 0


def test_autopilot_uses_codex_planner_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger, project = _make_tmp_repo(tmp_path)
    called = {"codex": 0}

    def fake_generate_plan_codex(**_kwargs: Any):
        called["codex"] += 1
        return (
            {
                "plan_id": "plan-codex",
                "project_id": project["id"],
                "iteration": 0,
                "objective": "obj",
                "self_confidence": 0.5,
                "evidence_strength": 0.5,
                "actions": [],
                "should_stop": False,
            },
            {"provider": "codex_cli"},
        )

    def _unexpected(**_kwargs: Any):
        raise AssertionError("OpenAI/Claude planner should not be called")

    monkeypatch.setattr("resorch.autopilot.generate_plan_codex", fake_generate_plan_codex)
    monkeypatch.setattr("resorch.autopilot.generate_plan_openai", _unexpected)
    monkeypatch.setattr("resorch.autopilot.generate_plan_claude", _unexpected)
    monkeypatch.setattr("resorch.autopilot.load_review_policy", lambda _root, **kw: {"review_phases": {}})

    out = run_autopilot_iteration(
        ledger=ledger,
        project_id=project["id"],
        objective="obj",
        model="gpt-5.2",
        iteration=0,
        dry_run=True,
        max_actions=2,
        background=False,
        config={"planner_provider": "codex_cli"},
    )

    assert called["codex"] == 1
    assert out["planner_meta"]["provider"] == "codex_cli"
