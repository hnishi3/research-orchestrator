from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

from resorch.agent_loop import run_agent_loop
from resorch.codex_runner import CodexRunResult
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.tasks import list_tasks


def _write_minimal_repo_files(repo_root: Path) -> None:
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "schemas").mkdir(parents=True, exist_ok=True)

    # Keep the plan schema permissive for this integration test.
    (repo_root / "schemas" / "autopilot_plan.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "AutopilotPlan",
                "type": "object",
                "required": [
                    "plan_id",
                    "project_id",
                    "iteration",
                    "objective",
                    "self_confidence",
                    "evidence_strength",
                    "actions",
                    "should_stop",
                ],
                "properties": {
                    "plan_id": {"type": "string"},
                    "project_id": {"type": "string"},
                    "iteration": {"type": "integer"},
                    "objective": {"type": "string"},
                    "self_confidence": {"type": "number"},
                    "evidence_strength": {"type": "number"},
                    "actions": {"type": "array", "items": {"type": "object"}},
                    "should_stop": {"type": "boolean"},
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (repo_root / "configs" / "review_policy.yaml").write_text(
        "\n".join(
            [
                "policy_version: 1",
                "review_phases:",
                "  code_review_gate:",
                "    enabled: true",
                "    provider: claude_code_cli",
                "    kind: code_review",
                "    max_fix_retries: 1",
                "  post_exec:",
                "    dual_on_hard: false",
                "targets:",
                "  default:",
                "    - notes/problem.md",
                "    - notes/method.md",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class _FakeOpenAIClient:
    def __init__(self, *, project_id: str, objective: str):
        self.project_id = project_id
        self.objective = objective
        self.planner_inputs: List[str] = []
        self.review_inputs: List[str] = []
        self._plan_calls = 0
        self._review_calls = 0

    def responses_create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tools = payload.get("tools") or []
        fn_names = [t.get("name") for t in tools if isinstance(t, dict) and t.get("type") == "function"]

        if "submit_plan" in fn_names:
            self._plan_calls += 1
            self.planner_inputs.append(str(payload.get("input") or ""))
            plan = {
                "plan_id": f"plan{self._plan_calls}",
                "project_id": self.project_id,
                "iteration": self._plan_calls - 1,
                "objective": self.objective,
                "self_confidence": 0.5,
                "evidence_strength": 0.5,
                "actions": [
                    {"title": "code", "task_type": "codex_exec", "spec": {"cd": ".", "prompt": "print('hi')"}},
                    {"title": "run", "task_type": "shell_exec", "spec": {"cd": ".", "command": "echo ok", "shell": True}},
                ],
                "should_stop": self._plan_calls >= 2,
                "stop_reason": "done" if self._plan_calls >= 2 else None,
            }
            return {"id": f"resp-plan-{self._plan_calls}", "status": "completed", "output": [{"name": "submit_plan", "arguments": plan}]}

        if "submit_review" in fn_names:
            self._review_calls += 1
            self.review_inputs.append(str(payload.get("input") or ""))
            review = {
                "overall": "ok",
                "recommendation": "minor",
                "findings": [
                    {
                        "severity": "minor",
                        "category": "analysis",
                        "message": "E2E_FINDING_ONE",
                        "target_paths": ["notes/problem.md"],
                    },
                    {
                        "severity": "minor",
                        "category": "reproducibility",
                        "message": "E2E_FINDING_TWO",
                        "target_paths": ["notes/method.md"],
                    },
                ],
            }
            return {
                "id": f"resp-review-{self._review_calls}",
                "status": "completed",
                "output": [{"name": "submit_review", "arguments": review}],
            }

        return {"id": "resp-other", "status": "completed", "output_text": "{}"}

    def responses_get(self, _response_id: str) -> Dict[str, Any]:
        raise AssertionError("responses_get should not be called in this test.")


def test_e2e_agent_loop_two_iterations_includes_previous_review_findings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_minimal_repo_files(repo_root)

    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)

    fake_client = _FakeOpenAIClient(project_id=project["id"], objective="Do something")

    # Planner + post-exec review: mock OpenAI Responses API.
    monkeypatch.setattr("resorch.providers.openai.OpenAIClient.from_env", classmethod(lambda cls: fake_client))

    # Pre-exec code review: mock Claude Code CLI JSON output.
    cc_calls = {"n": 0}

    def fake_claude_code_print_json(**_kwargs: Any) -> Dict[str, Any]:
        cc_calls["n"] += 1
        rec = "accept" if cc_calls["n"] == 1 else "minor"
        return {"structured_output": {"overall": "ok", "recommendation": rec, "findings": []}}

    monkeypatch.setattr("resorch.jobs.run_claude_code_print_json", lambda **kwargs: fake_claude_code_print_json(**kwargs))

    # Ensure pre-exec review has targets even without git.
    monkeypatch.setattr("resorch.autopilot._list_git_changed_paths", lambda _ws: ["src/generated.py"])

    # Shell exec: avoid actually running commands.
    def fake_subprocess_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        stdout = kwargs.get("stdout")
        stderr = kwargs.get("stderr")
        if stdout is not None:
            stdout.write("ok\n")
        if stderr is not None:
            stderr.write("")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("resorch.tasks.subprocess.run", fake_subprocess_run)

    # Codex exec + review_fix: avoid invoking Codex; write a minimal last_message.json.
    def fake_run_codex_exec_jsonl(**kwargs: Any) -> CodexRunResult:
        last_message_path = Path(kwargs["last_message_path"])
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        last_message_path.write_text(json.dumps({"artifacts_created": []}), encoding="utf-8")
        return CodexRunResult(
            returncode=0,
            jsonl_path=Path(kwargs["jsonl_path"]),
            last_message_path=last_message_path,
            stderr_path=Path(kwargs["stderr_path"]),
        )

    monkeypatch.setattr("resorch.tasks.run_codex_exec_jsonl", fake_run_codex_exec_jsonl)

    # Force a soft review on iteration 0 only (git_diff.lines > 200), none on iteration 1.
    git_calls = {"n": 0}

    def fake_git_summary(_workspace: Path) -> Dict[str, Any]:
        git_calls["n"] += 1
        if git_calls["n"] == 1:
            return {"is_git": True, "changed_lines": 201, "changed_files": 1, "changed_paths": ["src/generated.py"]}
        return {"is_git": True, "changed_lines": 0, "changed_files": 0, "changed_paths": []}

    monkeypatch.setattr("resorch.autopilot.compute_git_change_summary", fake_git_summary)

    out = run_agent_loop(
        ledger=ledger,
        project_id=project["id"],
        objective="Do something",
        max_steps=2,
        dry_run=False,
        config_path=None,
    )

    assert len(out["steps"]) == 2
    assert out["steps"][0]["review_job"] is not None

    # Iteration 1 planner prompt should embed the prior review findings.
    assert len(fake_client.planner_inputs) == 2
    assert "E2E_FINDING_ONE" in fake_client.planner_inputs[1]

    # All executed tasks should have succeeded.
    tasks = list_tasks(ledger, project_id=project["id"])
    ran = [t for t in tasks if t.get("type") in {"codex_exec", "shell_exec", "review_fix"}]
    assert ran, "expected codex_exec/shell_exec/review_fix tasks to be created"
    assert all(t.get("status") == "success" for t in ran)
