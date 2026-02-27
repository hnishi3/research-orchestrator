from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from resorch.verifier_loop import run_post_step_verification


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _checks_by_name(result: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    checks = result.get("checks")
    if not isinstance(checks, list):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for chk in checks:
        if not isinstance(chk, dict):
            continue
        name = str(chk.get("name") or "")
        if name:
            out[name] = chk
    return out


def test_full_workspace_runs_and_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    _write(workspace / "notes" / "problem.md", "# Problem\n")
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps({"primary_metric": {"name": "acc", "direction": "maximize", "current": 0.9}}) + "\n",
    )
    _write(workspace / "paper" / "manuscript.md", "# Paper\n")

    fake_report = SimpleNamespace(checks=[SimpleNamespace(check_id="ok", passed=True, message="ok", applicable=True)])
    monkeypatch.setattr("resorch.manuscript_checker.check_manuscript_consistency", lambda _ws: fake_report)

    result = run_post_step_verification(workspace)

    assert result["verdict"] in {"pass", "needs_human"}
    assert "timestamp" in result
    out_json = workspace / "notes" / "autopilot" / "verifier_last.json"
    out_md = workspace / "notes" / "autopilot" / "verifier_last.md"
    assert out_json.exists()
    assert out_md.exists()

    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["verdict"] == result["verdict"]
    md_text = out_md.read_text(encoding="utf-8")
    assert "# Post-Step Verification" in md_text
    assert "## Failed Checks" in md_text
    assert "## Needs Human" in md_text


def test_empty_workspace_does_not_crash(tmp_path: Path) -> None:
    workspace = tmp_path / "empty"
    workspace.mkdir(parents=True, exist_ok=True)

    result = run_post_step_verification(workspace)

    assert result["verdict"] in {"fail", "needs_human"}
    assert isinstance(result.get("checks"), list)
    assert (workspace / "notes" / "autopilot" / "verifier_last.json").exists()
    assert (workspace / "notes" / "autopilot" / "verifier_last.md").exists()


def test_workspace_without_paper_skips_manuscript_check(tmp_path: Path) -> None:
    workspace = tmp_path / "no-paper"
    _write(workspace / "notes" / "problem.md", "# Problem\n")
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps({"primary_metric": {"name": "m", "direction": "maximize", "current": 1.0}}) + "\n",
    )

    result = run_post_step_verification(workspace)
    checks = _checks_by_name(result)

    manuscript_check = checks.get("manuscript_consistency")
    assert manuscript_check is not None
    assert manuscript_check["status"] == "pass"
    assert "skipped" in manuscript_check["detail"]


def test_broken_scoreboard_fails_schema_check(tmp_path: Path) -> None:
    workspace = tmp_path / "broken-scoreboard"
    _write(workspace / "notes" / "problem.md", "# Problem\n")
    _write(workspace / "results" / "scoreboard.json", "{ invalid\n")

    result = run_post_step_verification(workspace)
    checks = _checks_by_name(result)

    assert result["verdict"] == "fail"
    assert checks["scoreboard_primary_metric"]["status"] == "fail"


def test_missing_manuscript_checker_is_graceful(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "missing-checker"
    _write(workspace / "notes" / "problem.md", "# Problem\n")
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps({"primary_metric": {"name": "acc", "direction": "maximize", "current": 0.5}}) + "\n",
    )
    _write(workspace / "paper" / "manuscript.md", "# Paper\n")

    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any):
        if name == "resorch.manuscript_checker":
            raise ImportError("mock missing manuscript_checker")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = run_post_step_verification(workspace)
    checks = _checks_by_name(result)

    assert result["verdict"] in {"needs_human", "fail"}
    assert checks["manuscript_consistency"]["status"] == "needs_human"


def test_checklist_fail_items_are_collected_when_context_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "checklist"
    _write(workspace / "notes" / "problem.md", "# Problem\n")
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps({"primary_metric": {"name": "m", "direction": "maximize", "current": 0.1}}) + "\n",
    )

    fake_items: List[Any] = [
        SimpleNamespace(id="a", auto_status="fail", auto_evidence="fix me", question="q1"),
        SimpleNamespace(id="b", auto_status="needs_human", auto_evidence="human check", question="q2"),
        SimpleNamespace(id="c", auto_status="pass", auto_evidence="ok", question="q3"),
    ]
    fake_checklist = SimpleNamespace(items=fake_items)
    call_kwargs: Dict[str, Any] = {}

    def _fake_generate(
        workspace_dir: Path,
        project_id: str = "",
        include_manuscript_checks: bool = True,
        lightweight: bool = False,
    ) -> Any:
        kwargs = {
            "workspace_dir": workspace_dir,
            "project_id": project_id,
            "include_manuscript_checks": include_manuscript_checks,
            "lightweight": lightweight,
        }
        call_kwargs.update(kwargs)
        return fake_checklist

    monkeypatch.setattr(
        "resorch.verification_checklist.generate_verification_checklist",
        _fake_generate,
    )

    result = run_post_step_verification(workspace, ledger=object(), project_id="p1")

    assert result["verdict"] == "fail"
    assert call_kwargs.get("lightweight") is True
    assert call_kwargs.get("include_manuscript_checks") is True
    assert any("verification_checklist/a" in item for item in result["fail_items"])
    assert any("verification_checklist/b" in item for item in result["needs_human_items"])
