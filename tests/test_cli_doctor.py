from __future__ import annotations

import json
import sys
from pathlib import Path

from resorch.cli import build_parser, main
from resorch.doctor import run_doctor
from resorch.ledger import Ledger
from resorch.paths import RepoPaths


def _make_tmp_repo(tmp_path: Path) -> tuple[Ledger, Path]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (repo_root / "resorch").mkdir()  # Needed for repo root detection
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger, repo_root


def test_build_parser_doctor() -> None:
    args = build_parser().parse_args(["doctor"])
    assert args._handler == "doctor"


def test_doctor_reports_python_version() -> None:
    result = run_doctor(repo_root=Path("/nonexistent"))
    names = [c["name"] for c in result["checks"]]
    assert "python_version" in names
    py_check = next(c for c in result["checks"] if c["name"] == "python_version")
    if sys.version_info >= (3, 10):
        assert py_check["status"] == "ok"
    else:
        assert py_check["status"] == "error"


def test_doctor_reports_packages() -> None:
    result = run_doctor(repo_root=Path("/nonexistent"))
    names = [c["name"] for c in result["checks"]]
    assert "PyYAML" in names
    assert "jsonschema" in names


def test_doctor_warns_missing_ledger(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    result = run_doctor(repo_root=repo_root)
    ledger_check = next(c for c in result["checks"] if c["name"] == "ledger")
    assert ledger_check["status"] == "warn"
    assert "not initialized" in ledger_check["detail"]


def test_doctor_ok_ledger(tmp_path: Path) -> None:
    ledger, repo_root = _make_tmp_repo(tmp_path)
    ledger.close()
    result = run_doctor(repo_root=repo_root)
    ledger_check = next(c for c in result["checks"] if c["name"] == "ledger")
    assert ledger_check["status"] == "ok"


def test_doctor_warns_missing_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    result = run_doctor(repo_root=repo_root)
    al_check = next(c for c in result["checks"] if c["name"] == "agent_loop.yaml")
    assert al_check["status"] == "warn"


def test_doctor_ok_valid_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "configs").mkdir()
    (repo_root / "configs" / "agent_loop.yaml").write_text("planner:\n  provider: claude_code_cli\n", encoding="utf-8")
    result = run_doctor(repo_root=repo_root)
    al_check = next(c for c in result["checks"] if c["name"] == "agent_loop.yaml")
    assert al_check["status"] == "ok"


def test_doctor_warns_invalid_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "configs").mkdir()
    (repo_root / "configs" / "agent_loop.yaml").write_text("- not a mapping\n", encoding="utf-8")
    result = run_doctor(repo_root=repo_root)
    al_check = next(c for c in result["checks"] if c["name"] == "agent_loop.yaml")
    assert al_check["status"] == "warn"
    assert "expected YAML mapping" in al_check["detail"]


def test_doctor_overall_status_reflects_worst() -> None:
    result = run_doctor(repo_root=Path("/nonexistent"))
    assert result["status"] in {"ok", "warn", "error"}


def test_cli_doctor_integration(tmp_path: Path, capsys) -> None:
    ledger, repo_root = _make_tmp_repo(tmp_path)
    (repo_root / "configs").mkdir()
    (repo_root / "configs" / "agent_loop.yaml").write_text("planner:\n  provider: claude_code_cli\n", encoding="utf-8")
    (repo_root / "configs" / "review_policy.yaml").write_text("policy_version: 1\n", encoding="utf-8")
    ledger.close()
    code = main(["--repo-root", str(repo_root), "doctor"])
    assert code == 0
    out = capsys.readouterr().out
    result = json.loads(out)
    assert result["status"] in {"ok", "warn"}
    assert "python_version" in result
    assert len(result["checks"]) >= 10
