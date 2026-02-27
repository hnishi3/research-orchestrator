from __future__ import annotations

import json
from pathlib import Path

from resorch.cli import build_parser, main
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project


def _make_tmp_repo(tmp_path: Path) -> tuple[Ledger, Path]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger, repo_root


def test_build_parser_verify_subcommands() -> None:
    args = build_parser().parse_args(["verify", "checklist", "--project", "p1"])
    assert args._handler == "verify_checklist"
    args = build_parser().parse_args(["verify", "consistency", "--project", "p1"])
    assert args._handler == "verify_consistency"
    args = build_parser().parse_args(["verify", "submission", "--project", "p1"])
    assert args._handler == "verify_submission"


def test_cli_verify_checklist_writes_markdown(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    ledger, repo_root = _make_tmp_repo(tmp_path)
    create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="analysis",
        git_init=False,
    )
    ledger.close()

    code = main(["--repo-root", str(repo_root), "verify", "checklist", "--project", "p1"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Auto-verified" in out

    checklist_path = repo_root / "workspaces" / "p1" / "reviews" / "verification_checklist.md"
    assert checklist_path.exists()


def test_cli_verify_consistency_writes_report(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    ledger, repo_root = _make_tmp_repo(tmp_path)
    create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="analysis",
        git_init=False,
    )
    ledger.close()

    code = main(["--repo-root", str(repo_root), "verify", "consistency", "--project", "p1"])
    assert code == 0
    out = capsys.readouterr().out
    assert "checks passed" in out

    report_path = repo_root / "workspaces" / "p1" / "results" / "manuscript_consistency_report.md"
    assert report_path.exists()


def test_cli_verify_submission_writes_report_json_and_bundle(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    ledger, repo_root = _make_tmp_repo(tmp_path)
    create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="analysis",
        git_init=False,
    )
    ledger.close()

    code = main(["--repo-root", str(repo_root), "verify", "submission", "--project", "p1"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["verdict"] in {"pass", "fail", "needs_human"}

    ws = repo_root / "workspaces" / "p1"
    assert (ws / "results" / "submission_verification_report.md").exists()
    assert (ws / "results" / "submission_verification.json").exists()
    assert (ws / "paper" / "submission_bundle.zip").exists()


def test_cli_verify_submission_empty_project_does_not_crash(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    ledger, repo_root = _make_tmp_repo(tmp_path)
    workspace = repo_root / "workspaces" / "p_empty"
    workspace.mkdir(parents=True, exist_ok=True)
    ledger.insert_project(
        project_id="p_empty",
        title="Empty Project",
        domain="",
        stage="analysis",
        repo_path=str(workspace),
        meta={},
    )
    ledger.close()

    code = main(["--repo-root", str(repo_root), "verify", "submission", "--project", "p_empty"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["verdict"] in {"fail", "needs_human"}

    assert (workspace / "results" / "submission_verification_report.md").exists()
    assert (workspace / "results" / "submission_verification.json").exists()


def test_cli_verify_submission_minimal_scoreboard_project(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    ledger, repo_root = _make_tmp_repo(tmp_path)
    workspace = repo_root / "workspaces" / "p_min"
    (workspace / "results").mkdir(parents=True, exist_ok=True)
    (workspace / "results" / "scoreboard.json").write_text(
        json.dumps(
            {
                "primary_metric": {
                    "name": "accuracy",
                    "direction": "maximize",
                    "current": 0.9,
                    "baseline": 0.8,
                },
                "metrics": {"test_pass_count": 281, "test_fail_count": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ledger.insert_project(
        project_id="p_min",
        title="Minimal Project",
        domain="",
        stage="analysis",
        repo_path=str(workspace),
        meta={},
    )
    ledger.close()

    code = main(["--repo-root", str(repo_root), "verify", "submission", "--project", "p_min"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert payload["verdict"] in {"fail", "needs_human"}
    assert any(c["name"] == "scoreboard_required_fields" for c in payload["checks"])
    assert (workspace / "results" / "submission_verification_report.md").exists()
    assert (workspace / "results" / "submission_verification.json").exists()
