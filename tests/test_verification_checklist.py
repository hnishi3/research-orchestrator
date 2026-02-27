from __future__ import annotations

import json
from pathlib import Path

import pytest

from resorch.manuscript_checker import check_manuscript_consistency
from resorch.verification_checklist import format_checklist_markdown, generate_verification_checklist


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _items_by_id(checklist):
    return {item.id: item for item in checklist.items}


def test_generate_checklist_minimal_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)

    checklist = generate_verification_checklist(workspace)

    assert len(checklist.items) == 17
    assert checklist.project_id == "ws"


def test_checklist_metric_items(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps(
            {
                "primary_metric": {
                    "name": "accuracy",
                    "direction": "maximize",
                    "current": {"mean": 0.86, "n_runs": 3, "ci_95": [0.84, 0.88]},
                    "baseline": {"mean": 0.8},
                },
                "metrics": {"accuracy": 0.86},
                "runs": [{}, {}, {}],
            }
        )
        + "\n",
    )
    _write(
        workspace / "notes" / "method.md",
        "\n".join(
            [
                "# Method",
                "## Metric Definitions",
                "- Metric: accuracy",
                "- Definition: fraction of correct predictions",
                "- Direction: maximize",
                "- Baseline: 0.8",
            ]
        )
        + "\n",
    )

    checklist = generate_verification_checklist(workspace)
    items = _items_by_id(checklist)

    assert items["metric_baseline_stated"].auto_status == "pass"
    assert items["metric_current_vs_baseline"].auto_status == "pass"
    assert items["metric_reproducible"].auto_status == "pass"
    assert items["metric_defined_operationally"].auto_status == "pass"


def test_checklist_missing_scoreboard(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)

    checklist = generate_verification_checklist(workspace)
    items = _items_by_id(checklist)

    assert items["metric_baseline_stated"].auto_status == "fail"
    assert items["metric_current_vs_baseline"].auto_status == "fail"
    assert items["metric_defined_operationally"].auto_status == "fail"
    assert items["metric_reproducible"].auto_status != "pass"


def test_checklist_code_items(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps(
            {
                "primary_metric": {
                    "name": "accuracy",
                    "direction": "maximize",
                    "current": 0.75,
                    "baseline": 0.7,
                },
                "metrics": {
                    "test_fail_count": 0,
                    "test_pass_count": 24,
                    "test_pass_count_baseline": 20,
                },
            }
        )
        + "\n",
    )
    (workspace / "src" / ".git").mkdir(parents=True, exist_ok=True)

    checklist = generate_verification_checklist(workspace)
    items = _items_by_id(checklist)

    assert items["code_tests_pass"].auto_status == "pass"
    assert items["code_no_regressions"].auto_status == "pass"
    assert items["code_in_version_control"].auto_status == "pass"


def test_checklist_needs_human_items(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "paper" / "manuscript.md",
        "# Paper\n\n## Results\nWe evaluated significance with a t-test.\n",
    )

    checklist = generate_verification_checklist(workspace)
    items = _items_by_id(checklist)

    assert items["stats_tests_appropriate"].auto_status == "needs_human"
    assert items["stats_multiple_comparisons"].auto_status == "needs_human"


def test_format_checklist_has_checkboxes(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps(
            {
                "primary_metric": {
                    "name": "accuracy",
                    "direction": "maximize",
                    "current": 0.75,
                    "baseline": 0.7,
                },
                "metrics": {"test_fail_count": 0, "test_pass_count": 8, "test_pass_count_baseline": 8},
            }
        )
        + "\n",
    )

    checklist = generate_verification_checklist(workspace)
    markdown = format_checklist_markdown(checklist)

    assert "- [x]" in markdown
    assert "- [ ]" in markdown


def test_checklist_coverage_calculation(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)

    checklist = generate_verification_checklist(workspace)
    total = len(checklist.items)
    expected = checklist.auto_pass_count / total

    assert checklist.coverage == pytest.approx(expected)


def test_manuscript_integration(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "paper" / "manuscript.md",
        "\n".join(
            [
                "# Paper",
                "",
                "## References",
                "1. This citation has no DOI.",
            ]
        )
        + "\n",
    )

    consistency = check_manuscript_consistency(workspace)
    consistency_checks = {check.check_id: check for check in consistency.checks}

    checklist = generate_verification_checklist(workspace, include_manuscript_checks=True)
    items = _items_by_id(checklist)

    assert items["manuscript_refs_have_dois"].auto_status == "fail"
    assert items["manuscript_refs_have_dois"].auto_evidence == consistency_checks["refs_have_dois"].message
