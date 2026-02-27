from __future__ import annotations

import json
from pathlib import Path

import pytest

from resorch.manuscript_checker import (
    check_manuscript_consistency,
    format_consistency_report,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _checks_by_id(report):
    return {check.check_id: check for check in report.checks}


def test_empty_workspace_returns_report(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)

    report = check_manuscript_consistency(workspace)
    checks = _checks_by_id(report)

    assert report.total_checks == 19
    assert report.failed_checks > 0
    assert report.consistency_score == pytest.approx(report.passed_checks / report.total_checks)
    assert checks["fig_ref_exists"].passed is False
    assert checks["scoreboard_exists"].passed is False


def test_scoreboard_checks_pass(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps(
            {
                "primary_metric": {
                    "name": "accuracy",
                    "direction": "maximize",
                    "current": 0.85,
                    "baseline": 0.8,
                },
                "metrics": {"accuracy": 0.85},
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
                "- Primary metric: accuracy",
                "- Definition: fraction correct",
                "- Direction: maximize",
                "- Baseline: 0.8",
            ]
        )
        + "\n",
    )
    _write(workspace / "src" / "model.py", "def predict():\n    return 1\n")

    report = check_manuscript_consistency(workspace)
    checks = _checks_by_id(report)

    assert checks["scoreboard_exists"].passed is True
    assert checks["scoreboard_has_primary_metric"].passed is True
    assert checks["method_describes_metrics"].passed is True
    assert checks["code_in_workspace"].passed is True


def test_figure_ref_detection(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(workspace / "paper" / "manuscript.md", "# Paper\n\nFigure 1 shows the result.\n")
    _write(workspace / "results" / "fig" / "fig1.png", "png")

    report = check_manuscript_consistency(workspace)
    checks = _checks_by_id(report)

    assert checks["fig_ref_exists"].passed is True


def test_figure_ref_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(workspace / "paper" / "manuscript.md", "# Paper\n\nFigure 3 is the key result.\n")
    _write(workspace / "results" / "fig" / "fig1.png", "png")
    _write(workspace / "results" / "fig" / "fig2.png", "png")

    report = check_manuscript_consistency(workspace)
    checks = _checks_by_id(report)

    assert checks["fig_ref_exists"].passed is False


def test_unreferenced_figure_detected(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(workspace / "paper" / "manuscript.md", "# Paper\n\nNo figure is discussed here.\n")
    _write(workspace / "results" / "fig" / "fig1.png", "png")

    report = check_manuscript_consistency(workspace)
    checks = _checks_by_id(report)

    assert checks["fig_file_referenced"].passed is False


def test_doi_detection(tmp_path: Path) -> None:
    pass_ws = tmp_path / "pass"
    _write(
        pass_ws / "paper" / "manuscript.md",
        "\n".join(
            [
                "# Paper",
                "",
                "## References",
                "1. A study. DOI:10.1234/example.1",
            ]
        )
        + "\n",
    )
    pass_report = check_manuscript_consistency(pass_ws)
    assert _checks_by_id(pass_report)["refs_have_dois"].passed is True

    fail_ws = tmp_path / "fail"
    _write(
        fail_ws / "paper" / "manuscript.md",
        "\n".join(
            [
                "# Paper",
                "",
                "## References",
                "1. A study without DOI.",
            ]
        )
        + "\n",
    )
    fail_report = check_manuscript_consistency(fail_ws)
    assert _checks_by_id(fail_report)["refs_have_dois"].passed is False


def test_p_value_without_effect_size(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "paper" / "manuscript.md",
        "# Paper\n\n## Results\nWe observed significance at p < 0.05 in this experiment.\n",
    )

    report = check_manuscript_consistency(workspace)
    check = _checks_by_id(report)["stats_have_effect_sizes"]

    assert check.passed is False
    assert "missing effect sizes" in check.message


def test_placeholder_refs_detected(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "paper" / "manuscript.md",
        "\n".join(
            [
                "# Paper",
                "",
                "## References",
                "1. TODO complete this citation",
                "2. [?]",
            ]
        )
        + "\n",
    )

    report = check_manuscript_consistency(workspace)
    assert _checks_by_id(report)["refs_no_placeholder"].passed is False


def test_consistency_score_calculation(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(workspace / "paper" / "manuscript.md", "# Paper\n\nFigure 2 is referenced.\n")

    report = check_manuscript_consistency(workspace)

    assert report.consistency_score == pytest.approx(report.passed_checks / report.total_checks)


def test_format_consistency_report_produces_markdown(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)

    report = check_manuscript_consistency(workspace)
    markdown = format_consistency_report(report)

    assert "# Manuscript Consistency Report" in markdown
    assert "## Summary" in markdown
    assert "## Issues" in markdown


def test_text_numbers_match_scoreboard(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps(
            {
                "primary_metric": {
                    "name": "accuracy",
                    "direction": "maximize",
                    "current": 0.85,
                    "baseline": 0.8,
                },
                "metrics": {"accuracy": 0.85},
            }
        )
        + "\n",
    )
    _write(
        workspace / "paper" / "manuscript.md",
        "# Paper\n\n## Results\nThe model achieved accuracy of 0.85 on the held-out test set.\n",
    )

    report_ok = check_manuscript_consistency(workspace)
    assert _checks_by_id(report_ok)["text_numbers_match_scoreboard"].passed is True

    _write(
        workspace / "paper" / "manuscript.md",
        "# Paper\n\n## Results\nThe model achieved accuracy of 0.90 on the held-out test set.\n",
    )

    report_bad = check_manuscript_consistency(workspace)
    assert _checks_by_id(report_bad)["text_numbers_match_scoreboard"].passed is False


def test_refs_citations_exist_catches_orphaned(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "paper" / "manuscript.md",
        "\n".join(
            [
                "# Paper",
                "",
                "## Results",
                "Prior work supports this setup [9].",
                "",
                "## References",
                "1. Ref one. DOI:10.1234/ref.1",
                "2. Ref two. DOI:10.1234/ref.2",
            ]
        )
        + "\n",
    )

    report = check_manuscript_consistency(workspace)
    assert _checks_by_id(report)["refs_citations_exist"].passed is False


def test_refs_citations_exist_passes_valid(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "paper" / "manuscript.md",
        "\n".join(
            [
                "# Paper",
                "",
                "## Results",
                "Prior work [1] and replication details [2] support this setup.",
                "",
                "## References",
                "1. Ref one. DOI:10.1234/ref.1",
                "2. Ref two. DOI:10.1234/ref.2",
            ]
        )
        + "\n",
    )

    report = check_manuscript_consistency(workspace)
    assert _checks_by_id(report)["refs_citations_exist"].passed is True


def test_abstract_body_consistency_catches_mismatch(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "paper" / "manuscript.md",
        "\n".join(
            [
                "# Paper",
                "",
                "## Abstract",
                "We improved performance by 40% over baseline.",
                "",
                "## Results",
                "In the main body, performance improved by 25% over baseline.",
            ]
        )
        + "\n",
    )

    report = check_manuscript_consistency(workspace)
    assert _checks_by_id(report)["abstract_body_consistency"].passed is False


def test_abstract_body_consistency_passes_consistent(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "paper" / "manuscript.md",
        "\n".join(
            [
                "# Paper",
                "",
                "## Abstract",
                "We improved performance by 40% over baseline.",
                "",
                "## Results",
                "In the main body, performance improved by 40% over baseline.",
            ]
        )
        + "\n",
    )

    report = check_manuscript_consistency(workspace)
    assert _checks_by_id(report)["abstract_body_consistency"].passed is True


def test_stats_significance_valid_catches_misuse(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "paper" / "manuscript.md",
        "# Paper\n\n## Results\nThe result was statistically significant (p = 0.07, Cohen's d = 0.12).\n",
    )

    report = check_manuscript_consistency(workspace)
    assert _checks_by_id(report)["stats_significance_claim_valid"].passed is False


def test_stats_significance_valid_passes_correct(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "paper" / "manuscript.md",
        "# Paper\n\n## Results\nThe result was statistically significant (p = 0.03, Cohen's d = 0.12).\n",
    )

    report = check_manuscript_consistency(workspace)
    assert _checks_by_id(report)["stats_significance_claim_valid"].passed is True


def test_vacuous_checks_marked_not_applicable(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(workspace / "paper" / "manuscript.md", "")
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps(
            {
                "primary_metric": {"name": "accuracy", "current": 0.8},
                "metrics": {"accuracy": 0.8},
            }
        )
        + "\n",
    )

    report = check_manuscript_consistency(workspace)
    not_applicable = [check for check in report.checks if not check.applicable]

    assert len(not_applicable) >= 5
    assert report.not_applicable_checks == len(not_applicable)
    assert report.applicable_checks + report.not_applicable_checks == report.total_checks


def test_effective_score_excludes_na(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "paper" / "manuscript.md",
        "# Paper\n\n## Results\nThe result was statistically significant (p = 0.07).\n",
    )
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps(
            {
                "primary_metric": {
                    "name": "accuracy",
                    "direction": "maximize",
                    "current": 0.85,
                    "baseline": 0.8,
                },
                "metrics": {"accuracy": 0.85},
            }
        )
        + "\n",
    )

    report = check_manuscript_consistency(workspace)
    applicable_checks = [check for check in report.checks if check.applicable]
    passed_applicable = sum(1 for check in applicable_checks if check.passed)
    expected_effective = passed_applicable / len(applicable_checks) if applicable_checks else 1.0

    assert report.not_applicable_checks > 0
    assert report.failed_checks > 0
    assert report.effective_score == pytest.approx(expected_effective)
    assert report.effective_score == pytest.approx(passed_applicable / report.applicable_checks)
