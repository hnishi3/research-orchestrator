from __future__ import annotations

import json
from pathlib import Path

from resorch.autopilot import _pivot_no_improvement_trigger
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project


def _make_tmp_repo(tmp_path: Path, *, pivot_policy_yaml: str) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "pivot_policy.yaml").write_text(pivot_policy_yaml, encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_pivot_no_improvement_triggers_on_ci_overlap(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(
        tmp_path,
        pivot_policy_yaml="\n".join(
            [
                "policy_version: 1",
                "no_improvement:",
                "  enabled: true",
                "  metric_path: primary_metric.current.mean",
                "  direction: maximize",
                "  min_delta: 0.0",
                "  window_runs: 3",
                "  review_level: soft",
                "  use_ci_overlap: true",
                "",
            ]
        ),
    )
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)
    ws = Path(project["repo_path"])

    (ws / "results" / "scoreboard.json").write_text(
        json.dumps(
            {
                "primary_metric": {
                    "name": "acc",
                    "direction": "maximize",
                    "current": {"mean": 0.5, "ci_95": [0.49, 0.51]},
                },
                "runs": [
                    {"primary_metric": {"current": {"mean": 0.5, "ci_95": [0.48, 0.52]}}},
                    {"primary_metric": {"current": {"mean": 0.5, "ci_95": [0.48, 0.52]}}},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = _pivot_no_improvement_trigger(repo_root=ledger.paths.root, workspace=ws)
    assert out is not None
    assert out[0] == "soft"
    assert "pivot_reason=ci_overlap" in out[1]


def test_pivot_no_improvement_not_triggered_when_ci_non_overlap_improves(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(
        tmp_path,
        pivot_policy_yaml="\n".join(
            [
                "policy_version: 1",
                "no_improvement:",
                "  enabled: true",
                "  metric_path: primary_metric.current.mean",
                "  direction: maximize",
                "  min_delta: 1.0",
                "  window_runs: 3",
                "  review_level: soft",
                "  use_ci_overlap: true",
                "",
            ]
        ),
    )
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)
    ws = Path(project["repo_path"])

    (ws / "results" / "scoreboard.json").write_text(
        json.dumps(
            {
                "primary_metric": {
                    "name": "acc",
                    "direction": "maximize",
                    "current": {"mean": 0.51, "ci_95": [0.50, 0.52]},
                },
                "runs": [
                    {"primary_metric": {"current": {"mean": 0.40, "ci_95": [0.39, 0.41]}}},
                    {"primary_metric": {"current": {"mean": 0.41, "ci_95": [0.40, 0.42]}}},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = _pivot_no_improvement_trigger(repo_root=ledger.paths.root, workspace=ws)
    assert out is None


def test_pivot_no_improvement_old_numeric_current_is_supported(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(
        tmp_path,
        pivot_policy_yaml="\n".join(
            [
                "policy_version: 1",
                "no_improvement:",
                "  enabled: true",
                "  metric_path: primary_metric.current.mean",
                "  direction: maximize",
                "  min_delta: 0.000001",
                "  window_runs: 3",
                "  review_level: soft",
                "  use_ci_overlap: true",
                "",
            ]
        ),
    )
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)
    ws = Path(project["repo_path"])

    (ws / "results" / "scoreboard.json").write_text(
        json.dumps(
            {
                "primary_metric": {"name": "acc", "direction": "maximize", "current": 0.5},
                "runs": [
                    {"primary_metric": {"current": 0.5}},
                    {"primary_metric": {"current": 0.5}},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = _pivot_no_improvement_trigger(repo_root=ledger.paths.root, workspace=ws)
    assert out is not None
    assert out[0] == "soft"
    assert "pivot_reason=min_delta_not_met" in out[1]


def test_pivot_policy_use_ci_overlap_false_preserves_min_delta_logic(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(
        tmp_path,
        pivot_policy_yaml="\n".join(
            [
                "policy_version: 1",
                "no_improvement:",
                "  enabled: true",
                "  metric_path: primary_metric.current.mean",
                "  direction: maximize",
                "  min_delta: 1.0",
                "  window_runs: 3",
                "  review_level: soft",
                "  use_ci_overlap: false",
                "",
            ]
        ),
    )
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)
    ws = Path(project["repo_path"])

    # Significant improvement by CI, but min_delta is intentionally too large.
    (ws / "results" / "scoreboard.json").write_text(
        json.dumps(
            {
                "primary_metric": {
                    "name": "acc",
                    "direction": "maximize",
                    "current": {"mean": 0.1, "ci_95": [0.1, 0.1]},
                },
                "runs": [
                    {"primary_metric": {"current": {"mean": 0.0, "ci_95": [0.0, 0.0]}}},
                    {"primary_metric": {"current": {"mean": 0.0, "ci_95": [0.0, 0.0]}}},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = _pivot_no_improvement_trigger(repo_root=ledger.paths.root, workspace=ws)
    assert out is not None
    assert out[0] == "soft"
    assert "pivot_reason=min_delta_not_met" in out[1]
