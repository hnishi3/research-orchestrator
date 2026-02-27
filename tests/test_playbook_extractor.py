from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_tmp_repo

from resorch.cli import main
from resorch.playbook_extractor import extract_and_save, extract_playbook_entry
from resorch.projects import create_project, set_project_stage


def test_extract_playbook_entry_from_digest_and_scoreboard(tmp_path: Path) -> None:
    ledger = make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="Protein Baseline Study",
        domain="bio",
        stage="analysis",
        git_init=False,
    )
    workspace = Path(project["repo_path"])

    (workspace / "notes" / "analysis_digest.md").write_text(
        "\n".join(
            [
                "# Analysis Digest",
                "",
                "## Latest",
                "- When training labels are noisy, baseline drifts after 20 epochs.",
                "",
                "## Next actions (top 3)",
                "- Add stratified split by cohort.",
                "- Run leakage check before model fitting.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "results" / "scoreboard.json").write_text(
        json.dumps(
            {
                "primary_metric": {
                    "name": "auroc",
                    "direction": "maximize",
                    "current": 0.79,
                    "baseline": 0.72,
                    "best": 0.81,
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "reviews" / "RESP-analysis-sample.json").write_text(
        json.dumps(
            {
                "project_id": "p1",
                "stage": "analysis",
                "reviewer": "qa",
                "recommendation": "major",
                "findings": [
                    {
                        "severity": "major",
                        "category": "analysis",
                        "message": "Potential data leakage from normalization over full dataset.",
                        "target_paths": ["src/train.py"],
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    entry = extract_playbook_entry(ledger=ledger, project_id="p1", mode="compact")

    assert entry["id"].startswith("pb_p1_")
    assert entry["title"].startswith("Protein Baseline Study - ")
    assert entry["topic"] == "bio"
    assert any("training labels are noisy" in x.lower() for x in entry["when_to_apply"])
    assert any("stratified split" in x.lower() for x in entry["steps"])
    assert any("leakage" in x.lower() for x in entry["anti_patterns"])
    assert entry["evidence"]["project_id"] == "p1"
    assert entry["evidence"]["delta_vs_baseline"] == pytest.approx(0.07)


def test_extract_and_save_creates_yaml_and_ledger_entry(tmp_path: Path) -> None:
    ledger = make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p2",
        title="NLP Cleanup",
        domain="nlp",
        stage="analysis",
        git_init=False,
    )
    workspace = Path(project["repo_path"])

    (workspace / "notes" / "analysis_digest.md").write_text(
        "# Analysis Digest\n\n## Next actions\n- Add tokenizer ablation.\n",
        encoding="utf-8",
    )

    out = extract_and_save(ledger=ledger, project_id="p2", mode="compact")

    assert out["ledger_stored"] is True
    yaml_path = Path(out["yaml_path"])
    assert yaml_path.exists()

    stored = ledger.get_playbook_entry(out["entry_id"])
    assert stored["id"] == out["entry_id"]
    rule = json.loads(stored["rule_json"])
    assert rule["id"] == out["entry_id"]


def test_extract_playbook_entry_handles_missing_files(tmp_path: Path) -> None:
    ledger = make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p3",
        title="Sparse Project",
        domain="",
        stage="analysis",
        git_init=False,
    )
    workspace = Path(project["repo_path"])

    (workspace / "notes" / "analysis_digest.md").unlink(missing_ok=True)
    (workspace / "results" / "scoreboard.json").unlink(missing_ok=True)

    entry = extract_playbook_entry(ledger=ledger, project_id="p3", mode="compact")

    assert entry["id"].startswith("pb_p3_")
    assert entry.get("needs_human") is True
    assert entry["evidence"]["project_id"] == "p3"


def test_set_project_stage_done_triggers_playbook_extract(tmp_path: Path) -> None:
    ledger = make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p4",
        title="Hook Trigger",
        domain="systems",
        stage="analysis",
        git_init=False,
    )
    workspace = Path(project["repo_path"])
    (workspace / "notes" / "analysis_digest.md").write_text(
        "# Analysis Digest\n\n## Next actions\n- Check baseline drift.\n",
        encoding="utf-8",
    )

    updated = set_project_stage(ledger=ledger, project_id="p4", stage="done")

    assert updated["stage"] == "done"
    extracted_dir = workspace / "playbook" / "extracted"
    assert extracted_dir.exists()
    assert list(extracted_dir.glob("*.yaml"))


def test_cli_playbook_extract_outputs_json(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    ledger = make_tmp_repo(tmp_path)
    repo_root = ledger.paths.root
    create_project(
        ledger=ledger,
        project_id="p5",
        title="CLI Project",
        domain="bio",
        stage="analysis",
        git_init=False,
    )
    ledger.close()

    code = main(["--repo-root", str(repo_root), "playbook", "extract", "--project", "p5"])
    assert code == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["entry_id"].startswith("pb_p5_")
    assert payload["ledger_stored"] is True
    assert Path(payload["yaml_path"]).exists()
