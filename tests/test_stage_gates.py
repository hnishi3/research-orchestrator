from __future__ import annotations

from pathlib import Path

from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.stage_gates import compute_gate_env, evaluate_transitions, load_stage_transitions


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (repo_root / "configs").mkdir()
    (repo_root / "configs" / "stage_transitions.yaml").write_text(
        "\n".join(
            [
                "transitions:",
                "  constraints_to_generate:",
                "    auto_pass_if:",
                "      - \"constraints_yaml_exists == true\"",
                "",
                "  generate_to_novelty:",
                "    auto_pass_if:",
                "      - \"idea_count >= 25\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_stage_gate_eval(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )
    ws = Path(project["repo_path"])
    (ws / "constraints.yaml").write_text("meta: {}\n", encoding="utf-8")

    cfg = load_stage_transitions(ledger.paths.root / "configs" / "stage_transitions.yaml")
    env = compute_gate_env(ledger=ledger, project_id=project["id"])
    res = evaluate_transitions(config=cfg, env=env)
    assert res["transitions"]["constraints_to_generate"]["decision"] == "auto_pass"
    assert res["transitions"]["generate_to_novelty"]["decision"] == "pending"

    # Insert enough ideas to auto-pass.
    for i in range(25):
        ledger.upsert_idea(
            idea_id=f"idea{i}",
            project_id=project["id"],
            status="candidate",
            score_total=None,
            data={"id": f"idea{i}", "status": "candidate"},
        )

    env2 = compute_gate_env(ledger=ledger, project_id=project["id"])
    res2 = evaluate_transitions(config=cfg, env=env2)
    assert res2["transitions"]["generate_to_novelty"]["decision"] == "auto_pass"

