from __future__ import annotations

from pathlib import Path

from resorch.artifacts import list_artifacts
from resorch.constraints import write_constraints_template
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.playbook_store import get_playbook_entry, list_playbook_entries, put_playbook_entry
from resorch.projects import create_project


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_playbook_put_get_list(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    entry_path = tmp_path / "entry.yaml"
    entry_path.write_text(
        "\n".join(
            [
                "id: ptn_001",
                "domain: nlp",
                "name: \"Check figures\"",
                "steps:",
                "  - \"Do X\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    put = put_playbook_entry(ledger=ledger, entry_path=str(entry_path))
    assert put["id"] == "ptn_001"

    got = get_playbook_entry(ledger, "ptn_001")
    assert got["rule"]["id"] == "ptn_001"

    listed = list_playbook_entries(ledger, topic="nlp", limit=10)
    assert len(listed) == 1


def test_constraints_template_and_artifact(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )
    out = write_constraints_template(ledger=ledger, project_id=project["id"], path="constraints.yaml", overwrite=False)
    p = Path(out["path"])
    assert p.exists()

    arts = list_artifacts(ledger, project_id=project["id"], prefix="", limit=200)
    assert any(a["kind"] == "constraints_yaml" for a in arts)

