from __future__ import annotations

from pathlib import Path

from resorch.artifacts import list_artifacts
from resorch.claims import create_claim
from resorch.evidence_store import add_evidence
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_create_claim_with_evidence(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )

    ev = add_evidence(
        ledger=ledger,
        project_id=project["id"],
        kind="paper",
        title="Test Paper",
        url="https://example.com/paper",
        summary="Summary here.",
    )["evidence"]

    out = create_claim(
        ledger=ledger,
        project_id=project["id"],
        statement="This is the claim.",
        evidence_ids=[ev["id"]],
    )
    claim_path = Path(out["path"])
    assert claim_path.exists()
    txt = claim_path.read_text(encoding="utf-8")
    assert ev["id"] in txt
    assert "https://example.com/paper" in txt

    arts = list_artifacts(ledger, project_id=project["id"], prefix="claims/", limit=50)
    assert any(a["kind"] == "claim_md" for a in arts)

