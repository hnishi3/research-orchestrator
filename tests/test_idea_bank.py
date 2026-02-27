from __future__ import annotations

from pathlib import Path

from resorch.idea_bank import build_idea_graph, link_ideas, park_idea, revive_idea_to_new_project, spawn_idea
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


def test_idea_spawn_creates_edge_and_child(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )
    ledger.upsert_idea(
        idea_id="idea1",
        project_id=project["id"],
        status="candidate",
        score_total=None,
        data={"id": "idea1", "title": "Idea 1", "evaluation_plan": {"baselines": [], "datasets": [], "metrics": [], "ablations": []}},
    )

    out = spawn_idea(ledger=ledger, parent_idea_id="idea1", operator="baseline_add", baseline_add="BaselineA")
    child = out["idea"]
    assert child["id"] != "idea1"
    assert child["project_id"] == project["id"]
    assert child["data"]["parent_idea_id"] == "idea1"
    assert "BaselineA" in (child["data"].get("evaluation_plan") or {}).get("baselines", [])
    assert out["edge"]["relation"] == "baseline_add"

    graph = build_idea_graph(ledger=ledger, project_id=project["id"])
    assert any(e.get("relation") == "baseline_add" for e in graph["edges"])


def test_idea_park_and_revive_into_new_project(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    p1 = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )
    ledger.upsert_idea(
        idea_id="idea1",
        project_id=p1["id"],
        status="candidate",
        score_total=None,
        data={"id": "idea1", "title": "Idea 1", "status": "candidate"},
    )

    parked = park_idea(ledger=ledger, idea_id="idea1", parked_reason="Need more compute", unblock_conditions=["Get GPU"], next_check_date="2026-03-01")
    assert parked["status"] == "parked"
    assert parked["data"]["parked_reason"] == "Need more compute"
    assert parked["data"]["unblock_conditions"] == ["Get GPU"]

    revived = revive_idea_to_new_project(
        ledger=ledger,
        idea_id="idea1",
        new_project_id="p2",
        new_project_title="P2",
        git_init=False,
        reason="GPU available",
    )
    assert revived["project"]["id"] == "p2"
    assert revived["idea"]["project_id"] == "p2"
    assert revived["idea"]["data"]["parent_idea_id"] == "idea1"
    assert revived["edge"]["relation"] == "revive"


def test_idea_graph_traversal_handles_cycles(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    p1 = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )
    ledger.upsert_idea(idea_id="a", project_id=p1["id"], status="candidate", score_total=None, data={"id": "a", "title": "A"})
    ledger.upsert_idea(idea_id="b", project_id=p1["id"], status="candidate", score_total=None, data={"id": "b", "title": "B"})

    link_ideas(ledger=ledger, src_idea_id="a", dst_idea_id="b", relation="narrow")
    link_ideas(ledger=ledger, src_idea_id="b", dst_idea_id="a", relation="narrow")

    graph = build_idea_graph(ledger=ledger, root_idea_id="a", max_nodes=10, max_edges=10)
    node_ids = {n["id"] for n in graph["nodes"]}
    assert node_ids == {"a", "b"}

