from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from resorch.artifacts import register_artifact
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.retrieval import fetch, search


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_retrieval_search_and_fetch(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="My Demo",
        domain="",
        stage="intake",
        git_init=False,
    )

    ws = Path(project["repo_path"])
    artifact_path = ws / "notes" / "hello.md"
    artifact_path.write_text("hello world\n", encoding="utf-8")

    register_artifact(
        ledger=ledger,
        project=project,
        kind="note",
        relative_path="notes/hello.md",
        meta={},
    )

    ledger.upsert_idea(
        idea_id="idea1",
        project_id=project["id"],
        status="candidate",
        score_total=None,
        data={"id": "idea1", "title": "Needle Idea", "one_sentence_claim": "needle", "status": "candidate"},
    )
    smoke = ledger.insert_smoke_test(
        idea_id="idea1",
        project_id=project["id"],
        verdict="pass",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:05:00Z",
        result={"idea_id": "idea1", "started_at": "2026-01-01T00:00:00Z", "verdict": "pass"},
        artifact_path=None,
    )
    ledger.insert_evidence(
        evidence_id="e1",
        project_id=project["id"],
        idea_id=None,
        kind="paper",
        title="Example Paper",
        url="https://example.com/paper",
        retrieved_at="2026-01-01T00:00:00Z",
        summary="This is a test paper.",
        relevance=0.5,
        meta={},
        artifact_path=None,
    )

    out = search(ledger, query="My Demo", kind="ledger", limit=10)
    assert any(h["id"] == "ledger:projects/p1" for h in out["hits"])

    out2 = search(ledger, query="hello", project_id="p1", kind="artifact", limit=10)
    assert any(h["id"] == "artifact:p1/notes/hello.md" for h in out2["hits"])

    fetched = fetch(ledger, id="artifact:p1/notes/hello.md")
    assert "hello world" in fetched["content"]

    out3 = search(ledger, query="Needle Idea", project_id="p1", kind="ledger", limit=10)
    assert any(h["id"] == "ledger:ideas/idea1" for h in out3["hits"])
    fetched_idea = fetch(ledger, id="ledger:ideas/idea1")
    assert "Needle Idea" in fetched_idea["content"]

    out4 = search(ledger, query="pass", project_id="p1", kind="ledger", limit=10)
    assert any(h["id"] == f"ledger:smoke_tests/{smoke['id']}" for h in out4["hits"])
    fetched_smoke = fetch(ledger, id=f"ledger:smoke_tests/{smoke['id']}")
    assert "\"verdict\": \"pass\"" in fetched_smoke["content"]

    out5 = search(ledger, query="example.com", project_id="p1", kind="ledger", limit=10)
    assert any(h["id"] == "ledger:evidence/e1" for h in out5["hits"])
    fetched_ev = fetch(ledger, id="ledger:evidence/e1")
    assert "Example Paper" in fetched_ev["content"]


def test_stdio_server_search(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p2",
        title="Server Demo",
        domain="",
        stage="intake",
        git_init=False,
    )
    ws = Path(project["repo_path"])
    (ws / "notes" / "x.md").write_text("needle\n", encoding="utf-8")
    register_artifact(ledger=ledger, project=project, kind="note", relative_path="notes/x.md", meta={})

    server_py = Path(__file__).resolve().parents[1] / "mcp_server" / "server.py"
    req = {"jsonrpc": "2.0", "id": 1, "method": "search", "params": {"query": "needle", "project_id": "p2", "kind": "artifact", "limit": 5}}
    proc = subprocess.run(
        [sys.executable, str(server_py), "--repo-root", str(ledger.paths.root)],
        input=json.dumps(req) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    resp = json.loads(proc.stdout.strip().splitlines()[-1])
    assert resp["id"] == 1
    hits = resp["result"]["hits"]
    assert any(h["id"] == "artifact:p2/notes/x.md" for h in hits)


def test_stdio_server_tools_call_roundtrip(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p3",
        title="Tool Demo",
        domain="",
        stage="intake",
        git_init=False,
    )

    # Seed an idea record for Topic Engine tools.
    ledger.upsert_idea(
        idea_id="idea1",
        project_id=project["id"],
        status="candidate",
        score_total=3.0,
        data={
            "id": "idea1",
            "status": "candidate",
            "title": "Demo Idea",
            "one_sentence_claim": "We can measure X.",
            "contribution_type": "analysis",
            "target_venues": ["arXiv"],
            "novelty_statement": "Different from prior work by Y.",
            "evaluation_plan": {"datasets": ["d1"], "metrics": ["m1"], "baselines": ["b1"], "ablations": ["a1"]},
            "feasibility": {"estimated_gpu_hours": 1, "estimated_calendar_days": 1, "blocking_dependencies": []},
            "risks": {"ethics": "low", "license": "low", "safety": "low", "reproducibility": "high"},
            "evidence": [],
        },
    )
    ws = Path(project["repo_path"])
    (ws / "smoke.json").write_text(
        "\n".join(
            [
                "{",
                '  \"idea_id\": \"idea1\",',
                '  \"started_at\": \"2026-01-01T00:00:00Z\",',
                '  \"completed_at\": \"2026-01-01T00:05:00Z\",',
                '  \"verdict\": \"pass\"',
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    server_py = Path(__file__).resolve().parents[1] / "mcp_server" / "server.py"

    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "ledger.list_projects", "arguments": {"limit": 10}}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "artifact.put", "arguments": {"project_id": project["id"], "path": "notes/from_tool.md", "content": "tool content\n"}},
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "artifact.get", "arguments": {"project_id": project["id"], "path": "notes/from_tool.md"}},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "task.create",
                "arguments": {"project_id": project["id"], "type": "shell_exec", "spec": {"command": ["python", "-c", "print('hi')"]}},
            },
        },
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "smoke.ingest", "arguments": {"project_id": project["id"], "result_path": "smoke.json"}}},
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "topic.brief", "arguments": {"project_id": project["id"], "idea_id": "idea1", "set_selected": True}},
        },
        {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "idea.get", "arguments": {"idea_id": "idea1"}}},
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {"name": "evidence.add", "arguments": {"project_id": project["id"], "kind": "paper", "title": "E", "url": "https://example.com", "summary": "S"}},
        },
    ]
    proc = subprocess.run(
        [sys.executable, str(server_py), "--repo-root", str(ledger.paths.root)],
        input="".join(json.dumps(r) + "\n" for r in reqs),
        text=True,
        capture_output=True,
        check=True,
    )

    responses = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert [r["id"] for r in responses] == [1, 2, 3, 4, 5, 6, 7, 8]
    projects = responses[0]["result"]["projects"]
    assert any(p["id"] == "p3" for p in projects)

    fetched = responses[2]["result"]
    assert "tool content" in fetched["content"]

    created_task = responses[3]["result"]["task"]
    assert responses[4]["result"]["smoke_test"]["verdict"] == "pass"
    assert Path(responses[5]["result"]["output_path"]).exists()
    assert responses[6]["result"]["idea"]["status"] == "selected"
    assert responses[7]["result"]["evidence"]["url"] == "https://example.com"

    run_req = {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "task.run", "arguments": {"task_id": created_task["id"]}}}

    # task.run is disabled by default for safety.
    proc2 = subprocess.run(
        [sys.executable, str(server_py), "--repo-root", str(ledger.paths.root)],
        input=json.dumps(run_req) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    resp = json.loads(proc2.stdout.strip().splitlines()[-1])
    assert resp["id"] == 9
    assert "error" in resp
    assert "RESORCH_MCP_ALLOW_TASK_RUN" in resp["error"]["message"]

    proc3 = subprocess.run(
        [sys.executable, str(server_py), "--repo-root", str(ledger.paths.root)],
        input=json.dumps(run_req) + "\n",
        text=True,
        capture_output=True,
        env={**os.environ, "RESORCH_MCP_ALLOW_TASK_RUN": "1"},
        check=True,
    )
    resp_ok = json.loads(proc3.stdout.strip().splitlines()[-1])
    assert resp_ok["id"] == 9
    assert resp_ok["result"]["task"]["status"] == "success"


def test_artifact_fetch_blocks_path_traversal(tmp_path: Path) -> None:
    """Regression test: artifact:project/../../etc/passwd must not escape workspace."""
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="Traversal Test",
        domain="test",
        stage="intake",
        git_init=False,
    )
    with pytest.raises(SystemExit, match="Path traversal blocked"):
        fetch(ledger, id="artifact:p1/../../etc/passwd")
