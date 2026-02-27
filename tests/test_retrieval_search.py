from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_tmp_repo
from resorch.ledger import Ledger
from resorch.projects import create_project
from resorch.retrieval import search


def _fts_tables_exist(ledger: Ledger) -> bool:
    row = ledger.conn().execute(
        "SELECT 1 AS ok FROM sqlite_master WHERE type='table' AND name='fts_projects'"
    ).fetchone()
    return bool(row)


def test_fts5_migration_runs_on_fresh_db(tmp_path: Path) -> None:
    ledger = make_tmp_repo(tmp_path)
    assert ledger.get_meta("schema_version") == "7"

    conn = ledger.conn()
    fts_tables = {
        "fts_projects",
        "fts_tasks",
        "fts_ideas",
        "fts_evidence",
        "fts_reviews",
        "fts_artifacts",
    }
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'fts_%'"
    ).fetchall()
    names = {str(r["name"]) for r in rows}

    if names:
        assert fts_tables.issubset(names)
        trigger_row = conn.execute(
            "SELECT COUNT(*) AS n FROM sqlite_master WHERE type='trigger' AND name LIKE '%_fts_%'"
        ).fetchone()
        assert int(trigger_row["n"]) >= 18


def test_search_with_fts5_returns_relevant_results(tmp_path: Path) -> None:
    ledger = make_tmp_repo(tmp_path)
    if not _fts_tables_exist(ledger):
        pytest.skip("SQLite build does not include FTS5")

    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="Needle Discovery Project",
        domain="ml",
        stage="intake",
        git_init=False,
    )

    ledger.upsert_idea(
        idea_id="idea1",
        project_id=project["id"],
        status="candidate",
        score_total=None,
        data={
            "id": "idea1",
            "title": "Needle Idea",
            "abstract": "Find a robust needle in noisy data.",
            "status": "candidate",
        },
    )

    out = search(ledger, query="Needle", kind="ledger", limit=10)
    hit_ids = {h["id"] for h in out["hits"]}
    assert "ledger:projects/p1" in hit_ids
    assert "ledger:ideas/idea1" in hit_ids
    assert any("[" in h["snippet"] and "]" in h["snippet"] for h in out["hits"])


def test_search_like_fallback_when_fts_table_missing(tmp_path: Path) -> None:
    ledger = make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="Fallback Search Project",
        domain="",
        stage="intake",
        git_init=False,
    )

    conn = ledger.conn()
    conn.execute("DROP TABLE IF EXISTS fts_projects;")
    conn.execute("DROP TABLE IF EXISTS fts_tasks;")
    conn.execute("DROP TABLE IF EXISTS fts_ideas;")
    conn.execute("DROP TABLE IF EXISTS fts_evidence;")
    conn.execute("DROP TABLE IF EXISTS fts_reviews;")
    conn.execute("DROP TABLE IF EXISTS fts_artifacts;")
    conn.commit()

    out = search(ledger, query="Fallback", kind="ledger", limit=10)
    assert any(h["id"] == f"ledger:projects/{project['id']}" for h in out["hits"])


def test_search_special_characters_do_not_crash(tmp_path: Path) -> None:
    ledger = make_tmp_repo(tmp_path)
    create_project(
        ledger=ledger,
        project_id="p1",
        title="Special Char Search",
        domain="",
        stage="intake",
        git_init=False,
    )

    queries = [
        '""" OR ) ( *',
        "!!! ((( ***",
        "needle +++ &&& ???",
    ]
    for q in queries:
        out = search(ledger, query=q, kind="ledger", limit=10)
        assert isinstance(out.get("hits"), list)
