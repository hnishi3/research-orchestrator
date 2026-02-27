from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st

from conftest import make_tmp_repo
from resorch.ideas import get_idea, import_ideas_jsonl, list_ideas, set_idea_status
from resorch.projects import create_project


EDGE_TEXT = st.one_of(
    st.text(
        alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
        min_size=0,
        max_size=200,
    ),
    st.sampled_from(
        [
            "",
            "こんにちは世界",
            "SQL: '\"; DROP TABLE ideas; --",
            "x" * 1024,
            "line1\nline2\tline3",
        ]
    ),
)
IDEA_ID = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", fullmatch=True)
IDEA_STATUS = st.sampled_from(
    ["candidate", "active", "rejected", "smoke_passed", "selected", "in_progress", "parked", "done"]
)


@given(idea_id=IDEA_ID, status=IDEA_STATUS, title=EDGE_TEXT, claim=EDGE_TEXT)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_import_and_get_idea_round_trip(
    tmp_path: Path,
    idea_id: str,
    status: str,
    title: str,
    claim: str,
) -> None:
    ledger = make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="Ideas Property Project",
        domain="test",
        stage="intake",
        git_init=False,
    )
    workspace = Path(project["repo_path"])
    ideas_path = workspace / "ideas" / "ideas.jsonl"
    record = {"id": idea_id, "status": status, "title": title, "one_sentence_claim": claim}
    ideas_path.write_text(json.dumps(record, ensure_ascii=True) + "\n", encoding="utf-8")

    out = import_ideas_jsonl(ledger=ledger, project_id=project["id"], input_path="ideas/ideas.jsonl")
    assert out["imported"] == 1

    got = get_idea(ledger=ledger, idea_id=idea_id)
    assert got["id"] == idea_id
    assert got["project_id"] == project["id"]
    assert got["status"] == status
    assert got["data"]["title"] == title
    assert got["data"]["one_sentence_claim"] == claim

    listed = list_ideas(ledger=ledger, project_id=project["id"], limit=50)
    assert any(item["id"] == idea_id for item in listed)


@given(initial_status=IDEA_STATUS, next_status=IDEA_STATUS, title=EDGE_TEXT)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_set_idea_status_round_trip(tmp_path: Path, initial_status: str, next_status: str, title: str) -> None:
    ledger = make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="Status Property Project",
        domain="test",
        stage="intake",
        git_init=False,
    )
    ledger.upsert_idea(
        idea_id="idea-prop-1",
        project_id=project["id"],
        status=initial_status,
        score_total=None,
        data={"id": "idea-prop-1", "status": initial_status, "title": title},
    )

    updated = set_idea_status(ledger=ledger, idea_id="idea-prop-1", status=next_status)
    assert updated["status"] == next_status
    assert updated["data"]["status"] == next_status

    fetched = get_idea(ledger=ledger, idea_id="idea-prop-1")
    assert fetched["status"] == next_status
    assert fetched["data"]["status"] == next_status
