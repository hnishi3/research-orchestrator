from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st

from conftest import make_tmp_repo


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
            "SQL: '\"; DROP TABLE ledger; --",
            "x" * 1024,
            "line1\nline2\tline3",
        ]
    ),
)
PROJECT_ID = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", fullmatch=True)
IDEA_ID = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", fullmatch=True)
IDEA_STATUS = st.sampled_from(["candidate", "active", "rejected", "smoke_passed", "selected", "in_progress", "parked", "done"])


@given(project_id=PROJECT_ID, title=EDGE_TEXT, domain=EDGE_TEXT, stage=EDGE_TEXT, meta_note=EDGE_TEXT)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_insert_and_get_project_round_trip(
    tmp_path: Path,
    project_id: str,
    title: str,
    domain: str,
    stage: str,
    meta_note: str,
) -> None:
    ledger = make_tmp_repo(tmp_path)
    project = ledger.insert_project(
        project_id=project_id,
        title=title,
        domain=domain,
        stage=stage,
        repo_path=str(tmp_path / "workspace"),
        meta={"note": meta_note},
    )
    fetched = ledger.get_project(project_id)

    assert fetched["id"] == project["id"]
    assert fetched["title"] == title
    assert fetched["domain"] == domain
    assert fetched["stage"] == stage
    assert json.loads(fetched["meta_json"])["note"] == meta_note


@given(project_id=PROJECT_ID, idea_id=IDEA_ID, status=IDEA_STATUS, payload=EDGE_TEXT)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_upsert_and_get_idea_round_trip(
    tmp_path: Path,
    project_id: str,
    idea_id: str,
    status: str,
    payload: str,
) -> None:
    ledger = make_tmp_repo(tmp_path)
    ledger.insert_project(
        project_id=project_id,
        title="Property Project",
        domain="test",
        stage="intake",
        repo_path=str(tmp_path / "workspace"),
        meta={},
    )

    row = ledger.upsert_idea(
        idea_id=idea_id,
        project_id=project_id,
        status=status,
        score_total=3.14,
        data={
            "id": idea_id,
            "status": status,
            "title": payload,
            "one_sentence_claim": payload,
            "notes": {"raw": payload},
        },
    )
    fetched = ledger.get_idea(idea_id)
    decoded = json.loads(fetched["data_json"])

    assert fetched["id"] == row["id"] == idea_id
    assert fetched["project_id"] == project_id
    assert fetched["status"] == status
    assert decoded["title"] == payload
    assert decoded["one_sentence_claim"] == payload
    assert decoded["notes"]["raw"] == payload
