from __future__ import annotations

import string
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st

from conftest import make_tmp_repo
from resorch.artifacts import register_artifact
from resorch.projects import create_project
from resorch.retrieval import fetch, search


EDGE_TEXT = st.one_of(
    st.text(
        alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
        min_size=0,
        max_size=400,
    ),
    st.sampled_from(
        [
            "",
            "こんにちは世界",
            "SQL: '\"; DROP TABLE artifacts; --",
            "x" * 2048,
            "line1\nline2\tline3",
        ]
    ),
)
TOKEN = st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=12)


@given(content=EDGE_TEXT)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_artifact_fetch_round_trip(tmp_path: Path, content: str) -> None:
    ledger = make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="Retrieval Property Project",
        domain="test",
        stage="intake",
        git_init=False,
    )
    workspace = Path(project["repo_path"])
    artifact_path = workspace / "notes" / "doc.md"
    artifact_path.write_text(content, encoding="utf-8")
    register_artifact(ledger=ledger, project=project, kind="note", relative_path="notes/doc.md", meta={})

    out = fetch(ledger, id=f"artifact:{project['id']}/notes/doc.md")
    assert out["content"] == content
    assert out["metadata"]["type"] == "artifact"


@given(broad=TOKEN, narrow=TOKEN)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_search_subset_monotonic_for_more_specific_queries(tmp_path: Path, broad: str, narrow: str) -> None:
    ledger = make_tmp_repo(tmp_path)
    create_project(
        ledger=ledger,
        project_id="p1",
        title=f"{broad} baseline",
        domain="test",
        stage="intake",
        git_init=False,
    )
    create_project(
        ledger=ledger,
        project_id="p2",
        title=f"{broad} {narrow} specific",
        domain="test",
        stage="intake",
        git_init=False,
    )

    broad_out = search(ledger, query=broad, kind="ledger", limit=50)
    specific_query = f"{broad} {narrow}"
    specific_out = search(ledger, query=specific_query, kind="ledger", limit=50)

    broad_ids = {hit["id"] for hit in broad_out["hits"]}
    specific_ids = {hit["id"] for hit in specific_out["hits"]}
    assert specific_ids <= broad_ids


@given(query=EDGE_TEXT)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_search_handles_edge_queries_without_crashing(tmp_path: Path, query: str) -> None:
    ledger = make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="Edge Query Project",
        domain="test",
        stage="intake",
        git_init=False,
    )
    out = search(ledger, query=query, project_id=project["id"], kind="ledger", limit=10)
    assert isinstance(out, dict)
    assert "hits" in out
    assert isinstance(out["hits"], list)
