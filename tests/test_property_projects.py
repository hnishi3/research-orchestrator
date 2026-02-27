from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from conftest import make_tmp_repo
from resorch.projects import _PROJECT_ID_RE, create_project, get_project
from resorch.utils import slugify


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
            "SQL: '\"; DROP TABLE projects; --",
            "x" * 1024,
            "line1\nline2\tline3",
        ]
    ),
)
VALID_PROJECT_ID = st.from_regex(_PROJECT_ID_RE, fullmatch=True)


@given(value=EDGE_TEXT)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_slugify_is_idempotent(value: str) -> None:
    once = slugify(value)
    twice = slugify(once)
    assert twice == once


@given(title=EDGE_TEXT, domain=EDGE_TEXT)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_create_project_auto_id_round_trip(tmp_path: Path, title: str, domain: str) -> None:
    auto_id = slugify(title)
    assume(_PROJECT_ID_RE.match(auto_id))

    ledger = make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id=None,
        title=title,
        domain=domain,
        stage="intake",
        git_init=False,
    )
    assert _PROJECT_ID_RE.match(project["id"])
    assert project["id"] == auto_id

    fetched = get_project(ledger, project["id"])
    assert fetched["id"] == project["id"]
    assert fetched["title"] == title
    assert fetched["domain"] == domain


@given(project_id=VALID_PROJECT_ID, title=EDGE_TEXT, domain=EDGE_TEXT)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_create_project_accepts_valid_requested_id(tmp_path: Path, project_id: str, title: str, domain: str) -> None:
    ledger = make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id=project_id,
        title=title,
        domain=domain,
        stage="intake",
        git_init=False,
    )
    assert project["id"] == project_id
    assert _PROJECT_ID_RE.match(project["id"])


@given(project_id=VALID_PROJECT_ID, title=EDGE_TEXT)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_create_project_rejects_invalid_requested_id(tmp_path: Path, project_id: str, title: str) -> None:
    ledger = make_tmp_repo(tmp_path)
    invalid_id = "-" + project_id
    with pytest.raises(SystemExit, match="Invalid project id"):
        create_project(
            ledger=ledger,
            project_id=invalid_id,
            title=title,
            domain="test",
            stage="intake",
            git_init=False,
        )
