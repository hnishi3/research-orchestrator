from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st

from conftest import make_tmp_repo
from resorch.projects import create_project
from resorch.stage_gates import Unknown, compute_gate_env, eval_expr, evaluate_transitions


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
            "SQL: '\"; DROP TABLE stage_gates; --",
            "x" * 1024,
            "line1\nline2\tline3",
        ]
    ),
)
IDENT = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,15}", fullmatch=True)


@given(name=IDENT)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_eval_expr_missing_identifier_is_unknown(name: str) -> None:
    out = eval_expr(name, {})
    assert isinstance(out, Unknown)
    assert out.reason == f"missing:{name}"


@given(idea_count=st.integers(min_value=0, max_value=15), evidence_len=st.integers(min_value=0, max_value=5), novelty=EDGE_TEXT)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_compute_gate_env_round_trip_counts(
    tmp_path: Path,
    idea_count: int,
    evidence_len: int,
    novelty: str,
) -> None:
    ledger = make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="Stage Gate Property Project",
        domain="test",
        stage="intake",
        git_init=False,
    )
    for i in range(idea_count):
        ledger.upsert_idea(
            idea_id=f"idea-{i}",
            project_id=project["id"],
            status="candidate",
            score_total=1.0,
            data={
                "id": f"idea-{i}",
                "status": "candidate",
                "evidence": [{} for _ in range(evidence_len)],
                "novelty_statement": novelty,
                "scores": {"total": 1.0},
            },
        )

    env = compute_gate_env(ledger=ledger, project_id=project["id"])
    assert env["idea_count"] == idea_count
    if idea_count == 0:
        assert isinstance(env["evidence_count"], Unknown)
    else:
        assert env["evidence_count"] == evidence_len

    expected_novelty = "" if (idea_count > 0 and novelty.strip() == "") else "ok"
    assert env["novelty_statement"] == expected_novelty


@given(x=st.booleans(), y=st.booleans())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_transition_requires_are_monotonic_under_stricter_constraints(x: bool, y: bool) -> None:
    env = {"x": x, "y": y}
    base_cfg = {
        "transitions": {
            "t": {
                "auto_pass_if": ["true"],
                "requires": ["x"],
            }
        }
    }
    strict_cfg = {
        "transitions": {
            "t": {
                "auto_pass_if": ["true"],
                "requires": ["x", "y"],
            }
        }
    }

    base_decision = evaluate_transitions(config=base_cfg, env=env)["transitions"]["t"]["decision"]
    strict_decision = evaluate_transitions(config=strict_cfg, env=env)["transitions"]["t"]["decision"]

    if strict_decision == "auto_pass":
        assert base_decision == "auto_pass"
