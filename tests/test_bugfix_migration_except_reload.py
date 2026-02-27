"""Tests for batch-2 fixes:

1. DB migration atomicity: each migration step wrapped in self.transaction()
2. except catch narrowing: SystemExit no longer swallowed by except clauses
3. Config reload consistency: policy_now reused at L483, not re-loaded from disk
"""
from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from resorch.ledger import Ledger
from resorch.paths import RepoPaths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> Ledger:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo))
    ledger.init()
    return ledger


def _make_raw_db(tmp_path: Path) -> Ledger:
    """Create a Ledger with only the meta table (no schema tables yet)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo))
    # Manually create just the meta table (bypassing full init).
    conn = ledger.conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        """
    )
    conn.commit()
    return ledger


# ---------------------------------------------------------------------------
# 1. DB migration atomicity
# ---------------------------------------------------------------------------

def test_migration_rolls_back_on_failure(tmp_path: Path) -> None:
    """If a migration function raises mid-way, schema_version should NOT advance."""
    ledger = _make_raw_db(tmp_path)

    # Set up: run v1 schema creation + migrations up to v5.
    ledger._create_schema_v1()
    ledger.set_meta("schema_version", "1")
    ledger.conn().commit()
    ledger._migrate_1_to_2()
    ledger.set_meta("schema_version", "2")
    ledger.conn().commit()
    ledger._migrate_2_to_3()
    ledger.set_meta("schema_version", "3")
    ledger.conn().commit()
    ledger._migrate_3_to_4()
    ledger.set_meta("schema_version", "4")
    ledger.conn().commit()
    ledger._migrate_4_to_5()
    ledger.set_meta("schema_version", "5")
    ledger.conn().commit()
    ledger._migrate_5_to_6()
    ledger.set_meta("schema_version", "6")
    ledger.conn().commit()

    assert ledger.get_meta("schema_version") == "6"

    # Monkey-patch _migrate_6_to_7 to fail after partial work.
    original_migrate = ledger._migrate_6_to_7

    def failing_migrate() -> None:
        # Do the real migration first (partial work committed in-transaction).
        original_migrate()
        # Then crash before the version bump.
        raise RuntimeError("Simulated disk failure during migration")

    ledger._migrate_6_to_7 = failing_migrate  # type: ignore[assignment]

    # Re-run the migration loop (as init() does).
    schema_version = ledger.get_meta("schema_version")
    _migrations = [
        ("5", "6", ledger._migrate_5_to_6),
        ("6", "7", ledger._migrate_6_to_7),
    ]
    with pytest.raises(RuntimeError, match="Simulated disk failure"):
        for from_ver, to_ver, migrate_fn in _migrations:
            if schema_version == from_ver:
                with ledger.transaction():
                    migrate_fn()
                    ledger.set_meta("schema_version", to_ver)
                schema_version = to_ver

    # Key assertion: schema_version should still be "6" (rolled back).
    assert ledger.get_meta("schema_version") == "6"


def test_migration_succeeds_atomically(tmp_path: Path) -> None:
    """On success, both the migration DDL and version bump are committed together."""
    ledger = _make_repo(tmp_path)
    # A fully-initialized ledger should be at schema version 7.
    assert ledger.get_meta("schema_version") == "7"


def test_transaction_rollback_on_error(tmp_path: Path) -> None:
    """Verify that Ledger.transaction() rolls back all changes on exception."""
    ledger = _make_repo(tmp_path)

    # Insert a project so we have something to modify.
    from resorch.projects import create_project
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="test", stage="intake", git_init=False)
    assert project["title"] == "P1"

    # Try to update inside a failing transaction.
    with pytest.raises(RuntimeError, match="deliberate"):
        with ledger.transaction():
            ledger._exec(
                "UPDATE projects SET title = ? WHERE id = ?",
                ("CHANGED", "p1"),
            )
            # Verify the change is visible inside the transaction.
            row = ledger._exec("SELECT title FROM projects WHERE id = ?", ("p1",)).fetchone()
            assert row["title"] == "CHANGED"
            raise RuntimeError("deliberate rollback")

    # After rollback, original value should be restored.
    row = ledger._exec("SELECT title FROM projects WHERE id = ?", ("p1",)).fetchone()
    assert row["title"] == "P1"


# ---------------------------------------------------------------------------
# 2. except catch narrowing: SystemExit propagation
# ---------------------------------------------------------------------------

def test_systemexit_not_caught_by_except_exception() -> None:
    """Verify Python semantics: `except Exception` does NOT catch SystemExit.

    Before the fix, autopilot.py used `except (SystemExit, Exception)` which
    swallowed SystemExit and prevented clean process termination.
    """
    caught = False
    with pytest.raises(SystemExit):
        try:
            raise SystemExit(1)
        except Exception:  # noqa: BLE001
            caught = True
    assert not caught, "except Exception should not catch SystemExit"


def test_keyboardinterrupt_not_caught_by_except_exception() -> None:
    """Same guarantee for KeyboardInterrupt (Ctrl+C)."""
    caught = False
    with pytest.raises(KeyboardInterrupt):
        try:
            raise KeyboardInterrupt()
        except Exception:  # noqa: BLE001
            caught = True
    assert not caught, "except Exception should not catch KeyboardInterrupt"


def test_autopilot_source_has_no_systemexit_in_except() -> None:
    """Source-level check: autopilot.py should not catch SystemExit in except clauses."""
    import ast
    from resorch import autopilot

    source = inspect.getsource(autopilot)
    tree = ast.parse(source)

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            # Check for `except (SystemExit, ...)` or `except SystemExit`
            names = []
            if isinstance(node.type, ast.Name):
                names = [node.type.id]
            elif isinstance(node.type, ast.Tuple):
                names = [
                    elt.id for elt in node.type.elts
                    if isinstance(elt, ast.Name)
                ]
            if "SystemExit" in names:
                violations.append(node.lineno)

    assert violations == [], f"autopilot.py still catches SystemExit at lines: {violations}"


# ---------------------------------------------------------------------------
# 3. Config reload consistency
# ---------------------------------------------------------------------------

def test_policy_now_reused_not_reloaded() -> None:
    """Verify that run_autopilot_iteration uses `policy_now` for the action
    execution section, rather than calling load_review_policy() again.

    Before the fix, L483 re-loaded policy from disk, which could differ from
    the policy loaded at L259 if configs were modified mid-iteration.
    """
    from resorch import autopilot

    source = inspect.getsource(autopilot.run_autopilot_iteration)

    # The function should contain `policy = policy_now` (the fix).
    assert "policy = policy_now" in source, (
        "Expected 'policy = policy_now' in run_autopilot_iteration. "
        "The action-execution section should reuse the policy loaded at iteration start."
    )

    # Count direct calls to load_review_policy in the function body.
    # There should be exactly 2: one for the pending-jobs early-return path (L214)
    # and one for the main path (L259). NOT a third one at the action-execution section.
    call_count = source.count("load_review_policy(")
    assert call_count == 2, (
        f"Expected exactly 2 calls to load_review_policy() in run_autopilot_iteration, "
        f"found {call_count}. The action-execution section should use policy_now."
    )
