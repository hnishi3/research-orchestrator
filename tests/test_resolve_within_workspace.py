"""Direct unit tests for resolve_within_workspace()."""
from __future__ import annotations

from pathlib import Path

import pytest

from resorch.paths import resolve_within_workspace


def test_relative_path_inside(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    result = resolve_within_workspace(ws, "subdir/file.txt")
    assert result == (ws / "subdir" / "file.txt").resolve()


def test_absolute_path_inside(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    target = ws / "subdir" / "file.txt"
    result = resolve_within_workspace(ws, str(target))
    assert result == target.resolve()


def test_absolute_path_outside_rejects(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    with pytest.raises(SystemExit, match="must be within workspace"):
        resolve_within_workspace(ws, "/tmp/evil.txt")


def test_dotdot_escape_rejects(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    with pytest.raises(SystemExit, match="must be within workspace"):
        resolve_within_workspace(ws, "../../etc/passwd")


def test_dot_resolves_to_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    result = resolve_within_workspace(ws, ".")
    assert result == ws.resolve()


def test_label_appears_in_error(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    with pytest.raises(SystemExit, match="custom label"):
        resolve_within_workspace(ws, "/tmp/evil", label="custom label")


def test_accepts_path_object(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    result = resolve_within_workspace(ws, Path("subdir/file.txt"))
    assert result == (ws / "subdir" / "file.txt").resolve()


def test_symlink_inside_workspace_accepted(tmp_path: Path) -> None:
    """A symlink inside workspace pointing outside should be accepted (logical path is inside)."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    real_data = tmp_path / "real_data"
    real_data.mkdir()
    (real_data / "file.txt").write_text("data", encoding="utf-8")
    (ws / "data").symlink_to(real_data)
    # Should NOT raise even though the symlink target is outside workspace
    result = resolve_within_workspace(ws, "data/file.txt")
    assert result.exists()


def test_symlink_returns_logical_path(tmp_path: Path) -> None:
    """The returned path should be the logical path, not the resolved symlink target."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    real_data = tmp_path / "real_data"
    real_data.mkdir()
    (ws / "data").symlink_to(real_data)
    result = resolve_within_workspace(ws, "data/file.txt")
    # The result should be under workspace/data/, not under real_data/
    assert str(result).startswith(str(ws))
