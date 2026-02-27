from __future__ import annotations

import os
import time
from pathlib import Path

from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.visual_inspection import approve_visual_inspection, get_visual_inspection_status


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_visual_inspection_status_pending(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "results" / "fig").mkdir(parents=True)
    fig = ws / "results" / "fig" / "a.png"
    fig.write_bytes(b"fakepng")

    policy = {
        "requires_visual_inspection": True,
        "visual_inspection": {"marker_path": "results/fig/visual_inspection.ok", "figure_globs": ["results/fig/*.png"]},
    }
    st = get_visual_inspection_status(policy=policy, workspace=ws)
    assert st.enabled is True
    assert "results/fig/a.png" in st.pending_figures

    marker = ws / "results" / "fig" / "visual_inspection.ok"
    marker.write_text("ok\n", encoding="utf-8")
    # Ensure marker is newer than the figure.
    now = time.time()
    os.utime(marker, (now + 1, now + 1))
    st2 = get_visual_inspection_status(policy=policy, workspace=ws)
    assert st2.pending_figures == []


def test_visual_inspection_approve_writes_marker(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)
    ws = Path(project["repo_path"]).resolve()
    (ws / "results" / "fig").mkdir(parents=True, exist_ok=True)

    out = approve_visual_inspection(ledger=ledger, project_id=project["id"], note="ok")
    marker = ws / out["marker_path"]
    assert marker.exists()

