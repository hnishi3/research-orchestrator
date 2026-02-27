from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from resorch.cohort import run_cohort
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project, get_project


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_cohort_run_creates_member_projects_and_lab_meeting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    base = create_project(ledger=ledger, project_id="base", title="Base", domain="", stage="analysis", git_init=False)

    # Seed a few ideas in the base project so cohort can reserve them.
    ledger.upsert_idea(idea_id="idea_a", project_id=base["id"], status="candidate", score_total=3.0, data={"id": "idea_a", "title": "A"})
    ledger.upsert_idea(idea_id="idea_b", project_id=base["id"], status="candidate", score_total=2.0, data={"id": "idea_b", "title": "B"})

    calls: List[str] = []

    def fake_run_agent_loop(*, ledger: Ledger, project_id: str, objective: str, max_steps: int, dry_run: bool, config_path: str, **kwargs: Any) -> Dict[str, Any]:  # noqa: ANN001
        calls.append(project_id)
        ws = Path(get_project(ledger, project_id)["repo_path"]).resolve()
        sb_path = ws / "results" / "scoreboard.json"
        sb = json.loads(sb_path.read_text(encoding="utf-8"))
        sb["primary_metric"]["current"] = {"mean": float(len(calls)) * 0.1}
        sb_path.write_text(json.dumps(sb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"project_id": project_id, "objective": objective, "steps": [], "stopped_reason": "done", "dry_run": dry_run, "config": {}}

    monkeypatch.setattr("resorch.cohort.run_agent_loop", fake_run_agent_loop)

    out = run_cohort(ledger=ledger, base_project_id=base["id"], objective="Test objective", n=2, max_steps=1, dry_run=True, ideas_per_agent=1)
    assert out["base_project_id"] == "base"
    assert len(out["members"]) == 2

    base_ws = Path(get_project(ledger, base["id"])["repo_path"]).resolve()
    lock_dir = base_ws / "ideas" / "locks"
    assert lock_dir.exists()
    assert len(list(lock_dir.glob("*.lock.json"))) >= 1

    lab_path = base_ws / out["lab_meeting_path"]
    assert lab_path.exists()
