from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.artifacts import register_artifact
from resorch.ideas import get_idea as get_idea_fn
from resorch.ideas import set_idea_status as set_idea_status_fn
from resorch.ledger import Ledger


def _parse_smoke_test_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["result"] = json.loads(out.pop("result_json") or "{}")
    return out


def ingest_smoke_test_result(
    *,
    ledger: Ledger,
    project_id: str,
    result_path: str,
    store_path: Optional[str] = None,
    register_as_artifact: bool = True,
    update_idea_status_on_pass: bool = True,
) -> Dict[str, Any]:
    project = ledger.get_project(project_id)
    workspace = Path(project["repo_path"]).resolve()

    inp = Path(result_path)
    if not inp.is_absolute():
        inp = (workspace / inp).resolve()
    if not inp.exists():
        raise SystemExit(f"Smoke test result file not found: {inp}")

    payload = json.loads(inp.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("Smoke test result must be a JSON object.")

    idea_id = str(payload.get("idea_id") or "").strip()
    if not idea_id:
        raise SystemExit("Smoke test result missing required field: idea_id")
    verdict = str(payload.get("verdict") or "").strip()
    if verdict not in {"pass", "fail", "timeout"}:
        raise SystemExit("Smoke test result verdict must be one of: pass, fail, timeout")
    started_at = str(payload.get("started_at") or "").strip()
    if not started_at:
        raise SystemExit("Smoke test result missing required field: started_at")
    completed_at = payload.get("completed_at")
    if completed_at is not None:
        completed_at = str(completed_at)

    idea = ledger.get_idea(idea_id)
    if str(idea.get("project_id")) != project_id:
        raise SystemExit(f"Idea {idea_id} does not belong to project {project_id}.")

    rel_store = store_path or f"runs/smoke/{idea_id}/result.json"
    store_abs = Path(rel_store)
    if not store_abs.is_absolute():
        store_abs = (workspace / store_abs).resolve()
    store_abs.parent.mkdir(parents=True, exist_ok=True)
    store_abs.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stored_rel = store_abs.resolve().relative_to(workspace).as_posix()

    artifact = None
    if register_as_artifact:
        artifact = register_artifact(
            ledger=ledger,
            project={"id": project_id, "repo_path": str(workspace)},
            kind="smoke_test_result",
            relative_path=stored_rel,
            meta={"idea_id": idea_id, "source_path": str(inp)},
        )

    smoke_row = ledger.insert_smoke_test(
        idea_id=idea_id,
        project_id=project_id,
        verdict=verdict,
        started_at=started_at,
        completed_at=completed_at if isinstance(completed_at, str) else None,
        result=payload,
        artifact_path=stored_rel if artifact is not None else None,
    )

    updated_idea = None
    if update_idea_status_on_pass and verdict == "pass":
        cur = get_idea_fn(ledger=ledger, idea_id=idea_id)
        if str(cur.get("status") or "") != "selected":
            updated_idea = set_idea_status_fn(ledger=ledger, idea_id=idea_id, status="smoke_passed")
        else:
            updated_idea = cur

    return {
        "stored_path": str(store_abs),
        "artifact": artifact,
        "smoke_test": _parse_smoke_test_row(smoke_row),
        "idea": updated_idea,
    }


def list_smoke_tests(
    *,
    ledger: Ledger,
    project_id: str,
    idea_id: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    rows = ledger.list_smoke_tests(project_id=project_id, idea_id=idea_id, limit=limit)
    return [_parse_smoke_test_row(r) for r in rows]
