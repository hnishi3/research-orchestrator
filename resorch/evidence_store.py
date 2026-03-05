from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from resorch.artifacts import register_artifact
from resorch.ledger import Ledger
from resorch.paths import resolve_within_workspace
from resorch.utils import utc_now_iso


_EVIDENCE_KINDS = {"paper", "blog", "doc", "dataset", "benchmark", "repo", "other"}


def _parse_evidence_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["meta"] = json.loads(out.pop("meta_json") or "{}")
    return out


def validate_evidence_url(url: str) -> Dict[str, Any]:
    normalized_url = str(url).strip()
    parsed = urlparse(normalized_url)
    scheme = parsed.scheme.lower()
    is_valid = scheme in {"http", "https"} and bool(parsed.netloc)
    result: Dict[str, Any] = {
        "valid": is_valid,
        "url": normalized_url,
        "scheme": scheme or None,
        "netloc": parsed.netloc or None,
    }
    if not is_valid:
        result["error"] = "Invalid evidence URL: must be http/https with a valid hostname"
    return result


def _check_url_reachable(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    req = Request(url, method="HEAD")
    try:
        with urlopen(req, timeout=timeout) as response:
            status_code = getattr(response, "status", None) or response.getcode()
            return {"reachable": True, "status_code": int(status_code)}
    except HTTPError as exc:
        return {"reachable": True, "status_code": int(exc.code)}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def add_evidence(
    *,
    ledger: Ledger,
    project_id: str,
    kind: str,
    title: str,
    url: str,
    summary: str,
    idea_id: Optional[str] = None,
    retrieved_at: Optional[str] = None,
    relevance: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    register_as_artifact: bool = True,
    validate_url_reachable: bool = False,
) -> Dict[str, Any]:
    project = ledger.get_project(project_id)
    workspace = Path(project["repo_path"]).resolve()

    kind = str(kind).strip()
    if kind not in _EVIDENCE_KINDS:
        raise SystemExit(f"Invalid evidence kind: {kind}. Expected one of: {', '.join(sorted(_EVIDENCE_KINDS))}")
    title = str(title).strip()
    url = str(url).strip()
    summary = str(summary).strip()
    if not title:
        raise SystemExit("evidence.add requires --title")
    if not url:
        raise SystemExit("evidence.add requires --url")
    if not summary:
        raise SystemExit("evidence.add requires --summary")

    url_validation = validate_evidence_url(url)
    if not url_validation["valid"]:
        raise SystemExit(url_validation["error"])

    if idea_id is not None:
        idea_id = str(idea_id).strip() or None
        if idea_id:
            idea = ledger.get_idea(idea_id)
            if str(idea.get("project_id")) != project_id:
                raise SystemExit(f"Idea {idea_id} does not belong to project {project_id}.")

    if retrieved_at is None:
        retrieved_at = utc_now_iso()

    evidence_id = uuid4().hex
    rel_out = output_path or f"evidence/{evidence_id}.json"
    out_p = resolve_within_workspace(workspace, rel_out, label="evidence output path")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    meta_out = dict(meta) if meta else {}
    if validate_url_reachable:
        meta_out["url_validation"] = _check_url_reachable(url)

    payload: Dict[str, Any] = {
        "id": evidence_id,
        "project_id": project_id,
        "idea_id": idea_id,
        "kind": kind,
        "title": title,
        "url": url,
        "retrieved_at": retrieved_at,
        "summary": summary,
        "relevance": relevance,
        "meta": meta_out,
    }
    out_p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stored_rel = out_p.resolve().relative_to(workspace).as_posix()

    artifact = None
    if register_as_artifact:
        artifact = register_artifact(
            ledger=ledger,
            project={"id": project_id, "repo_path": str(workspace)},
            kind="evidence_json",
            relative_path=stored_rel,
            meta={"evidence_id": evidence_id, "url": url},
        )

    row = ledger.insert_evidence(
        evidence_id=evidence_id,
        project_id=project_id,
        idea_id=idea_id,
        kind=kind,
        title=title,
        url=url,
        retrieved_at=str(retrieved_at),
        summary=summary,
        relevance=float(relevance) if relevance is not None else None,
        meta=meta_out,
        artifact_path=stored_rel if artifact is not None else None,
    )
    return {"evidence": _parse_evidence_row(row), "artifact": artifact, "stored_path": str(out_p)}


def list_evidence(
    *,
    ledger: Ledger,
    project_id: str,
    idea_id: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    rows = ledger.list_evidence(project_id=project_id, idea_id=idea_id, limit=limit)
    return [_parse_evidence_row(r) for r in rows]


def get_evidence(*, ledger: Ledger, evidence_id: str) -> Dict[str, Any]:
    row = ledger.get_evidence(evidence_id)
    return _parse_evidence_row(row)
