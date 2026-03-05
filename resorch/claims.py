from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.artifacts import register_artifact
from resorch.ledger import Ledger
from resorch.paths import resolve_within_workspace
from resorch.utils import utc_now_iso


_CLAIM_FILE_RE = re.compile(r"^claim_(\d{3,})\.md$")


def _next_claim_id(claims_dir: Path) -> str:
    max_n = 0
    if claims_dir.exists():
        for p in claims_dir.iterdir():
            if not p.is_file():
                continue
            m = _CLAIM_FILE_RE.match(p.name)
            if not m:
                continue
            try:
                max_n = max(max_n, int(m.group(1)))
            except ValueError:
                continue
    return f"claim_{max_n + 1:03d}"


def _render_claim_md(
    *,
    claim_id: str,
    created_at: str,
    statement: str,
    evidence_rows: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(f"# Claim {claim_id}\n\n")
    lines.append(f"- claim_id: `{claim_id}`\n")
    lines.append(f"- created_at: `{created_at}`\n")
    lines.append("- evidence_ids:\n")
    if evidence_rows:
        for ev in evidence_rows:
            lines.append(f"  - {ev['id']}\n")
    else:
        lines.append("  - (none)\n")
    lines.append("\n")

    lines.append("## Statement\n\n")
    lines.append((statement.strip() or "(missing)") + "\n\n")

    lines.append("## Evidence\n\n")
    if evidence_rows:
        for ev in evidence_rows:
            title = str(ev.get("title") or "").strip() or "(untitled)"
            url = str(ev.get("url") or "").strip()
            summary = str(ev.get("summary") or "").strip()
            lines.append(f"- {ev['id']}: {title}\n")
            if url:
                lines.append(f"  - url: {url}\n")
            if summary:
                lines.append(f"  - summary: {summary}\n")
    else:
        lines.append("- (none)\n")
    lines.append("\n")

    lines.append("## Notes\n\n- \n")
    return "".join(lines)


def create_claim(
    *,
    ledger: Ledger,
    project_id: str,
    statement: str,
    evidence_ids: Optional[List[str]] = None,
    path: Optional[str] = None,
    overwrite: bool = False,
    register_as_artifact: bool = True,
) -> Dict[str, Any]:
    project = ledger.get_project(project_id)
    workspace = Path(project["repo_path"]).resolve()
    claims_dir = workspace / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)

    claim_id = _next_claim_id(claims_dir) if not path else Path(path).stem
    created_at = utc_now_iso()

    evidence_rows: List[Dict[str, Any]] = []
    for eid in evidence_ids or []:
        eid = str(eid).strip()
        if not eid:
            continue
        ev = ledger.get_evidence(eid)
        if str(ev.get("project_id")) != project_id:
            raise SystemExit(f"Evidence {eid} does not belong to project {project_id}.")
        evidence_rows.append(ev)

    rel_path = path or f"claims/{claim_id}.md"
    out_p = resolve_within_workspace(workspace, rel_path, label="claim output path")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists() and not overwrite:
        raise SystemExit(f"Claim already exists: {out_p} (use --overwrite)")

    out_p.write_text(
        _render_claim_md(
            claim_id=claim_id,
            created_at=created_at,
            statement=statement,
            evidence_rows=evidence_rows,
        ),
        encoding="utf-8",
    )

    artifact = None
    if register_as_artifact:
        artifact = register_artifact(
            ledger=ledger,
            project={"id": project_id, "repo_path": str(workspace)},
            kind="claim_md",
            relative_path=out_p.resolve().relative_to(workspace).as_posix(),
            meta={"claim_id": claim_id},
        )

    return {"claim_id": claim_id, "path": str(out_p), "artifact": artifact}

