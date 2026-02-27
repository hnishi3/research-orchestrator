from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from resorch.ledger import Ledger
from resorch.utils import sha256_file

log = logging.getLogger(__name__)


def _within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def register_artifact(
    *,
    ledger: Ledger,
    project: Dict[str, Any],
    kind: str,
    relative_path: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    workspace = Path(project["repo_path"]).resolve()
    path = (workspace / relative_path).resolve()
    if not _within(path, workspace):
        log.warning("Artifact path outside workspace (skipped). workspace=%s path=%s", workspace, path)
        return {"id": None, "project_id": project["id"], "kind": kind, "path": relative_path, "sha256": None, "meta": meta or {}, "skipped": True}

    sha = sha256_file(path) if path.exists() and path.is_file() else None
    artifact = ledger.insert_artifact(
        artifact_id=uuid4().hex,
        project_id=project["id"],
        kind=kind,
        path=str(Path(relative_path).as_posix()),
        sha256=sha,
        meta=meta or {},
    )

    artifact["meta"] = json.loads(artifact.pop("meta_json") or "{}")
    return artifact


def list_artifacts(
    ledger: Ledger, *, project_id: str, prefix: Optional[str] = None, limit: int = 200
) -> List[Dict[str, Any]]:
    rows = ledger.list_artifacts(project_id, prefix=prefix, limit=limit)
    out: List[Dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        item["meta"] = json.loads(item.pop("meta_json") or "{}")
        out.append(item)
    return out


def put_artifact(
    *,
    ledger: Ledger,
    project: Dict[str, Any],
    relative_path: str,
    content: str,
    mode: str = "overwrite",
    kind: Optional[str] = None,
) -> Dict[str, Any]:
    workspace = Path(project["repo_path"]).resolve()
    rel = Path(relative_path)
    abs_path = (workspace / rel).resolve()
    if not _within(abs_path, workspace):
        log.warning("Artifact path outside workspace (skipped). workspace=%s path=%s", workspace, abs_path)
        return {"id": None, "project_id": project["id"], "kind": kind, "path": relative_path, "sha256": None, "meta": {"mode": mode}, "skipped": True}

    abs_path.parent.mkdir(parents=True, exist_ok=True)
    if mode not in {"overwrite", "append"}:
        raise SystemExit("artifact.put mode must be 'overwrite' or 'append'")
    if mode == "append":
        with abs_path.open("a", encoding="utf-8") as f:
            f.write(content)
    else:
        import os
        import tempfile
        fd, tmp = tempfile.mkstemp(dir=str(abs_path.parent), suffix=".tmp")
        os.close(fd)  # close immediately; reopen by path to avoid fd leak
        try:
            Path(tmp).write_text(content, encoding="utf-8")
            Path(tmp).replace(abs_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    return register_artifact(
        ledger=ledger,
        project=project,
        kind=kind or "artifact",
        relative_path=rel.as_posix(),
        meta={"mode": mode},
    )
