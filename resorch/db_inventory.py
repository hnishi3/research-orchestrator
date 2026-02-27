from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.artifacts import put_artifact
from resorch.constraints import get_databases_inventory, load_constraints
from resorch.ledger import Ledger
from resorch.projects import get_project
from resorch.utils import utc_now_iso


def _resolve_db_path(*, workspace: Path, root: str, item_path: str) -> Path:
    p = Path(item_path).expanduser()
    if p.is_absolute():
        return p.resolve()

    root_s = str(root or "").strip()
    if not root_s:
        raise SystemExit("constraints.yaml databases.root is required when databases.items[].path is relative")

    root_p = Path(root_s).expanduser()
    if not root_p.is_absolute():
        root_p = (workspace / root_p).resolve()
    return (root_p / p).resolve()


def _file_info(path: Path) -> Dict[str, Any]:
    try:
        st = path.stat()
        mtime = datetime.fromtimestamp(float(st.st_mtime), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            "exists": True,
            "size_bytes": int(st.st_size),
            "mtime_utc": mtime,
        }
    except OSError:
        return {"exists": False}


def ensure_databases(
    *,
    ledger: Ledger,
    project_id: str,
    constraints_path: str = "constraints.yaml",
    out_dir: str = "runs/db",
) -> Dict[str, Any]:
    project = get_project(ledger, project_id)
    workspace = Path(project["repo_path"]).resolve()

    constraints = load_constraints(ledger=ledger, project_id=project_id, path=constraints_path)
    inv = get_databases_inventory(constraints)
    root = str(inv.get("root") or "").strip()
    items = inv.get("items") or []
    if not isinstance(items, list):
        items = []

    found: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    auto_download_requested: List[Dict[str, Any]] = []

    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "").strip()
        path_s = str(it.get("path") or "").strip()
        if not name or not path_s:
            continue
        abs_path = _resolve_db_path(workspace=workspace, root=root, item_path=path_s)
        info = _file_info(abs_path)
        rec = {
            "name": name,
            "path": path_s,
            "abs_path": str(abs_path),
            "license": str(it.get("license") or ""),
            "version": str(it.get("version") or ""),
            "auto_download": bool(it.get("auto_download", False)),
            **info,
        }
        if rec["auto_download"]:
            auto_download_requested.append(rec)
        if info.get("exists") is True:
            found.append(rec)
        else:
            missing.append(rec)

    ts = utc_now_iso().replace(":", "").replace("Z", "Z")
    out_dir_rel = str(out_dir).strip().rstrip("/")
    if not out_dir_rel:
        out_dir_rel = "runs/db"

    report_rel = f"{out_dir_rel}/ensure-{ts}.json"
    script_rel = f"{out_dir_rel}/ensure-{ts}.sh"

    report = {
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "constraints_path": constraints_path,
        "databases": {"root": root, "items_count": len(items)},
        "found": found,
        "missing": missing,
        "auto_download_requested": auto_download_requested,
        "notes": "This command does not download anything. It only reports missing DB files and generates a manual script.",
    }
    put_artifact(
        ledger=ledger,
        project=project,
        relative_path=report_rel,
        content=json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        mode="overwrite",
        kind="db_ensure_report_json",
    )

    lines: List[str] = []
    lines.append("#!/usr/bin/env bash\n")
    lines.append("set -euo pipefail\n\n")
    lines.append("# DB ensure script (generated; safe by default)\n")
    lines.append(f"# Generated at: {utc_now_iso()}\n")
    lines.append("# Fill in download/rsync steps for your lab environment.\n")
    lines.append("# NOTE: Do NOT commit secrets/tokens.\n\n")
    lines.append("echo \"Missing DB items:\" \n")
    if not missing:
        lines.append("echo \"- (none)\" \n")
    for it in missing:
        lines.append(f"echo \"- {it['name']}: {it['abs_path']}\" \n")
        if it.get("license"):
            lines.append(f"echo \"  license: {it['license']}\" \n")
        if it.get("version"):
            lines.append(f"echo \"  version: {it['version']}\" \n")
        lines.append("echo \"  TODO: add commands to provision this DB\" \n")
        lines.append("echo \"\" \n")

    put_artifact(
        ledger=ledger,
        project=project,
        relative_path=script_rel,
        content="".join(lines),
        mode="overwrite",
        kind="db_ensure_script_sh",
    )
    try:
        (workspace / script_rel).chmod(0o755)
    except OSError:
        pass

    return {"report_path": report_rel, "script_path": script_rel, "missing": len(missing), "found": len(found)}
