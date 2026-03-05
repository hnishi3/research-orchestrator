from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

from resorch.artifacts import register_artifact
from resorch.ledger import Ledger
from resorch.paths import resolve_within_workspace
from resorch.utils import utc_now_iso


_TEMPLATE = """# constraints.yaml

# Stage 0: Constraints Capture (Topic Engine)

meta:
  created_at: "{created_at}"

research:
  domain: ""              # e.g., nlp / systems / bioinfo
  target_venues: []       # e.g., ["NeurIPS", "ICLR", "ACL"]
  contribution_types: []  # e.g., ["method", "system", "analysis", "survey"]

resources:
  gpu_available: false
  gpu_type: ""
  max_gpu_hours: 0
  budget_usd: 0
  deadline: ""            # ISO date or free text

compute:
  backend: local           # local | slurm
  local:
    max_parallel: 30       # lab server default; lower if needed
  slurm:
    sbatch: "sbatch"
    sacct: "sacct"
    squeue: "squeue"
    extra_sbatch_args: []  # e.g., ["--partition=gpu", "--time=02:00:00"]

data:
  allowed_sources: []     # e.g., ["public_only"]
  pii_allowed: false
  license_constraints: "" # e.g., "MIT-only"
  mounts: []              # external data references (do not copy large data into the workspace)
  # mounts example:
  #   - name: fastq
  #     path: /mnt/labdata/fastq
  #     read_only: true
  #     required: false
  #     notes: ""

databases:
  root: ""                # optional base path (absolute recommended)
  items: []               # DB inventory (no auto-download by default)
  # items example:
  #   - name: uniprot
  #     path: uniprot/uniprot_sprot.fasta.gz
  #     license: ""
  #     version: ""
  #     auto_download: false

risks:
  safety: ""
  ethics: ""
  reproducibility: ""

notes: ""
"""


def write_constraints_template(
    *,
    ledger: Ledger,
    project_id: str,
    path: str = "constraints.yaml",
    overwrite: bool = False,
) -> Dict[str, Any]:
    project = ledger.get_project(project_id)
    workspace = Path(project["repo_path"]).resolve()
    out = resolve_within_workspace(workspace, path, label="constraints output path")
    if out.exists() and not overwrite:
        raise SystemExit(f"constraints file already exists: {out} (use --overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_TEMPLATE.format(created_at=utc_now_iso()), encoding="utf-8")

    rel = out.resolve().relative_to(workspace).as_posix()
    artifact = register_artifact(
        ledger=ledger,
        project={"id": project_id, "repo_path": str(workspace)},
        kind="constraints_yaml",
        relative_path=rel,
        meta={},
    )
    return {"path": str(out), "artifact": artifact}


def load_constraints(
    *,
    ledger: Ledger,
    project_id: str,
    path: str = "constraints.yaml",
) -> Dict[str, Any]:
    project = ledger.get_project(project_id)
    workspace = Path(project["repo_path"]).resolve()
    p = Path(path)
    if not p.is_absolute():
        p = (workspace / p).resolve()
    if not p.exists():
        return {}
    if yaml is None:  # pragma: no cover
        raise SystemExit("PyYAML is required to read constraints.yaml (pip install pyyaml).")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"Invalid constraints file (expected a YAML object at top-level): {p}")
    return raw


def get_compute_config(constraints: Dict[str, Any]) -> Dict[str, Any]:
    compute = constraints.get("compute") or {}
    if not isinstance(compute, dict):
        compute = {}

    backend = str(compute.get("backend") or "local").strip().lower()
    if backend not in {"local", "slurm"}:
        raise SystemExit("constraints.yaml compute.backend must be one of: local, slurm")

    local = compute.get("local") or {}
    if not isinstance(local, dict):
        local = {}
    slurm = compute.get("slurm") or {}
    if not isinstance(slurm, dict):
        slurm = {}

    max_parallel = local.get("max_parallel")
    if max_parallel is None:
        max_parallel = compute.get("max_parallel")
    try:
        max_parallel_int = int(max_parallel) if max_parallel is not None else 1
    except (TypeError, ValueError):
        max_parallel_int = 1
    if max_parallel_int < 1:
        max_parallel_int = 1

    extra_sbatch_args = slurm.get("extra_sbatch_args") or []
    if not isinstance(extra_sbatch_args, list):
        extra_sbatch_args = []

    return {
        "backend": backend,
        "local": {"max_parallel": max_parallel_int},
        "slurm": {
            "sbatch": str(slurm.get("sbatch") or "sbatch"),
            "sacct": str(slurm.get("sacct") or "sacct"),
            "squeue": str(slurm.get("squeue") or "squeue"),
            "extra_sbatch_args": [str(x) for x in extra_sbatch_args if x],
        },
    }


def get_data_mounts(constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = constraints.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    mounts = data.get("mounts") or []
    if not isinstance(mounts, list):
        mounts = []

    out: List[Dict[str, Any]] = []
    for m in mounts:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        path = str(m.get("path") or "").strip()
        if not name or not path:
            continue
        out.append(
            {
                "name": name,
                "path": path,
                "read_only": bool(m.get("read_only", True)),
                "required": bool(m.get("required", True)),
                "notes": str(m.get("notes") or ""),
            }
        )
    return out


def get_databases_inventory(constraints: Dict[str, Any]) -> Dict[str, Any]:
    db = constraints.get("databases") or {}
    if not isinstance(db, dict):
        db = {}

    root = str(db.get("root") or "").strip()
    items = db.get("items") or []
    if not isinstance(items, list):
        items = []

    out_items: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "").strip()
        path = str(it.get("path") or "").strip()
        if not name or not path:
            continue
        out_items.append(
            {
                "name": name,
                "path": path,
                "license": str(it.get("license") or ""),
                "version": str(it.get("version") or ""),
                "auto_download": bool(it.get("auto_download", False)),
            }
        )

    return {"root": root, "items": out_items}
