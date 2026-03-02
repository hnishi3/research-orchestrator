from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.ledger import Ledger
from resorch.utils import slugify, utc_now_iso

log = logging.getLogger(__name__)


_PROJECT_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


def _ensure_standard_workspace_layout(workspace: Path) -> None:
    (workspace / "notes").mkdir(parents=True, exist_ok=True)
    (workspace / "notes" / "autopilot").mkdir(parents=True, exist_ok=True)
    (workspace / "ideas").mkdir(parents=True, exist_ok=True)
    (workspace / "evidence").mkdir(parents=True, exist_ok=True)
    (workspace / "claims").mkdir(parents=True, exist_ok=True)
    (workspace / "runs" / "smoke").mkdir(parents=True, exist_ok=True)
    (workspace / "runs" / "agent").mkdir(parents=True, exist_ok=True)
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "experiments").mkdir(parents=True, exist_ok=True)
    (workspace / "results" / "raw").mkdir(parents=True, exist_ok=True)
    (workspace / "results" / "fig").mkdir(parents=True, exist_ok=True)
    (workspace / "paper").mkdir(parents=True, exist_ok=True)
    (workspace / "reviews").mkdir(parents=True, exist_ok=True)
    (workspace / "jobs").mkdir(parents=True, exist_ok=True)

    problem_md = workspace / "notes" / "problem.md"
    if not problem_md.exists():
        problem_md.write_text(
            "\n".join(
                [
                    "# Problem",
                    "",
                    "- Question:",
                    "- Hypothesis:",
                    "- Evaluation metrics:",
                    "- Constraints:",
                    "",
                    f"_Created: {utc_now_iso()}_",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    method_md = workspace / "notes" / "method.md"
    if not method_md.exists():
        method_md.write_text(
            "\n".join(
                [
                    "# Method",
                    "",
                    "- Proposed approach (1 paragraph):",
                    "- Baselines:",
                    "- Ablations:",
                    "- Key implementation details:",
                    "- Experimental setup:",
                    "",
                    f"_Created: {utc_now_iso()}_",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    scoreboard = workspace / "results" / "scoreboard.json"
    if not scoreboard.exists():
        scoreboard.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "updated_at": utc_now_iso(),
                    "primary_metric": {
                        "name": "",
                        "direction": "maximize",
                        "current": None,
                        "best": None,
                        "baseline": None,
                        "delta_vs_baseline": None,
                    },
                    "metrics": {},
                    "runs": [],
                    "notes": "",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    analysis_digest_md = workspace / "notes" / "analysis_digest.md"
    if not analysis_digest_md.exists():
        analysis_digest_md.write_text(
            "\n".join(
                [
                    "# Analysis Digest",
                    "",
                    "A rolling log of results, what we learned, and what to try next.",
                    "",
                    "## Latest",
                    "- (Fill after running experiments; link to `results/` artifacts.)",
                    "",
                    "## Next actions (top 3)",
                    "- ",
                    "- ",
                    "- ",
                    "",
                    "## Notes",
                    "- ",
                    "",
                    f"_Created: {utc_now_iso()}_",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    manuscript_md = workspace / "paper" / "manuscript.md"
    if not manuscript_md.exists():
        manuscript_md.write_text(
            "\n".join(
                [
                    "# Title",
                    "",
                    "## Abstract",
                    "",
                    "## 1. Introduction",
                    "",
                    "## 2. Related Work",
                    "",
                    "## 3. Method",
                    "",
                    "## 4. Experiments",
                    "",
                    "## 5. Results",
                    "",
                    "<!-- Figures: reference figures inline as ![Caption](../results/fig/filename.png) -->",
                    "<!-- They will be placed at that location in the compiled PDF. -->",
                    "",
                    "## 6. Discussion",
                    "",
                    "## 7. Conclusion",
                    "",
                    "## References",
                    "<!-- List references with DOIs: N. Author et al. *Journal* (Year). DOI:10.xxxx/yyyy -->",
                    "<!-- All DOIs and PMIDs must be verifiable — do not fabricate references. -->",
                    "",
                    f"_Created: {utc_now_iso()}_",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def _pick_project_id(ledger: Ledger, requested_id: Optional[str], title: str) -> str:
    if requested_id:
        pid = requested_id.strip()
        if not _PROJECT_ID_RE.match(pid):
            raise SystemExit(
                "Invalid project id. Use 1-64 chars: letters/numbers plus . _ - (must start with alnum)."
            )
        return pid

    base = slugify(title)
    pid = base
    existing = {p["id"] for p in ledger.list_projects()}
    i = 2
    while pid in existing:
        pid = f"{base}-{i}"
        i += 1
    return pid


def create_project(
    *,
    ledger: Ledger,
    project_id: Optional[str],
    title: str,
    domain: str,
    stage: str,
    git_init: bool,
    idea_id: Optional[str] = None,
) -> Dict[str, Any]:
    pid = _pick_project_id(ledger, project_id, title)

    workspace = ledger.paths.workspaces_dir / pid
    if workspace.exists():
        raise SystemExit(f"Workspace already exists: {workspace}")
    workspace.mkdir(parents=True, exist_ok=False)

    _ensure_standard_workspace_layout(workspace)

    if git_init:
        subprocess.run(
            ["git", "init", "-b", "main", "--quiet"],
            cwd=str(workspace),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        (workspace / ".gitignore").write_text(
            "\n".join(
                [
                    "# OS",
                    ".DS_Store",
                    "Thumbs.db",
                    "",
                    "# Secrets",
                    ".env",
                    ".env.*",
                    "",
                    "# Python",
                    "__pycache__/",
                    "*.py[cod]",
                    "*.so",
                    ".venv*/",
                    ".pytest_cache/",
                    ".mypy_cache/",
                    ".ruff_cache/",
                    "",
                    "# Jupyter",
                    ".ipynb_checkpoints/",
                    "",
                    "# Logs",
                    "*.log",
                    "",
                    "# Large data (raw archives, databases)",
                    "*.tar.gz",
                    "*.tar.bz2",
                    "*.zip",
                    "*.gz",
                    "!*.json.gz",
                    "",
                    "# Sequence data",
                    "*.fastq",
                    "*.fastq.gz",
                    "*.fq.gz",
                    "*.bam",
                    "*.bam.bai",
                    "*.cram",
                    "*.sam",
                    "*.vcf.gz",
                    "*.bcf",
                    "",
                    "# Structure biology",
                    "data/pdb/",
                    "*.pdb",
                    "*.mmcif",
                    "*.cif",
                    "*.ent",
                    "*.a3m",
                    "",
                    "# ML checkpoints & embeddings",
                    "*.pt",
                    "*.pth",
                    "*.ckpt",
                    "*.pkl",
                    "*.h5",
                    "*.hdf5",
                    "*.npy",
                    "*.npz",
                    "",
                    "# Workflow engines",
                    ".snakemake/",
                    ".nextflow/",
                    ".nextflow.log*",
                    "work/",
                    "",
                    "# Workspace temporaries",
                    "runs/tmp/",
                    "results/raw/",
                    "results/mmseqs2/tmp/",
                    "data/cache/",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    meta: Dict[str, Any] = {"git_init": git_init}
    if idea_id is not None:
        meta["idea_id"] = idea_id

    project = ledger.insert_project(
        project_id=pid,
        title=title,
        domain=domain,
        stage=stage,
        repo_path=str(workspace),
        meta=meta,
    )
    project["meta"] = json.loads(project.pop("meta_json") or "{}")
    return project


def list_projects(ledger: Ledger) -> List[Dict[str, Any]]:
    projects = ledger.list_projects()
    for p in projects:
        p["meta"] = json.loads(p.pop("meta_json") or "{}")
    return projects


def get_project(ledger: Ledger, project_id: str) -> Dict[str, Any]:
    project = ledger.get_project(project_id)
    project["meta"] = json.loads(project.pop("meta_json") or "{}")
    return project


def set_project_stage(ledger: Ledger, project_id: str, stage: str) -> Dict[str, Any]:
    project = ledger.update_project_stage(project_id, stage)
    project["meta"] = json.loads(project.pop("meta_json") or "{}")

    stage_norm = str(stage or "").strip().lower()
    if stage_norm in {"done", "parked", "rejected"}:
        try:
            # Lazy import avoids a projects <-> playbook_extractor import cycle.
            from resorch.playbook_extractor import extract_and_save

            extracted = extract_and_save(ledger=ledger, project_id=project_id, mode="compact")
            log.info(
                "Playbook extraction hook succeeded for project=%s stage=%s entry_id=%s yaml_path=%s",
                project_id,
                stage,
                extracted.get("entry_id"),
                extracted.get("yaml_path"),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Playbook extraction hook failed for project=%s stage=%s: %s",
                project_id,
                stage,
                exc,
            )
    return project


# ---------------------------------------------------------------------------
# Successor / continuation projects
# ---------------------------------------------------------------------------

# Directories to inherit (relative to workspace root).
_INHERIT_DIRS = ("data", "src", "configs")


def _generate_predecessor_summary(pred_workspace: Path) -> str:
    """Build a compact summary of the predecessor project for Planner context.

    Reads analysis_digest.md, scoreboard.json, and problem.md from the
    predecessor workspace and produces a Markdown document small enough
    to fit within the Planner's reference-material budget.
    """
    sections: List[str] = ["# Predecessor Project Summary\n"]

    # --- problem.md (first 40 lines) ---
    problem = pred_workspace / "notes" / "problem.md"
    if problem.exists():
        lines = problem.read_text(encoding="utf-8").splitlines()[:40]
        sections.append("## Problem\n" + "\n".join(lines) + "\n")

    # --- method.md (first 60 lines) ---
    method = pred_workspace / "notes" / "method.md"
    if method.exists():
        lines = method.read_text(encoding="utf-8").splitlines()[:60]
        sections.append("## Method\n" + "\n".join(lines) + "\n")

    # --- scoreboard.json (compact) ---
    scoreboard = pred_workspace / "results" / "scoreboard.json"
    if scoreboard.exists():
        try:
            sb = json.loads(scoreboard.read_text(encoding="utf-8"))
            pm = sb.get("primary_metric") or {}
            metric_line = (
                f"- Primary metric: {pm.get('name', '?')} = {pm.get('best', '?')} "
                f"(direction: {pm.get('direction', '?')})"
            )
            runs = sb.get("runs") or []
            runs_summary = f"- Total runs: {len(runs)}"
            sections.append(
                "## Scoreboard\n" + metric_line + "\n" + runs_summary + "\n"
            )
        except (json.JSONDecodeError, KeyError):
            pass

    # --- analysis_digest.md (first 80 lines) ---
    digest = pred_workspace / "notes" / "analysis_digest.md"
    if digest.exists():
        lines = digest.read_text(encoding="utf-8").splitlines()[:80]
        sections.append("## Analysis Digest (truncated)\n" + "\n".join(lines) + "\n")

    sections.append(f"\n_Generated: {utc_now_iso()}_\n")
    return "\n".join(sections)


def create_successor_project(
    *,
    ledger: Ledger,
    predecessor_id: str,
    project_id: Optional[str] = None,
    title: Optional[str] = None,
    domain: Optional[str] = None,
    stage: str = "intake",
    git_init: bool = True,
    inherit: Optional[List[str]] = None,
    idea_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new project that continues from *predecessor_id*.

    The successor gets:
    - A fresh workspace with standard template files.
    - Symlinks to inherited directories (data, src, configs by default).
    - ``notes/predecessor_summary.md`` — compressed context for Planner.
    - ``meta.predecessor`` recording the lineage.
    """
    pred = get_project(ledger, predecessor_id)
    pred_workspace = Path(pred["repo_path"])
    if not pred_workspace.is_dir():
        raise SystemExit(f"Predecessor workspace not found: {pred_workspace}")

    # Defaults from predecessor.
    new_title = title or f"{pred['title']} (continuation)"
    new_domain = domain if domain is not None else pred.get("domain", "")
    pred_meta = pred.get("meta") if isinstance(pred.get("meta"), dict) else {}
    inherited_idea_id = pred_meta.get("idea_id") if isinstance(pred_meta, dict) else None
    new_idea_id = idea_id if idea_id is not None else inherited_idea_id

    # Create the new project via the normal path.
    new_project = create_project(
        ledger=ledger,
        project_id=project_id,
        title=new_title,
        domain=new_domain,
        stage=stage,
        git_init=git_init,
        idea_id=new_idea_id,
    )
    new_workspace = Path(new_project["repo_path"])

    # --- Inherit directories via symlink ---
    dirs_to_inherit = inherit if inherit is not None else list(_INHERIT_DIRS)
    inherited: List[str] = []
    for dirname in dirs_to_inherit:
        # Validate: no absolute paths, no '..' traversal
        dirname = str(dirname).strip()
        if not dirname or Path(dirname).is_absolute() or ".." in Path(dirname).parts:
            raise SystemExit(
                f"inherit directory must be a safe relative name (got: {dirname!r})"
            )
        src = (pred_workspace / dirname).resolve()
        # Ensure resolved source stays within predecessor workspace
        try:
            src.relative_to(pred_workspace.resolve())
        except ValueError:
            raise SystemExit(
                f"inherit directory escapes predecessor workspace: {dirname!r}"
            )
        if not src.is_dir():
            continue
        dst = new_workspace / dirname
        if dst.exists():
            # Template might have created an empty dir — remove it.
            if dst.is_dir() and not any(dst.iterdir()):
                dst.rmdir()
            else:
                continue
        dst.symlink_to(src.resolve())
        inherited.append(dirname)

    # --- Generate predecessor_summary.md ---
    summary_text = _generate_predecessor_summary(pred_workspace)
    summary_path = new_workspace / "notes" / "predecessor_summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")

    # --- Update meta with predecessor info ---
    meta = new_project.get("meta", {})
    meta["predecessor"] = {
        "id": predecessor_id,
        "title": pred["title"],
        "inherited_dirs": inherited,
        "created_at": utc_now_iso(),
    }
    # Write updated meta back to the database.
    ledger._exec(
        "UPDATE projects SET meta_json = ?, updated_at = ? WHERE id = ?",
        (json.dumps(meta, ensure_ascii=False), utc_now_iso(), new_project["id"]),
    )
    ledger._maybe_commit()

    # Re-fetch to return fresh data.
    result = get_project(ledger, new_project["id"])
    result["inherited_dirs"] = inherited
    return result
