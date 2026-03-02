from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from resorch.artifacts import register_artifact
from resorch.ledger import Ledger
from resorch.tasks import create_task


def _today_ymd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _render_review_packet_md(
    *,
    project_id: str,
    stage: str,
    targets: List[str],
    questions: List[str],
    created_iso: str,
    rubric_path: Optional[str],
    rubric_text: Optional[str],
) -> str:
    lines: List[str] = []
    lines.append("# Review Packet\n\n")
    lines.append("## Context\n")
    lines.append(f"- Project: {project_id}\n")
    lines.append(f"- Stage: {stage}\n")
    lines.append("- Goal (1 sentence):\n")
    lines.append("- Current claim(s):\n")
    lines.append("- What changed since last review:\n")
    lines.append(f"\n_Created: {created_iso}_\n\n")

    lines.append("## What to review\n")
    lines.append("- Target artifacts (paths):\n")
    for t in targets:
        lines.append(f"  - {t}\n")
    if rubric_path:
        lines.append(f"- Rubric / Prompt: {rubric_path}\n")
    lines.append("- Questions for reviewer:\n")
    for i, q in enumerate(questions, start=1):
        lines.append(f"  {i}) {q}\n")
    lines.append("\n")

    if rubric_text:
        lines.append("## Rubric / Prompt (embedded)\n")
        lines.append("```text\n")
        lines.append(rubric_text.rstrip() + "\n")
        lines.append("```\n\n")

    lines.append("## Constraints\n")
    lines.append("- Time budget:\n")
    lines.append("- Allowed changes (yes/no):\n")
    lines.append("- Non-goals:\n\n")

    lines.append("## Known risks / open issues\n- \n\n")
    lines.append("## Evidence map (claim -> evidence)\n- Claim A:\n  - evidence:\n")
    return "".join(lines)


def _load_rubric_text(*, rubric: str, repo_root: Path, workspace: Path) -> str:
    p = Path(str(rubric))
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(workspace / p)
        candidates.append(repo_root / p)
    for c in candidates:
        if c.exists():
            return c.read_text(encoding="utf-8")
    raise SystemExit(f"rubric file not found. tried: {', '.join(str(c) for c in candidates)}")


def write_review_request(
    *,
    ledger: Ledger,
    project: Dict[str, Any],
    stage: str,
    mode: str,
    targets: List[str],
    questions: List[str],
    rubric: Optional[str],
    time_budget_minutes: Optional[int],
) -> Dict[str, Any]:
    workspace = Path(project["repo_path"])
    reviews_dir = workspace / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)

    uid = uuid4().hex[:6]
    prefix = f"REQ-{stage}-{_today_ymd()}-{uid}"
    packet_path = reviews_dir / f"{prefix}.md"
    request_json_path = reviews_dir / f"{prefix}.request.json"

    created_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    request_obj: Dict[str, Any] = {
        "project_id": project["id"],
        "stage": stage,
        "mode": mode,
        "targets": targets,
        "questions": questions,
        "rubric": rubric,
        "time_budget_minutes": time_budget_minutes,
    }

    rubric_text = None
    if rubric:
        rubric_text = _load_rubric_text(rubric=str(rubric), repo_root=ledger.paths.root, workspace=workspace)

    request_json_path.write_text(json.dumps(request_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    packet_path.write_text(
        _render_review_packet_md(
            project_id=project["id"],
            stage=stage,
            targets=targets,
            questions=questions,
            created_iso=created_iso,
            rubric_path=rubric,
            rubric_text=rubric_text,
        ),
        encoding="utf-8",
    )

    request_artifact = register_artifact(
        ledger=ledger,
        project=project,
        kind="review_request_json",
        relative_path=str(Path("reviews") / request_json_path.name),
        meta={},
    )
    packet_artifact = register_artifact(
        ledger=ledger,
        project=project,
        kind="review_packet_md",
        relative_path=str(Path("reviews") / packet_path.name),
        meta={},
    )

    return {
        "packet_path": str(packet_path),
        "request_json_path": str(request_json_path),
        "artifacts": [packet_artifact, request_artifact],
    }


def _update_finding_recurrence(reviews_dir: Path) -> None:
    """Build reviews/finding_recurrence.md from all RESP files.

    Scans every RESP-*.json in ``reviews_dir``, groups major/blocker
    findings by category, and writes a human-readable recurrence tracker.
    Both Planner and Reviewer can use this to identify inherent limitations
    that keep recurring without resolution.

    Counts are based on **distinct review files** containing findings in each
    category rather than raw finding counts, preventing inflation from
    multiple findings per review.  Exact-duplicate messages are collapsed.
    """
    resp_files = sorted(reviews_dir.glob("RESP-*.json"))
    # category -> [(source, message, resolvability)]
    by_category: Dict[str, List[tuple]] = {}
    # category -> set of RESP filenames that contain findings
    sources_by_cat: Dict[str, set] = {}
    for rp in resp_files:
        try:
            data = json.loads(rp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for f in data.get("findings") or []:
            if not isinstance(f, dict):
                continue
            sev = str(f.get("severity") or "").lower()
            if sev not in ("major", "blocker"):
                continue
            cat = str(f.get("category") or "other")
            msg = str(f.get("message") or "").strip()[:200]
            resolv = str(f.get("resolvability") or "unset")
            by_category.setdefault(cat, []).append((rp.name, msg, resolv))
            sources_by_cat.setdefault(cat, set()).add(rp.name)

    n_resp = len(resp_files)
    lines = ["# Finding Recurrence Tracker\n\n"]
    lines.append(
        f"Major/blocker findings across {n_resp} review files, grouped by category.\n"
    )
    lines.append(
        "Review count = distinct RESP files containing findings in that category.\n"
    )
    lines.append(
        "Categories flagged in >50% of reviews are likely inherent_limitation.\n\n"
    )
    for cat in sorted(by_category.keys()):
        entries = by_category[cat]
        n_reviews = len(sources_by_cat[cat])
        # Deduplicate exact-match messages, keep first source
        seen_msgs: set = set()
        unique_entries: list = []
        for source, msg, resolv in entries:
            if msg not in seen_msgs:
                seen_msgs.add(msg)
                unique_entries.append((source, msg, resolv))
        lines.append(
            f"## {cat} ({n_reviews}/{n_resp} reviews, "
            f"{len(unique_entries)} unique findings)\n"
        )
        for source, msg, resolv in unique_entries:
            lines.append(f"- [{resolv}] {msg} ({source})\n")
        threshold = max(4, n_resp // 2)
        if n_reviews >= threshold:
            pct = round(100 * n_reviews / n_resp) if n_resp else 0
            lines.append(
                f"\n**WARNING: {cat} flagged in {pct}% of reviews ({n_reviews}/{n_resp})"
                " — strongly consider classifying as inherent_limitation**\n"
            )
        lines.append("\n")

    if not by_category:
        lines.append("No major/blocker findings recorded yet.\n")

    (reviews_dir / "finding_recurrence.md").write_text("".join(lines), encoding="utf-8")


def ingest_review_result(*, ledger: Ledger, result_path: Path, reviews_rel: str = "reviews", create_fix_tasks: str = "all") -> Dict[str, Any]:
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    project_id = raw.get("project_id")
    stage = raw.get("stage")
    reviewer = raw.get("reviewer")
    if not project_id or not stage or not reviewer:
        raise SystemExit("review_result.json must include project_id, stage, reviewer.")

    project = ledger.get_project(project_id)
    project_meta = json.loads(project.pop("meta_json") or "{}")
    project["meta"] = project_meta

    workspace = Path(project["repo_path"]).resolve()
    rel_dir = Path(str(reviews_rel or "reviews"))
    if rel_dir.is_absolute():
        raise SystemExit("reviews_rel must be a workspace-relative path.")
    reviews_dir = (workspace / rel_dir).resolve()
    try:
        reviews_dir.relative_to(workspace)
    except ValueError as e:
        raise SystemExit("reviews_rel must stay within the workspace.") from e
    reviews_dir.mkdir(parents=True, exist_ok=True)

    # If result_path is already inside reviews_dir (e.g. written by _run_*_job),
    # reuse it directly instead of creating a duplicate copy with a new UUID.
    try:
        result_path.resolve().relative_to(reviews_dir)
        dest_path = result_path
    except ValueError:
        uid = uuid4().hex[:6]
        dest_name = f"RESP-{stage}-{_today_ymd()}-{uid}-{reviewer}.json"
        dest_path = reviews_dir / dest_name
        if result_path.resolve() != dest_path.resolve():
            shutil.copy2(result_path, dest_path)

    review_id = uuid4().hex
    review = ledger.insert_review(
        review_id=review_id,
        project_id=project_id,
        stage=stage,
        reviewer=reviewer,
        rubric={},
        findings=raw,
    )
    review["rubric"] = json.loads(review.pop("rubric_json") or "{}")
    review["findings"] = json.loads(review.pop("findings_json") or "{}")

    artifact = register_artifact(
        ledger=ledger,
        project=project,
        kind="review_result_json",
        relative_path=str(rel_dir / dest_path.name),
        meta={"source_path": str(result_path)},
    )

    # Write a short human-readable summary that the planner/reviewer can consume.
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    findings = raw.get("findings") or []
    severity_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    resolvability_counts: Dict[str, int] = {}
    blockers: List[str] = []
    limitations: List[str] = []
    pivots: List[str] = []
    if isinstance(findings, list):
        for f in findings:
            if not isinstance(f, dict):
                continue
            sev = str(f.get("severity") or "unknown")
            cat = str(f.get("category") or "unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            category_counts[cat] = category_counts.get(cat, 0) + 1
            resolv = f.get("resolvability")
            if resolv:
                resolvability_counts[str(resolv)] = resolvability_counts.get(str(resolv), 0) + 1
            if sev == "blocker":
                msg = str(f.get("message") or "").strip()
                if msg:
                    blockers.append(msg)
            if resolv == "inherent_limitation" and sev in ("major", "blocker"):
                msg = str(f.get("message") or "").strip()
                if msg:
                    limitations.append(msg)
            if resolv == "requires_pivot" and sev in ("major", "blocker"):
                msg = str(f.get("message") or "").strip()
                if msg:
                    pivots.append(msg)

    summary_lines: List[str] = []
    summary_lines.append("# Last Review Summary\n\n")
    summary_lines.append(f"- Date: {now_iso}\n")
    summary_lines.append(f"- Stage: {stage}\n")
    summary_lines.append(f"- Reviewer: {reviewer}\n")
    summary_lines.append(f"- Recommendation: {raw.get('recommendation')}\n")
    summary_lines.append(f"- Source: {str(rel_dir / dest_path.name)}\n\n")

    overall = raw.get("overall")
    if isinstance(overall, str) and overall.strip():
        summary_lines.append("## Overall\n")
        summary_lines.append(overall.strip() + "\n\n")

    summary_lines.append("## Findings\n")
    if severity_counts:
        summary_lines.append("- By severity:\n")
        for k in sorted(severity_counts.keys()):
            summary_lines.append(f"  - {k}: {severity_counts[k]}\n")
    if category_counts:
        summary_lines.append("- By category:\n")
        for k in sorted(category_counts.keys()):
            summary_lines.append(f"  - {k}: {category_counts[k]}\n")
    if resolvability_counts:
        summary_lines.append("- By resolvability:\n")
        for k in sorted(resolvability_counts.keys()):
            summary_lines.append(f"  - {k}: {resolvability_counts[k]}\n")
    summary_lines.append("\n")

    if blockers:
        summary_lines.append("### Blockers\n")
        for idx, msg in enumerate(blockers[:10], start=1):
            summary_lines.append(f"{idx}. {msg}\n")
        summary_lines.append("\n")

    if limitations:
        summary_lines.append("### Inherent Limitations (do NOT attempt to fix — acknowledge in Discussion)\n")
        for idx, msg in enumerate(limitations[:10], start=1):
            summary_lines.append(f"{idx}. {msg}\n")
        summary_lines.append("\n")

    if pivots:
        summary_lines.append("### Requires Pivot (fundamental design change needed — consult PI)\n")
        for idx, msg in enumerate(pivots[:10], start=1):
            summary_lines.append(f"{idx}. {msg}\n")
        summary_lines.append("\n")

    summary_path = reviews_dir / "last_review_summary.md"
    summary_path.write_text("".join(summary_lines), encoding="utf-8")

    # Update the finding recurrence tracker across all reviews.
    _update_finding_recurrence(reviews_dir)

    summary_artifact = register_artifact(
        ledger=ledger,
        project=project,
        kind="review_summary_md",
        relative_path=str(rel_dir / summary_path.name),
        meta={"source": str(rel_dir / dest_path.name)},
    )

    # create_fix_tasks controls which findings become review_fix tasks:
    #   "all"           – every finding (legacy behaviour)
    #   "major_blocker" – only severity in {major, blocker}
    #   "none"          – skip task creation entirely
    created_tasks = []
    if create_fix_tasks != "none":
        findings = raw.get("findings") or []
        if isinstance(findings, list):
            for idx, f in enumerate(findings):
                if not isinstance(f, dict):
                    continue
                if create_fix_tasks == "major_blocker":
                    if str(f.get("severity") or "").lower() not in ("major", "blocker"):
                        continue
                spec = {
                    "source": "review",
                    "review_id": review_id,
                    "finding_index": idx,
                    "stage": stage,
                    "severity": f.get("severity"),
                    "category": f.get("category"),
                    "message": f.get("message"),
                    "target_paths": f.get("target_paths") or [],
                    "suggested_fix": f.get("suggested_fix"),
                    "estimated_effort": f.get("estimated_effort"),
                    "evidence": f.get("evidence"),
                }
                created_tasks.append(
                    create_task(
                        ledger=ledger,
                        project_id=project_id,
                        task_type="review_fix",
                        spec=spec,
                    )
                )

    return {
        "review": review,
        "artifact": artifact,
        "summary_artifact": summary_artifact,
        "stored_path": str(dest_path),
        "tasks_created": created_tasks,
    }
