from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from resorch.artifacts import put_artifact
from resorch.autopilot_config import ReviewRecommendation
from resorch.autopilot_pivot import _nested_get, _as_float
from resorch.ledger import Ledger
from resorch.utils import read_text, utc_now_iso

log = logging.getLogger(__name__)


def _as_compact_task_ref(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": task.get("id"),
        "type": task.get("type"),
        "status": task.get("status"),
        "updated_at": task.get("updated_at"),
    }


def summarize_pre_exec_reviews(tasks_ran: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate pre-exec review gate results across all tasks in a step."""
    all_reviews: List[Dict[str, Any]] = []
    for t in tasks_ran:
        if not isinstance(t, dict):
            continue
        reviews = t.get("pre_exec_review_results")
        if isinstance(reviews, list):
            all_reviews.extend(reviews)
    if not all_reviews:
        return {"total": 0, "pass": 0, "fail": 0, "reviews": []}
    pass_count = sum(1 for r in all_reviews if r.get("verdict") == "PASS")
    fail_count = sum(1 for r in all_reviews if r.get("verdict") == "FAIL")
    return {
        "total": len(all_reviews),
        "pass": pass_count,
        "fail": fail_count,
        "reviews": all_reviews,
    }


def summarize_codex_exec_statuses(tasks_ran: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Best-effort summary of codex_exec statuses for stall detection."""
    summary: Dict[str, Any] = {
        "total": 0,
        "blocked": 0,
        "success": 0,
        "failed": 0,
        "rate_limited": 0,
        "other": 0,
        "all_blocked": False,
    }
    try:
        codex_tasks = [t for t in tasks_ran if isinstance(t, dict) and str(t.get("type") or "") == "codex_exec"]
        summary["total"] = len(codex_tasks)
        if not codex_tasks:
            return summary

        for t in codex_tasks:
            st = str(t.get("status") or "")
            if st == "blocked":
                summary["blocked"] += 1
            elif st == "success":
                summary["success"] += 1
            elif st == "failed":
                summary["failed"] += 1
            elif st == "rate_limited":
                summary["rate_limited"] += 1
            else:
                summary["other"] += 1

        summary["all_blocked"] = bool(summary["total"] > 0 and summary["blocked"] == summary["total"])
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to summarize codex_exec statuses; continuing: %s", exc)
    return summary


def _update_pdca_digests(
    *,
    ledger: Ledger,
    project: Dict[str, Any],
    iteration: int,
    started_at: str,
    plan_artifact_path: str,
    tasks_created: List[Dict[str, Any]],
    tasks_ran: List[Dict[str, Any]],
    git_change_summary: Dict[str, Any],
    review_recommendation: ReviewRecommendation,
) -> None:
    workspace = Path(project["repo_path"]).resolve()

    # 1) Machine-readable scoreboard.
    scoreboard_rel = "results/scoreboard.json"
    scoreboard_path = (workspace / scoreboard_rel).resolve()
    _scoreboard_parse_failed = False
    try:
        raw = read_text(scoreboard_path)
        scoreboard = json.loads(raw) if raw.strip() else {}
    except OSError:
        scoreboard = {}  # File doesn't exist yet — safe to initialize.
    except json.JSONDecodeError as exc:
        log.warning("Scoreboard JSON is corrupted (%s); skipping update to avoid data loss.", exc)
        _scoreboard_parse_failed = True
        scoreboard = {}
    if not isinstance(scoreboard, dict):
        log.warning("Scoreboard is not a dict (%s); skipping update to avoid data loss.", type(scoreboard).__name__)
        _scoreboard_parse_failed = True
        scoreboard = {}

    scoreboard.setdefault("schema_version", 3)
    if not isinstance(scoreboard.get("schema_version"), int):
        scoreboard["schema_version"] = 3
    # Bump to v3 if still on older schema (v3 separates iterations from runs).
    if scoreboard["schema_version"] < 3:
        scoreboard["schema_version"] = 3
    scoreboard["updated_at"] = utc_now_iso()
    pm = scoreboard.get("primary_metric")
    pm_snapshot = pm if isinstance(pm, dict) else None
    metrics = scoreboard.get("metrics")
    metrics_snapshot = metrics if isinstance(metrics, (dict, list)) else None
    runs = scoreboard.get("runs")
    if not isinstance(runs, list):
        runs = []

    changed_paths = git_change_summary.get("changed_paths") if isinstance(git_change_summary, dict) else None
    changed_paths_list = [str(x) for x in changed_paths if x] if isinstance(changed_paths, list) else []
    max_paths = 20

    # Orchestrator iteration metadata goes to "iterations", NOT "runs".
    # "runs" is reserved for research experiment data (written by summary_ingest
    # and Codex tasks).  Mixing them caused Bug #33 (schema mixing).
    iterations = scoreboard.get("iterations")
    if not isinstance(iterations, list):
        iterations = []
    iterations.append(
        {
            "ts": utc_now_iso(),
            "iteration": int(iteration),
            "stage": str(project.get("stage") or ""),
            "primary_metric": pm_snapshot,
            "metrics": metrics_snapshot,
            "plan_artifact_path": str(plan_artifact_path),
            "tasks_created": [_as_compact_task_ref(t) for t in tasks_created],
            "tasks_ran": [_as_compact_task_ref(t) for t in tasks_ran],
            "git": {
                "changed_files": int(git_change_summary.get("changed_files", 0)),
                "changed_lines": int(git_change_summary.get("changed_lines", 0)),
                "changed_paths": changed_paths_list[:max_paths],
                "changed_paths_truncated": len(changed_paths_list) > max_paths,
            },
            "review_recommendation": {
                "level": review_recommendation.level,
                "reasons": list(review_recommendation.reasons),
            },
            "notes": "",
        }
    )
    scoreboard["iterations"] = iterations[-200:]  # keep bounded

    # Migrate any orchestrator entries that were previously mixed into "runs"
    # (Bug #33 legacy data).  Entries with "tasks_created" or "tasks_ran" keys
    # are orchestrator metadata; move them to "legacy_runs".
    if runs:
        clean_runs: List[Dict[str, Any]] = []
        legacy: List[Dict[str, Any]] = []
        for r in runs:
            if isinstance(r, dict) and ("tasks_created" in r or "tasks_ran" in r):
                legacy.append(r)
            else:
                clean_runs.append(r)
        if legacy:
            existing_legacy = scoreboard.get("legacy_runs")
            if not isinstance(existing_legacy, list):
                existing_legacy = []
            existing_legacy.extend(legacy)
            scoreboard["legacy_runs"] = existing_legacy[-200:]
        scoreboard["runs"] = clean_runs[-200:]

    if not _scoreboard_parse_failed:
        put_artifact(
            ledger=ledger,
            project=project,
            relative_path=scoreboard_rel,
            content=json.dumps(scoreboard, ensure_ascii=False, indent=2) + "\n",
            mode="overwrite",
            kind="scoreboard_json",
        )

    # 2) Human-readable analysis digest (append-only).
    digest_rel = "notes/analysis_digest.md"
    digest_lines: List[str] = []
    digest_lines.append(f"\n## Autopilot Iteration {iteration:03d}\n")
    digest_lines.append(f"- Started: {started_at}\n")
    digest_lines.append(f"- Plan: `{plan_artifact_path}`\n")
    digest_lines.append(f"- Tasks ran: {len(tasks_ran)} (created: {len(tasks_created)})\n")
    pm_current = _as_float(_nested_get(scoreboard, "primary_metric.current.mean"))
    if pm_current is None:
        pm_current = _as_float(_nested_get(scoreboard, "primary_metric.current"))
    pm_best = _as_float(_nested_get(scoreboard, "primary_metric.best.mean"))
    if pm_best is None:
        pm_best = _as_float(_nested_get(scoreboard, "primary_metric.best"))
    pm_baseline = _as_float(_nested_get(scoreboard, "primary_metric.baseline.mean"))
    if pm_baseline is None:
        pm_baseline = _as_float(_nested_get(scoreboard, "primary_metric.baseline"))
    pm_delta = _as_float(_nested_get(scoreboard, "primary_metric.delta_vs_baseline"))
    pm_name = str(_nested_get(scoreboard, "primary_metric.name") or "").strip()
    pm_direction = str(_nested_get(scoreboard, "primary_metric.direction") or "").strip()
    if pm_current is None:
        digest_lines.append(
            "- Primary metric: (not set) — update `results/scoreboard.json.primary_metric.current.mean` to enable pivot policy\n"
        )
    else:
        parts: List[str] = []
        if pm_name:
            parts.append(pm_name)
        if pm_direction:
            parts.append(pm_direction)
        metric_hdr = " / ".join(parts) if parts else "primary_metric"
        metric_bits: List[str] = [f"current.mean={pm_current}"]
        if pm_best is not None:
            metric_bits.append(f"best.mean={pm_best}")
        if pm_baseline is not None:
            metric_bits.append(f"baseline.mean={pm_baseline}")
        if pm_delta is not None:
            metric_bits.append(f"delta_vs_baseline={pm_delta}")
        digest_lines.append(f"- {metric_hdr}: " + ", ".join(metric_bits) + "\n")
    if metrics_snapshot:
        if isinstance(metrics_snapshot, dict):
            keys = [str(k) for k in list(metrics_snapshot.keys())[:10]]
            more = " (truncated)" if len(metrics_snapshot.keys()) > 10 else ""
            digest_lines.append(f"- metrics: {', '.join(keys)}{more}\n")
        else:
            digest_lines.append(f"- metrics: list[{len(metrics_snapshot)}]\n")
    digest_lines.append(
        f"- Git changes: {int(git_change_summary.get('changed_files', 0))} files, {int(git_change_summary.get('changed_lines', 0))} lines\n"
    )
    digest_lines.append(f"- Review recommendation: **{review_recommendation.level}**\n")
    if review_recommendation.reasons:
        digest_lines.append("  - Reasons:\n")
        for r in review_recommendation.reasons:
            digest_lines.append(f"    - {r}\n")
    digest_lines.append("\n### Results\n- \n\n### Next experiments\n- \n\n")

    put_artifact(
        ledger=ledger,
        project=project,
        relative_path=digest_rel,
        content="".join(digest_lines),
        mode="append",
        kind="analysis_digest_md",
    )


def _write_last_errors(*, workspace: Path, tasks_ran: List[Dict[str, Any]], ledger) -> None:
    """Write notes/last_errors.md with stderr tails from failed tasks.

    If no tasks failed, removes the file so the Planner doesn't see stale errors.
    """
    out_path = workspace / "notes" / "last_errors.md"
    failed = [t for t in tasks_ran if t.get("status") not in {"success", None}]
    if not failed:
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
        return

    max_lines = 30
    max_tasks = 5
    lines: List[str] = ["# Execution Errors (previous iteration, auto-generated)\n"]
    for t in failed[:max_tasks]:
        task_type = t.get("type") or "unknown"
        status = t.get("status") or "unknown"
        task_id = t.get("id") or "?"
        lines.append(f"## Task {task_id} ({task_type}) — status: {status}")

        # Try to read stderr from the task's log directory.
        stderr_text = ""
        try:
            rows = ledger._exec(
                "SELECT meta_json FROM task_runs WHERE task_id = ? ORDER BY started_at DESC LIMIT 1",
                (task_id,),
            ).fetchall()
            if rows:
                meta_raw = rows[0].get("meta_json") or "{}"
                meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw if isinstance(meta_raw, dict) else {})
                sp = meta.get("stderr_path")
                if sp:
                    p = Path(sp)
                    if p.exists():
                        raw = p.read_text(encoding="utf-8", errors="replace")
                        raw_lines = raw.splitlines()
                        if len(raw_lines) > max_lines:
                            stderr_text = "\n".join(raw_lines[-max_lines:])
                        else:
                            stderr_text = raw
        except Exception:  # noqa: BLE001
            pass

        if stderr_text.strip():
            lines.append("```")
            lines.append(stderr_text.strip())
            lines.append("```")
        else:
            lines.append("_(stderr not available)_")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_last_challenger(*, workspace: Path, challenger_result: Dict[str, Any]) -> None:
    """Write notes/last_challenger.md with challenger flags.

    If challenger didn't fire or concern is low, removes the file.
    """
    out_path = workspace / "notes" / "last_challenger.md"
    concern = str(challenger_result.get("overall_concern_level") or "").strip().lower()
    flags = challenger_result.get("flags")
    if not isinstance(flags, list):
        flags = []
    flags = [str(f) for f in flags if str(f).strip()]

    if concern not in {"medium", "high"} or not flags:
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
        return

    lines = [
        "# Interpretation Challenger Flags (auto-generated)",
        f"concern_level: {concern}",
        "",
    ]
    for f in flags:
        lines.append(f"- {f}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Exploration log (rolling summary of alternatives_considered)
# ---------------------------------------------------------------------------

_EXPLORATION_LOG_RECENT_LIMIT = 3


def _parse_exploration_log(text: str) -> tuple:
    """Parse existing exploration_log.md into (rejected_lines, recent_entries).

    Returns:
        rejected_lines: list of "- approach: reason (iter N)" strings
        recent_entries: list of multi-line entry strings (one per iteration)
    """
    rejected_lines: List[str] = []
    recent_entries: List[str] = []

    if not text.strip():
        return rejected_lines, recent_entries

    in_rejected = False
    in_recent = False
    current_entry_lines: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Rejected directions"):
            in_rejected = True
            in_recent = False
            continue
        if stripped.startswith("## Recent alternatives"):
            in_rejected = False
            in_recent = True
            continue
        if stripped.startswith("# "):
            in_rejected = False
            in_recent = False
            continue

        if in_rejected:
            if stripped.startswith("- ") and ":" in stripped:
                rejected_lines.append(stripped)
        elif in_recent:
            if stripped.startswith("### Iteration"):
                if current_entry_lines:
                    recent_entries.append("\n".join(current_entry_lines))
                current_entry_lines = [stripped]
            elif current_entry_lines:
                current_entry_lines.append(line.rstrip())

    if current_entry_lines:
        recent_entries.append("\n".join(current_entry_lines))

    return rejected_lines, recent_entries


def _update_exploration_log(
    *,
    ledger: Ledger,
    project: Dict[str, Any],
    iteration: int,
    plan: Dict[str, Any],
) -> None:
    """Append alternatives_considered to exploration_log.md with rolling summary."""
    alternatives = plan.get("alternatives_considered")
    if not alternatives or not isinstance(alternatives, list):
        return

    workspace = Path(project["repo_path"]).resolve()
    log_path = workspace / "notes" / "exploration_log.md"

    existing = ""
    if log_path.exists():
        try:
            existing = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            existing = ""

    rejected_lines, recent_entries = _parse_exploration_log(existing)

    # Build new entry for this iteration.
    new_entry_lines = [f"### Iteration {iteration}"]
    chosen = str(plan.get("notes") or "").strip()
    if chosen:
        summary = chosen[:200] + "..." if len(chosen) > 200 else chosen
        new_entry_lines.append(f"Chosen approach: {summary}")
    for alt in alternatives:
        if not isinstance(alt, dict):
            continue
        approach = str(alt.get("approach") or "").strip()
        reason = str(alt.get("reason_rejected") or "").strip()
        if approach:
            new_entry_lines.append(f"- **{approach}**: {reason}")
            rejected_lines.append(f"- {approach}: {reason} (iter {iteration})")
    new_entry_lines.append("")

    recent_entries.append("\n".join(new_entry_lines))

    # Rolling: keep only last N entries in Recent section.
    if len(recent_entries) > _EXPLORATION_LOG_RECENT_LIMIT:
        recent_entries = recent_entries[-_EXPLORATION_LOG_RECENT_LIMIT:]

    # Deduplicate rejected directions (keep latest per approach name).
    seen: Dict[str, str] = {}
    for line in rejected_lines:
        key = line.split(":")[0].strip().lstrip("- ").lower()
        if key:
            seen[key] = line
    deduped_rejected = list(seen.values())

    # Rebuild the file.
    output = "# Exploration Log\n\n"
    output += "## Rejected directions (cumulative)\n"
    if deduped_rejected:
        output += "\n".join(deduped_rejected) + "\n"
    else:
        output += "(none yet)\n"
    output += "\n## Recent alternatives (last 3 iterations)\n"
    output += "\n".join(recent_entries) + "\n"

    put_artifact(
        ledger=ledger,
        project=project,
        relative_path="notes/exploration_log.md",
        content=output,
        mode="overwrite",
        kind="exploration_log",
    )
