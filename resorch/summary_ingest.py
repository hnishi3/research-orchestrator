from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from resorch.artifacts import put_artifact, register_artifact
from resorch.ledger import Ledger
from resorch.projects import get_project
from resorch.utils import read_text, utc_now_iso


def _as_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _coerce_metric_stats(v: Any) -> Optional[Dict[str, Any]]:
    """Coerce a numeric or stats-object value into the scoreboard metric_stats shape."""
    if isinstance(v, dict):
        mean = _as_float(v.get("mean"))
        if mean is None:
            return None
        out: Dict[str, Any] = dict(v)
        out["mean"] = float(mean)
        return out
    mean = _as_float(v)
    if mean is None:
        return None
    return {"mean": float(mean)}


def ingest_summary(
    *,
    ledger: Ledger,
    project_id: str,
    summary_path: str = "results/summary.json",
    register_summary_artifact: bool = True,
) -> Dict[str, Any]:
    project = get_project(ledger, project_id)
    workspace = Path(project["repo_path"]).resolve()

    sp = Path(summary_path)
    if not sp.is_absolute():
        sp = (workspace / sp).resolve()
    try:
        sp.relative_to(workspace)
    except ValueError:
        raise SystemExit(f"summary.json must be under the workspace: {sp}")
    if not sp.exists():
        raise SystemExit(f"summary.json not found: {sp}")

    raw = read_text(sp)
    try:
        summary = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in summary file: {sp} ({e})") from e
    if not isinstance(summary, dict):
        raise SystemExit(f"summary.json must be a JSON object: {sp}")

    rel_summary = sp.relative_to(workspace).as_posix()
    if register_summary_artifact:
        try:
            register_artifact(
                ledger=ledger,
                project=project,
                kind="pipeline_summary_json",
                relative_path=rel_summary,
                meta={},
            )
        except SystemExit:
            pass

    scoreboard_rel = "results/scoreboard.json"
    scoreboard_path = (workspace / scoreboard_rel).resolve()
    try:
        scoreboard_raw = read_text(scoreboard_path)
        scoreboard = json.loads(scoreboard_raw) if scoreboard_raw.strip() else {}
    except (OSError, json.JSONDecodeError):
        scoreboard = {}
    if not isinstance(scoreboard, dict):
        scoreboard = {}

    scoreboard.setdefault("schema_version", 2)
    scoreboard["updated_at"] = utc_now_iso()
    scoreboard.setdefault("primary_metric", {})
    if not isinstance(scoreboard.get("primary_metric"), dict):
        scoreboard["primary_metric"] = {}
    scoreboard.setdefault("metrics", {})

    pm_in = summary.get("primary_metric") or {}
    if not isinstance(pm_in, dict):
        pm_in = {}
    metrics_in = summary.get("metrics")

    pm_out = dict(scoreboard.get("primary_metric") or {})
    pm_name = pm_in.get("name")
    if isinstance(pm_name, str):
        pm_out["name"] = pm_name
    pm_direction = pm_in.get("direction")
    if isinstance(pm_direction, str) and pm_direction.strip().lower() in {"maximize", "minimize"}:
        pm_out["direction"] = pm_direction.strip().lower()

    cur_stats = _coerce_metric_stats(pm_in.get("current"))
    if cur_stats is None:
        cur_stats = _coerce_metric_stats(pm_in.get("value"))
    if cur_stats is not None:
        pm_out["current"] = cur_stats
    cur = _as_float((cur_stats or {}).get("mean"))

    baseline_stats = _coerce_metric_stats(pm_in.get("baseline"))
    if baseline_stats is not None:
        pm_out["baseline"] = baseline_stats
    baseline = _as_float((baseline_stats or {}).get("mean"))
    if cur is not None and baseline is not None:
        pm_out["delta_vs_baseline"] = float(cur - baseline)

    best_prev = _as_float(pm_out.get("best"))
    if best_prev is None and isinstance(pm_out.get("best"), dict):
        best_prev = _as_float((pm_out.get("best") or {}).get("mean"))
    direction = str(pm_out.get("direction") or "maximize").strip().lower()
    if cur is not None:
        if best_prev is None:
            pm_out["best"] = cur
        elif direction == "minimize":
            pm_out["best"] = min(best_prev, cur)
        else:
            pm_out["best"] = max(best_prev, cur)

    scoreboard["primary_metric"] = pm_out

    if isinstance(metrics_in, dict) and isinstance(scoreboard.get("metrics"), dict):
        merged = dict(scoreboard.get("metrics") or {})
        merged.update(metrics_in)
        scoreboard["metrics"] = merged
    elif isinstance(metrics_in, (dict, list)):
        scoreboard["metrics"] = metrics_in

    runs = scoreboard.get("runs")
    if not isinstance(runs, list):
        runs = []
    runs.append(
        {
            "ts": utc_now_iso(),
            "source": "summary_ingest",
            "summary_path": rel_summary,
            "primary_metric": dict(scoreboard.get("primary_metric") or {}),
            "metrics": scoreboard.get("metrics"),
            "notes": str(summary.get("notes") or ""),
        }
    )
    scoreboard["runs"] = runs[-400:]

    put_artifact(
        ledger=ledger,
        project=project,
        relative_path=scoreboard_rel,
        content=json.dumps(scoreboard, ensure_ascii=False, indent=2) + "\n",
        mode="overwrite",
        kind="scoreboard_json",
    )

    digest_rel = "notes/analysis_digest.md"
    digest_lines = []
    digest_lines.append("\n## Pipeline Summary Ingest\n")
    digest_lines.append(f"- When: {utc_now_iso()}\n")
    digest_lines.append(f"- Summary: `{rel_summary}`\n")
    if cur is not None:
        digest_lines.append(f"- primary_metric.current.mean: {cur}\n")
    if isinstance(scoreboard.get("metrics"), dict) and scoreboard.get("metrics"):
        keys = [str(k) for k in list((scoreboard.get("metrics") or {}).keys())[:12]]
        more = " (truncated)" if len((scoreboard.get("metrics") or {}).keys()) > 12 else ""
        digest_lines.append(f"- metrics: {', '.join(keys)}{more}\n")
    digest_lines.append("\n")

    put_artifact(
        ledger=ledger,
        project=project,
        relative_path=digest_rel,
        content="".join(digest_lines),
        mode="append",
        kind="analysis_digest_md",
    )

    return {"scoreboard_path": scoreboard_rel, "digest_path": digest_rel, "summary_path": rel_summary}
