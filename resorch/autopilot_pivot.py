from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from resorch.autopilot_config import load_pivot_policy
from resorch.utils import read_text


def _nested_get(obj: Any, path: str) -> Any:
    cur: Any = obj
    for part in (path or "").split("."):
        if not part:
            continue
        if not isinstance(cur, dict):
            return None
        if part not in cur:
            return None
        cur = cur[part]
    return cur


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


def _pivot_no_improvement_trigger(
    *,
    repo_root: Path,
    workspace: Path,
) -> Optional[Tuple[str, str]]:
    policy = load_pivot_policy(repo_root)
    spec = policy.get("no_improvement") or {}
    if not isinstance(spec, dict) or not bool(spec.get("enabled", True)):
        return None

    metric_path = str(spec.get("metric_path") or "primary_metric.current.mean").strip()
    direction = str(spec.get("direction") or "maximize").strip().lower()
    if direction not in {"maximize", "minimize"}:
        direction = "maximize"
    use_ci_overlap = bool(spec.get("use_ci_overlap", False))

    try:
        min_delta = float(spec.get("min_delta") or 0.0)
    except (TypeError, ValueError):
        min_delta = 0.0
    try:
        window_runs = int(spec.get("window_runs") or 0)
    except (TypeError, ValueError):
        window_runs = 0
    if window_runs < 2:
        return None

    scoreboard_path = (workspace / "results" / "scoreboard.json").resolve()
    try:
        raw = read_text(scoreboard_path)
        scoreboard = json.loads(raw) if raw.strip() else {}
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(scoreboard, dict):
        return None

    # Backward compatibility: if primary_metric.current is a number (old format),
    # treat it as {"mean": current} so metric_path like `primary_metric.current.mean` works.
    def _normalize_primary_metric_current(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        pm = obj.get("primary_metric")
        if not isinstance(pm, dict):
            return
        cur = pm.get("current")
        if isinstance(cur, bool):
            return
        if isinstance(cur, (int, float)):
            pm["current"] = {"mean": float(cur)}

    _normalize_primary_metric_current(scoreboard)

    runs = scoreboard.get("runs")
    if not isinstance(runs, list):
        runs = []
    for r in runs:
        _normalize_primary_metric_current(r)

    current_val = _as_float(_nested_get(scoreboard, metric_path))
    if current_val is None:
        return None

    sample_runs: List[Dict[str, Any]] = []
    # Previous values from past runs (most recent first).
    for r in reversed(runs):
        if len(sample_runs) >= (window_runs - 1):
            break
        if not isinstance(r, dict):
            continue
        val = _as_float(_nested_get(r, metric_path))
        if val is None:
            continue
        sample_runs.append(r)
    sample_objs: List[Dict[str, Any]] = list(reversed(sample_runs)) + [scoreboard]

    values: List[float] = []
    for obj in sample_objs:
        v = _as_float(_nested_get(obj, metric_path))
        if v is None:
            return None
        values.append(v)
    if len(values) < window_runs:
        return None

    def _parse_ci_95(v: Any) -> Optional[Tuple[float, float]]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            return None
        a = _as_float(v[0])
        b = _as_float(v[1])
        if a is None or b is None:
            return None
        lo = min(float(a), float(b))
        hi = max(float(a), float(b))
        return (lo, hi)

    if use_ci_overlap:
        ci_path = ""
        if metric_path.endswith(".mean"):
            ci_path = metric_path.rsplit(".", 1)[0] + ".ci_95"
        if ci_path:
            cis: List[Optional[Tuple[float, float]]] = [_parse_ci_95(_nested_get(obj, ci_path)) for obj in sample_objs]
            if all(ci is not None for ci in cis):
                # If there is any statistically significant improvement in the window, do NOT pivot.
                for i in range(1, len(cis)):
                    prev_ci = cis[i - 1]
                    cur_ci = cis[i]
                    if prev_ci is None or cur_ci is None:
                        continue
                    prev_lo, prev_hi = prev_ci
                    cur_lo, cur_hi = cur_ci
                    overlaps = not (cur_lo > prev_hi or prev_lo > cur_hi)
                    if overlaps:
                        continue
                    if direction == "minimize":
                        # Significant decrease: current entirely below previous.
                        if cur_hi < prev_lo:
                            return None
                    else:
                        # Significant increase: current entirely above previous.
                        if cur_lo > prev_hi:
                            return None

                level = str(spec.get("review_level") or "soft").strip().lower()
                if level not in {"soft", "hard"}:
                    level = "soft"
                reason = (
                    f"pivot_no_improvement({metric_path} window_runs={window_runs} min_delta={min_delta} pivot_reason=ci_overlap)"
                )
                return level, reason

    def _improvement(prev: float, cur: float) -> float:
        return (cur - prev) if direction == "maximize" else (prev - cur)

    improvements = [_improvement(values[i - 1], values[i]) for i in range(1, len(values))]
    if not all(imp < min_delta for imp in improvements):
        return None

    level = str(spec.get("review_level") or "soft").strip().lower()
    if level not in {"soft", "hard"}:
        level = "soft"
    reason = f"pivot_no_improvement({metric_path} window_runs={window_runs} min_delta={min_delta} pivot_reason=min_delta_not_met)"
    return level, reason
