from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

from resorch.artifacts import put_artifact
from resorch.autopilot import (
    load_pivot_policy,
    load_review_policy,
    run_autopilot_iteration,
    summarize_codex_exec_statuses,
)
from resorch.jobs import create_job, run_job
from resorch.projects import get_project, set_project_stage
from resorch.tasks import create_task, list_tasks, run_task
from resorch.utils import utc_now_iso


STAGE_ORDER: List[str] = [
    "intake",
    "literature",
    "method",
    "implementation",
    "analysis",
    "writing",
    "submission",
]


def _is_stage_backward(current: str, requested: str) -> bool:
    """Return True if *requested* stage is earlier than *current* in STAGE_ORDER."""
    cur = current.strip().lower()
    req = requested.strip().lower()
    try:
        return STAGE_ORDER.index(req) < STAGE_ORDER.index(cur)
    except ValueError:
        return False  # unknown stage → treat as forward (safe side)


@dataclass(frozen=True)
class AgentLoopConfig:
    planner_model: str = "opus"
    planner_provider: str = "claude_code_cli"  # "openai" | "claude_code_cli" | "codex_cli"
    planner_background: bool = True
    planner_reasoning_effort: Optional[str] = None  # "low" | "medium" | "high"
    planner_timeout: int = 1800  # seconds; Claude CLI needs more time than OpenAI
    max_actions: int = 6
    max_fix_tasks_per_review: int = 10
    review_questions: Optional[List[str]] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "AgentLoopConfig":
        if yaml is None:  # pragma: no cover
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise SystemExit(f"Invalid agent loop config (expected object): {path}")

        planner = raw.get("planner") or {}
        if not isinstance(planner, dict):
            planner = {}
        review = raw.get("review") or {}
        if not isinstance(review, dict):
            review = {}

        questions = review.get("questions")
        if questions is not None and not isinstance(questions, list):
            raise SystemExit("agent_loop.yaml: review.questions must be a list of strings")

        return cls(
            planner_model=str(planner.get("model") or cls.planner_model),
            planner_provider=str(planner.get("provider") or cls.planner_provider),
            planner_background=bool(planner.get("background", cls.planner_background)),
            planner_reasoning_effort=str(planner.get("reasoning_effort")) if planner.get("reasoning_effort") else None,
            planner_timeout=int(planner.get("timeout", cls.planner_timeout)),
            max_actions=int(planner.get("max_actions", cls.max_actions)),
            max_fix_tasks_per_review=int(review.get("max_fix_tasks_per_review", cls.max_fix_tasks_per_review)),
            review_questions=[str(x) for x in (questions or [])] if questions is not None else None,
        )


def _default_config() -> AgentLoopConfig:
    model = os.environ.get("OPENAI_PLANNER_MODEL") or "gpt-5.2-pro"
    provider = os.environ.get("PLANNER_PROVIDER") or "openai"
    return AgentLoopConfig(planner_model=model, planner_provider=provider)


def load_agent_loop_config(
    repo_root: Path,
    workspace: Optional[Path] = None,
    explicit_path: Optional[str] = None,
) -> AgentLoopConfig:
    """Load agent loop config with workspace-level override support.

    Resolution order:
      - If *explicit_path* is given (user passed ``--config``), use it directly.
      - Otherwise auto-discover:
          1. workspace/configs/agent_loop.yaml  (if workspace given and file exists)
          2. repo_root/configs/agent_loop.yaml  (global fallback)
          3. env-based default
    """
    if explicit_path:
        p = Path(explicit_path)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        if p.exists():
            log.info("agent_loop config: explicit %s", p)
            return AgentLoopConfig.from_yaml(p)
        log.warning("agent_loop config: explicit path %s not found, using defaults", p)
        return _default_config()

    # Auto-discovery
    if workspace is not None:
        ws_path = Path(workspace) / "configs" / "agent_loop.yaml"
        if ws_path.exists():
            log.info("agent_loop config: workspace %s", ws_path)
            return AgentLoopConfig.from_yaml(ws_path)
    global_path = repo_root / "configs" / "agent_loop.yaml"
    if global_path.exists():
        log.info("agent_loop config: global %s", global_path)
        return AgentLoopConfig.from_yaml(global_path)
    log.info("agent_loop config: env defaults")
    return _default_config()


def _pick_reviewer(policy: Dict[str, Any], level: str) -> Dict[str, Any]:
    reviewers = policy.get("reviewers") or {}
    if not isinstance(reviewers, dict):
        reviewers = {}
    if level == "hard":
        rv = reviewers.get("escalation") or {}
    else:
        rv = reviewers.get("primary") or {}
    if not isinstance(rv, dict):
        rv = {}
    return rv


def _extract_review_result(job: Dict[str, Any]) -> Dict[str, Any]:
    result = job.get("result") if isinstance(job, dict) else None
    if not isinstance(result, dict):
        return {}
    review_result = result.get("review_result")
    if isinstance(review_result, dict):
        return review_result
    fb = result.get("fallback_job_result")
    if isinstance(fb, dict):
        review_result = fb.get("review_result")
        if isinstance(review_result, dict):
            return review_result
    return {}


def _is_lightweight_fixable(findings: list[dict]) -> bool:
    """All findings are minor/nit + code-level categories → skip Planner."""
    if not findings:
        return False
    for f in findings:
        if not isinstance(f, dict):
            return False
        severity = str(f.get("severity") or "").lower()
        category = str(f.get("category") or "").lower()
        if severity in {"blocker", "major"}:
            return False
        if category in {"novelty", "method", "metrics"}:
            return False
    return True


def _get_last_shell_actions(iter_out: Dict[str, Any]) -> List[Dict[str, Any]]:
    plan = iter_out.get("plan") or {}
    if not isinstance(plan, dict):
        return []
    actions = plan.get("actions") or []
    if not isinstance(actions, list):
        return []
    shell_actions = [a for a in actions if isinstance(a, dict) and str(a.get("task_type") or "").strip() == "shell_exec"]
    return shell_actions


def _generate_stagnation_report(
    workspace: Path,
    stopped_reason: str,
    steps: List[Dict[str, Any]],
) -> str:
    """Generate a structured stagnation report from scoreboard + step history.

    Returns the report as a markdown string.  Writes it to
    ``workspace / notes / stagnation_report.md`` as a side-effect.
    """
    lines: List[str] = [
        "# Stagnation Report (auto-generated)",
        f"generated_at: {utc_now_iso()}",
        f"stopped_reason: {stopped_reason}",
        "",
    ]

    # --- Scoreboard summary ---
    sb_path = workspace / "results" / "scoreboard.json"
    sb: Dict[str, Any] = {}
    try:
        raw = sb_path.read_text(encoding="utf-8", errors="replace")
        sb = json.loads(raw) if raw.strip() else {}
    except (OSError, json.JSONDecodeError):
        pass
    pm = sb.get("primary_metric") or {}
    if not isinstance(pm, dict):
        pm = {}
    cur = pm.get("current") or {}
    if not isinstance(cur, dict):
        cur = {}
    best = pm.get("best") or {}
    if not isinstance(best, dict):
        best = {}

    cur_mean = cur.get("mean")
    cur_n = cur.get("n_requested") or cur.get("n")
    cur_ci = cur.get("ci_95") or {}
    best_mean = best.get("mean")
    best_n = best.get("n_requested") or best.get("n")
    best_ci = best.get("ci_95") or {}

    lines.append("## Primary Metric")
    lines.append(f"- name: {pm.get('name', 'unknown')}")
    lines.append(f"- direction: {pm.get('direction', 'maximize')}")
    lines.append(f"- current_mean: {cur_mean}")
    lines.append(f"- current_n: {cur_n}")
    if isinstance(cur_ci, dict):
        lines.append(f"- current_ci95: [{cur_ci.get('low')}, {cur_ci.get('high')}]")
    lines.append(f"- best_mean: {best_mean}")
    lines.append(f"- best_n: {best_n}")
    if isinstance(best_ci, dict):
        lines.append(f"- best_ci95: [{best_ci.get('low')}, {best_ci.get('high')}]")

    # CI overlap detection
    ci_overlap = None
    if isinstance(cur_ci, dict) and isinstance(best_ci, dict):
        try:
            c_lo, c_hi = float(cur_ci["low"]), float(cur_ci["high"])
            b_lo, b_hi = float(best_ci["low"]), float(best_ci["high"])
            ci_overlap = not (c_lo > b_hi or b_lo > c_hi)
        except (KeyError, TypeError, ValueError):
            pass
    if ci_overlap is not None:
        lines.append(f"- ci_overlap_current_vs_best: {ci_overlap}")
        if ci_overlap:
            lines.append("- interpretation: best is NOT statistically distinguishable from current")

    # Small-sample warning for best
    try:
        _cur_n = int(cur_n) if cur_n is not None else 0
        _best_n = int(best_n) if best_n is not None else 0
    except (TypeError, ValueError):
        _cur_n, _best_n = 0, 0
    if _best_n > 0 and _cur_n > 0 and _best_n < _cur_n / 3:
        lines.append(f"- best_sample_warning: best is from n={_best_n} vs current n={_cur_n} ({_cur_n//_best_n}x smaller)")
        lines.append("  A small-sample best is unreliable; the true rate is likely closer to current.")

    lines.append("")

    # --- Step history summary ---
    lines.append("## Step History")
    for rec in steps:
        si = rec.get("step", "?")
        ap = rec.get("autopilot") or {}
        rr = (ap.get("review_recommendation") or {}).get("reasons") or []
        pivot_hit = any(str(r).startswith("pivot_no_improvement(") for r in rr if isinstance(r, str))
        plan = ap.get("plan") or {}
        n_actions = len(plan.get("actions") or []) if isinstance(plan, dict) else 0
        do_failed = bool(ap.get("do_phase_failed"))
        lines.append(f"- step_{si:03d}: actions={n_actions}, do_failed={do_failed}, pivot_fired={pivot_hit}")
    lines.append("")

    # --- Actionable summary for Planner ---
    lines.append("## Implications")
    if ci_overlap:
        lines.append("- The recorded best's CI overlaps with current — best is NOT statistically better.")
        lines.append("  Do NOT treat best as a reachable target.")
    if _best_n > 0 and _cur_n > 0 and _best_n < _cur_n / 3:
        lines.append(f"- best came from only n={_best_n} samples (vs current n={_cur_n}).")
        lines.append("  Small-sample best is unreliable and likely inflated. Do NOT chase it.")
    lines.append(f"- {len(steps)} consecutive iterations showed no significant metric improvement.")
    lines.append("- Consider: revising the target, changing approach, or advancing to the next project stage.")

    report = "\n".join(lines) + "\n"

    # Write to workspace
    out_path = workspace / "notes" / "stagnation_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    return report


def _load_last_plan_from_workspace(workspace: Path) -> Optional[Dict[str, Any]]:
    """Find the most recent step JSON and extract its plan.

    Returns the plan dict if found, else None.
    """
    runs_dir = workspace / "runs" / "agent"
    if not runs_dir.is_dir():
        return None

    # Search both new (run-<ts>/step_*/step.json) and legacy (step_*/step.json)
    # layouts; sort by modification time (newest first).
    step_dirs = sorted(
        list(runs_dir.glob("run-*/step_*/step.json"))
        + list(runs_dir.glob("step_*/step.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for step_json_path in step_dirs:
        try:
            raw = step_json_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            continue
        autopilot = data.get("autopilot")
        if not isinstance(autopilot, dict):
            continue
        plan = autopilot.get("plan")
        if isinstance(plan, dict) and plan.get("actions"):
            log.info("Loaded plan from %s for reuse.", step_json_path)
            return plan
    return None


def run_agent_loop(
    *,
    ledger,
    project_id: str,
    objective: str,
    max_steps: int,
    dry_run: bool,
    config_path: Optional[str] = None,
    model: Optional[str] = None,
    background: Optional[bool] = None,
    max_actions: Optional[int] = None,
    reuse_last_plan: bool = False,
) -> Dict[str, Any]:
    repo_root = ledger.paths.root

    project = get_project(ledger, project_id)
    workspace = Path(project["repo_path"]).resolve()

    cfg = load_agent_loop_config(repo_root, workspace=workspace, explicit_path=config_path)

    if model:
        cfg = AgentLoopConfig(**{**cfg.__dict__, "planner_model": str(model)})
    if background is not None:
        cfg = AgentLoopConfig(**{**cfg.__dict__, "planner_background": bool(background)})
    if max_actions is not None:
        cfg = AgentLoopConfig(**{**cfg.__dict__, "max_actions": int(max_actions)})

    policy = load_review_policy(repo_root, workspace=workspace)

    steps: List[Dict[str, Any]] = []
    decision_log: List[Dict[str, Any]] = []
    stopped_reason = None

    lr_cfg = policy.get("lightweight_retry") or {}
    if not isinstance(lr_cfg, dict):
        lr_cfg = {}
    lr_enabled = bool(lr_cfg.get("enabled", False))
    try:
        lr_max_consecutive = int(lr_cfg.get("max_consecutive", 2) or 0)
    except (TypeError, ValueError):
        lr_max_consecutive = 2
    if lr_max_consecutive < 0:
        lr_max_consecutive = 0

    soft_triggers = policy.get("soft_triggers") or {}
    if not isinstance(soft_triggers, dict):
        soft_triggers = {}
    cost_guard = soft_triggers.get("cost_guard") or {}
    if not isinstance(cost_guard, dict):
        cost_guard = {}
    try:
        cost_window_minutes = int(cost_guard.get("window_minutes") or 0)
    except (TypeError, ValueError):
        cost_window_minutes = 0
    try:
        cost_max_total_tokens = int(cost_guard.get("max_total_tokens") or 0)
    except (TypeError, ValueError):
        cost_max_total_tokens = 0
    token_events: List[tuple[float, int]] = []

    auto_stage_cfg = policy.get("auto_stage_update") or {}
    if not isinstance(auto_stage_cfg, dict):
        auto_stage_cfg = {}
    auto_stage_enabled = bool(auto_stage_cfg.get("enabled", False))
    auto_stage_apply_on = auto_stage_cfg.get("apply_on") or ["accept", "minor"]
    if not isinstance(auto_stage_apply_on, list):
        auto_stage_apply_on = ["accept", "minor"]
    auto_stage_apply_on_set = {str(x).strip().lower() for x in auto_stage_apply_on if x}

    rerun_shell_actions: Optional[List[Dict[str, Any]]] = None
    lightweight_retry_streak = 0

    # --reuse-last-plan: load the plan from the most recent step JSON
    # and use it on the first iteration instead of calling the Planner.
    cached_reuse_plan: Optional[Dict[str, Any]] = None
    if reuse_last_plan:
        cached_reuse_plan = _load_last_plan_from_workspace(workspace)
        if cached_reuse_plan is None:
            log.warning("--reuse-last-plan: no previous step with a valid plan found; will call Planner normally.")
        else:
            log.info("--reuse-last-plan: will reuse cached plan on first iteration (skipping Planner).")

    # Do-phase failure retry state: reuse plan when codex_exec fails transiently.
    do_retry_actions: Optional[List[Dict[str, Any]]] = None
    do_retry_streak = 0
    _do_retry_cfg = policy.get("do_phase_retry") or {}
    if not isinstance(_do_retry_cfg, dict):
        _do_retry_cfg = {}
    try:
        do_retry_max = int(_do_retry_cfg.get("max_consecutive", 3))
    except (TypeError, ValueError):
        do_retry_max = 3
    try:
        do_retry_backoff_sec = float(_do_retry_cfg.get("backoff_sec", 30))
    except (TypeError, ValueError):
        do_retry_backoff_sec = 30.0

    # Pivot stagnation: force-stop after N consecutive pivot_no_improvement firings.
    _pivot_policy = load_pivot_policy(repo_root)
    _no_imp = _pivot_policy.get("no_improvement") or {}
    if not isinstance(_no_imp, dict):
        _no_imp = {}
    try:
        force_stop_after = int(_no_imp.get("force_stop_after", 0))
    except (TypeError, ValueError):
        force_stop_after = 0
    pivot_stagnation_streak = 0

    # Fail-streak force-stop: if overall task failure streak exceeds threshold, halt.
    _soft_triggers = policy.get("soft_triggers") or {}
    if not isinstance(_soft_triggers, dict):
        _soft_triggers = {}
    _fail_streak_cfg = _soft_triggers.get("fail_streak") or {}
    if not isinstance(_fail_streak_cfg, dict):
        _fail_streak_cfg = {}
    try:
        fail_streak_force_stop = int(_fail_streak_cfg.get("force_stop_after", 0))
    except (TypeError, ValueError):
        fail_streak_force_stop = 0

    # --- Metric watchdogs: detect null or unchanged primary metric ---
    _watchdog_cfg = _pivot_policy.get("metric_watchdog") or {}
    if not isinstance(_watchdog_cfg, dict):
        _watchdog_cfg = {}
    try:
        _null_metric_max = int(_watchdog_cfg.get("null_metric_force_stop_after", 3))
    except (TypeError, ValueError):
        _null_metric_max = 3
    try:
        _unchanged_metric_max = int(_watchdog_cfg.get("unchanged_metric_force_stop_after", 3))
    except (TypeError, ValueError):
        _unchanged_metric_max = 3
    _null_metric_streak = 0
    _unchanged_metric_streak = 0
    _prev_metric_value: Optional[float] = None  # sentinel: None means "not yet read"
    _watchdog_exempt_stages: set[str] = set()
    _raw_exempt = _watchdog_cfg.get("exempt_stages")
    if isinstance(_raw_exempt, list):
        _watchdog_exempt_stages = {str(s).strip().lower() for s in _raw_exempt if s}
    else:
        _watchdog_exempt_stages = {"writing", "validation", "revision", "final"}

    try:
        _codex_stall_detect_after = int(_watchdog_cfg.get("codex_blocked_detect_after", 2))
    except (TypeError, ValueError):
        _codex_stall_detect_after = 2
    if _codex_stall_detect_after < 1:
        _codex_stall_detect_after = 1
    _codex_stall_error_after = 3
    _consecutive_codex_blocked = 0

    def _read_primary_metric_mean() -> Optional[float]:
        """Read primary_metric.current.mean from the workspace scoreboard."""
        sb_path = workspace / "results" / "scoreboard.json"
        try:
            sb = json.loads(sb_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(sb, dict):
            return None
        pm = sb.get("primary_metric")
        if not isinstance(pm, dict):
            return None
        cur = pm.get("current")
        # Backward compat: flat numeric current (matches autopilot.py normalize)
        if isinstance(cur, (int, float)) and not isinstance(cur, bool):
            return float(cur)
        if not isinstance(cur, dict):
            return None
        val = cur.get("mean")
        if isinstance(val, bool) or val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    # Unique run-level directory so step files are never overwritten across runs.
    _run_ts = utc_now_iso().replace(":", "")
    _run_dir = f"runs/agent/run-{_run_ts}"

    for step_idx in range(max_steps):
        # --- Force-stop: null metric watchdog ---
        if _null_metric_max > 0 and _null_metric_streak >= _null_metric_max:
            stopped_reason = (
                f"null_metric_force_stop(streak={_null_metric_streak},"
                f"threshold={_null_metric_max})"
            )
            log.warning(
                "Primary metric has been null for %d consecutive iterations "
                "(threshold %d) — stopping. Verify the pipeline can produce "
                "a numeric result.",
                _null_metric_streak, _null_metric_max,
            )
            _generate_stagnation_report(workspace, stopped_reason, steps)
            break

        # --- Force-stop: unchanged metric watchdog ---
        if _unchanged_metric_max > 0 and _unchanged_metric_streak >= _unchanged_metric_max:
            _cur_stage = str(get_project(ledger, project_id).get("stage") or "").strip().lower()
            if _cur_stage in _watchdog_exempt_stages:
                log.info(
                    "Metric watchdog: unchanged streak %d reached threshold %d "
                    "but stage '%s' is exempt (metrics expected to be stable).",
                    _unchanged_metric_streak, _unchanged_metric_max, _cur_stage,
                )
            else:
                stopped_reason = (
                    f"unchanged_metric_force_stop(streak={_unchanged_metric_streak},"
                    f"threshold={_unchanged_metric_max},value={_prev_metric_value})"
                )
                log.warning(
                    "Primary metric unchanged at %s for %d consecutive iterations "
                    "(threshold %d) — stopping.",
                    _prev_metric_value, _unchanged_metric_streak, _unchanged_metric_max,
                )
                _generate_stagnation_report(workspace, stopped_reason, steps)
                break

        # --- Force-stop: pivot stagnation streak exceeded threshold ---
        if force_stop_after > 0 and pivot_stagnation_streak >= force_stop_after:
            stopped_reason = f"pivot_stagnation_force_stop(streak={pivot_stagnation_streak},threshold={force_stop_after})"
            log.warning(
                "Pivot stagnation streak %d >= force_stop_after %d — stopping before Planner call.",
                pivot_stagnation_streak, force_stop_after,
            )
            _generate_stagnation_report(workspace, stopped_reason, steps)
            log.info("Stagnation report written to %s/notes/stagnation_report.md", workspace)
            break

        rerun_mode = False
        actions_for_rerun: Optional[List[Dict[str, Any]]] = None
        retry_mode = False
        actions_for_retry: Optional[List[Dict[str, Any]]] = None
        reuse_plan: Optional[Dict[str, Any]] = None

        if do_retry_actions is not None:
            retry_mode = True
            actions_for_retry = do_retry_actions
            do_retry_actions = None
            # Backoff before retrying to let rate limits recover.
            backoff = do_retry_backoff_sec * (2 ** (do_retry_streak - 1))
            log.info("Do-phase retry %d/%d — sleeping %.0fs before retry.", do_retry_streak, do_retry_max, backoff)
            time.sleep(backoff)
        elif rerun_shell_actions is not None:
            rerun_mode = True
            actions_for_rerun = rerun_shell_actions
            rerun_shell_actions = None
        elif cached_reuse_plan is not None:
            # --reuse-last-plan: use cached plan on first iteration only.
            reuse_plan = cached_reuse_plan
            cached_reuse_plan = None  # only applies once
            log.info("--reuse-last-plan: using cached plan (skipping Planner call).")

        iter_out = run_autopilot_iteration(
            ledger=ledger,
            project_id=project_id,
            objective=objective,
            model=cfg.planner_model,
            iteration=step_idx,
            dry_run=dry_run,
            max_actions=cfg.max_actions,
            background=cfg.planner_background,
            config=cfg.__dict__,
            rerun_mode=rerun_mode,
            rerun_actions=actions_for_rerun,
            retry_mode=retry_mode,
            retry_actions=actions_for_retry,
            reuse_plan=reuse_plan,
        )

        step_notes: List[str] = []
        step_decision_log: List[Dict[str, Any]] = []
        try:
            _codex_summary = iter_out.get("codex_exec_summary")
            if not isinstance(_codex_summary, dict):
                _codex_summary = summarize_codex_exec_statuses(iter_out.get("tasks_ran") or [])

            codex_total = int(_codex_summary.get("total", 0) or 0)
            codex_blocked = int(_codex_summary.get("blocked", 0) or 0)
            codex_success = int(_codex_summary.get("success", 0) or 0)
            codex_all_blocked = bool(codex_total > 0 and codex_blocked == codex_total)

            if codex_success > 0:
                if _consecutive_codex_blocked > 0:
                    log.info(
                        "Codex stall detector: success observed; resetting blocked streak from %d.",
                        _consecutive_codex_blocked,
                    )
                _consecutive_codex_blocked = 0
            elif codex_all_blocked:
                _consecutive_codex_blocked += 1
            else:
                _consecutive_codex_blocked = 0

            codex_stall_detected = bool(
                _consecutive_codex_blocked >= _codex_stall_detect_after and _consecutive_codex_blocked > 0
            )
            iter_out["codex_exec_summary"] = _codex_summary
            iter_out["codex_stall_detected"] = codex_stall_detected
            iter_out["consecutive_codex_blocked"] = int(_consecutive_codex_blocked)

            if codex_stall_detected:
                warn_msg = (
                    "Codex stall detector: %d consecutive iterations with all codex_exec tasks blocked "
                    "(threshold=%d)."
                ) % (_consecutive_codex_blocked, _codex_stall_detect_after)
                log.warning(warn_msg)
                step_decision_log.append(
                    {
                        "event": "codex_stall_warning",
                        "step": int(step_idx),
                        "consecutive_codex_blocked": int(_consecutive_codex_blocked),
                        "threshold": int(_codex_stall_detect_after),
                        "codex_total": int(codex_total),
                        "codex_blocked": int(codex_blocked),
                    }
                )

            if _consecutive_codex_blocked >= _codex_stall_error_after:
                stall_note = (
                    "STALL DETECTED: %d consecutive iterations with all Codex tasks blocked. "
                    "Recommend stopping and diagnosing Codex CLI."
                ) % _consecutive_codex_blocked
                log.error(stall_note)
                step_notes.append(stall_note)
                step_decision_log.append(
                    {
                        "event": "codex_stall_error",
                        "step": int(step_idx),
                        "consecutive_codex_blocked": int(_consecutive_codex_blocked),
                        "threshold": int(_codex_stall_error_after),
                        "message": stall_note,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            iter_out["codex_stall_detected"] = False
            log.warning("Codex stall detector failed-open at step %d: %s", step_idx, exc)
            step_decision_log.append(
                {
                    "event": "codex_stall_detector_error",
                    "step": int(step_idx),
                    "error": str(exc),
                }
            )

        decision_log.extend(step_decision_log)

        # Soft trigger: cost guard based on recent planner token usage.
        if cost_window_minutes > 0 and cost_max_total_tokens > 0:
            now = time.time()
            meta = iter_out.get("planner_meta") or {}
            tokens = meta.get("usage_total_tokens") if isinstance(meta, dict) else None
            if isinstance(tokens, int) and tokens >= 0:
                token_events.append((now, tokens))
            window_sec = float(cost_window_minutes) * 60.0
            token_events = [(t, n) for (t, n) in token_events if (now - t) <= window_sec]
            tokens_in_window = sum(n for (_t, n) in token_events)
            triggered = tokens_in_window > cost_max_total_tokens

            iter_out["cost_guard"] = {
                "window_minutes": int(cost_window_minutes),
                "max_total_tokens": int(cost_max_total_tokens),
                "tokens_in_window": int(tokens_in_window),
                "triggered": bool(triggered),
            }

            if triggered:
                rr = iter_out.get("review_recommendation") or {}
                if not isinstance(rr, dict):
                    rr = {}
                level = str(rr.get("level") or "none")
                if level == "none":
                    rr["level"] = "soft"
                reasons = rr.get("reasons")
                if not isinstance(reasons, list):
                    reasons = []
                reasons.append(f"cost_guard_tokens_in_window={tokens_in_window}>{cost_max_total_tokens}")
                rr["reasons"] = [str(x) for x in reasons if x]
                iter_out["review_recommendation"] = rr

        step_record: Dict[str, Any] = {
            "step": step_idx,
            "iteration": step_idx,
            "autopilot": iter_out,
            "review_job": None,
            "escalation_review_job": None,
            "fix_tasks_ran": [],
            "decision_log": step_decision_log,
            "notes": step_notes,
        }

        # Save a step log under the workspace and register it as an artifact.
        step_rel = f"{_run_dir}/step_{step_idx:03d}/step.json"
        put_artifact(
            ledger=ledger,
            project=get_project(ledger, project_id),
            relative_path=step_rel,
            content=json.dumps(step_record, ensure_ascii=False, indent=2) + "\n",
            mode="overwrite",
            kind="agent_step_json",
        )

        # --- Do-phase failure: schedule retry with same plan, skip review ---
        if iter_out.get("do_phase_failed") and not dry_run:
            plan_actions = (iter_out.get("plan") or {}).get("actions") or []
            if do_retry_streak < do_retry_max:
                do_retry_streak += 1
                do_retry_actions = [a for a in plan_actions if isinstance(a, dict)]
                step_record["do_phase_retry"] = {
                    "scheduled": True,
                    "streak": do_retry_streak,
                    "max": do_retry_max,
                    "rate_limited": bool(iter_out.get("do_phase_rate_limited")),
                }
                log.warning(
                    "Do-phase failed (streak %d/%d). Will retry with same plan after backoff.",
                    do_retry_streak, do_retry_max,
                )
            else:
                step_record["do_phase_retry"] = {
                    "scheduled": False,
                    "streak": do_retry_streak,
                    "max": do_retry_max,
                    "exhausted": True,
                }
                stopped_reason = "do_phase_retry_exhausted"
                log.error(
                    "Do-phase failed %d consecutive times. Stopping agent loop.",
                    do_retry_streak,
                )
            # Overwrite step log with retry info.
            put_artifact(
                ledger=ledger,
                project=get_project(ledger, project_id),
                relative_path=step_rel,
                content=json.dumps(step_record, ensure_ascii=False, indent=2) + "\n",
                mode="overwrite",
                kind="agent_step_json",
            )
            steps.append(step_record)
            if do_retry_actions is not None:
                continue  # retry next iteration
            break  # exhausted retries

        # Reset do-retry streak on successful Do phase.
        if do_retry_streak > 0:
            log.info("Do-phase succeeded after %d retries — resetting streak.", do_retry_streak)
        do_retry_streak = 0

        # Update pivot stagnation streak.
        _rr_reasons = (iter_out.get("review_recommendation") or {}).get("reasons") or []
        if not isinstance(_rr_reasons, list):
            _rr_reasons = []
        _pivot_fired = any(str(r).startswith("pivot_no_improvement(") for r in _rr_reasons)
        if _pivot_fired:
            pivot_stagnation_streak += 1
            log.info("Pivot no-improvement fired (streak %d/%s).", pivot_stagnation_streak, force_stop_after or "disabled")
        else:
            if pivot_stagnation_streak > 0:
                log.info("Pivot no-improvement cleared — resetting streak from %d.", pivot_stagnation_streak)
            pivot_stagnation_streak = 0

        # --- Force-stop: fail_streak exceeded threshold ---
        if fail_streak_force_stop > 0:
            _streaks = iter_out.get("failure_streaks") or {}
            _fs_any = int(_streaks.get("any", 0)) if isinstance(_streaks, dict) else 0
            if _fs_any >= fail_streak_force_stop:
                stopped_reason = f"fail_streak_force_stop(any={_fs_any},threshold={fail_streak_force_stop})"
                log.warning(
                    "Fail streak %d >= force_stop_after %d — stopping agent loop.",
                    _fs_any, fail_streak_force_stop,
                )
                steps.append(step_record)
                break

        # --- Update metric watchdog streaks ---
        _cur_metric = _read_primary_metric_mean()
        if _cur_metric is None:
            _null_metric_streak += 1
            _unchanged_metric_streak = 0  # null is a separate condition
            log.info(
                "Metric watchdog: primary_metric.current.mean is null "
                "(null_streak=%d/%s).",
                _null_metric_streak, _null_metric_max or "disabled",
            )
        else:
            if _null_metric_streak > 0:
                log.info(
                    "Metric watchdog: null streak cleared (was %d), "
                    "metric now %.6f.",
                    _null_metric_streak, _cur_metric,
                )
            _null_metric_streak = 0
            if _prev_metric_value is not None and _cur_metric == _prev_metric_value:
                _unchanged_metric_streak += 1
                log.info(
                    "Metric watchdog: primary metric unchanged at %.6f "
                    "(unchanged_streak=%d/%s).",
                    _cur_metric, _unchanged_metric_streak,
                    _unchanged_metric_max or "disabled",
                )
            else:
                if _unchanged_metric_streak > 0:
                    log.info(
                        "Metric watchdog: metric changed %.6f -> %.6f, "
                        "resetting unchanged streak from %d.",
                        _prev_metric_value or 0.0, _cur_metric,
                        _unchanged_metric_streak,
                    )
                _unchanged_metric_streak = 0
        _prev_metric_value = _cur_metric

        _review_rec = iter_out.get("review_recommendation") or {}
        rec = _review_rec.get("level")
        _review_deferred = bool(_review_rec.get("deferred_due_to_do_phase_failure"))
        if not dry_run and rec in {"soft", "hard"} and not _review_deferred:
            post_exec_cfg = (policy.get("review_phases") or {}).get("post_exec") or {}
            if not isinstance(post_exec_cfg, dict):
                post_exec_cfg = {}
            dual_on_hard = bool(post_exec_cfg.get("dual_on_hard", False))

            primary_cfg = _pick_reviewer(policy, "soft")
            escalation_cfg = _pick_reviewer(policy, "hard")

            reviewer_cfg = primary_cfg if (str(rec) == "hard" and dual_on_hard) else _pick_reviewer(policy, str(rec))
            provider = str(reviewer_cfg.get("provider") or "openai")
            reviewer_model = reviewer_cfg.get("model")
            stage = str(get_project(ledger, project_id).get("stage") or "analysis")
            targets = (iter_out.get("review_recommendation") or {}).get("targets") or []
            if not isinstance(targets, list):
                targets = []
            targets = [str(x) for x in targets if x]
            if not targets:
                # Fallback: use recent changed files from iteration output.
                _changed = (iter_out.get("git_change_summary") or {}).get("changed_paths") or []
                targets = [str(x) for x in _changed[:20] if x]
            questions = cfg.review_questions or ["Any major issues? Any missing baselines/related work?"]

            # Extract challenger flags for review prompt injection
            challenger_result = iter_out.get("interpretation_challenger") or {}
            challenger_flags_list: list = []
            if (
                challenger_result.get("enabled")
                and str(challenger_result.get("overall_concern_level") or "").strip().lower()
                in {"medium", "high"}
            ):
                _cflags = challenger_result.get("flags")
                if isinstance(_cflags, list) and _cflags:
                    challenger_flags_list = [str(f) for f in _cflags[:10] if f]

            created_at = utc_now_iso()
            if not targets:
                log.warning("Review skipped: no targets available (step %d).", step_num)
                step_record["review_skipped"] = "empty_targets"
            else:
                job = create_job(
                    ledger=ledger,
                    project_id=project_id,
                    provider=provider,
                    kind="review",
                    spec={
                        "stage": stage,
                        "targets": targets,
                        "questions": questions,
                        "reviewer": provider,
                        **({"model": reviewer_model} if reviewer_model else {}),
                        **({"reasoning_effort": reviewer_cfg.get("reasoning_effort")} if reviewer_cfg.get("reasoning_effort") else {}),
                        **({"challenger_flags": challenger_flags_list} if challenger_flags_list else {}),
                        "background": False,
                    },
                )
                job = run_job(ledger=ledger, job_id=job["id"])
                step_record["review_job"] = job

                if str(rec) == "hard" and dual_on_hard:
                    esc_provider = str(escalation_cfg.get("provider") or "").strip()
                    esc_model = escalation_cfg.get("model")
                    if esc_provider:
                        esc_job = create_job(
                            ledger=ledger,
                            project_id=project_id,
                            provider=esc_provider,
                            kind="review",
                            spec={
                                "stage": stage,
                                "targets": targets,
                                "questions": questions,
                                "reviewer": esc_provider,
                                **({"model": esc_model} if esc_model else {}),
                                **({"reasoning_effort": escalation_cfg.get("reasoning_effort")} if escalation_cfg.get("reasoning_effort") else {}),
                                **({"challenger_flags": challenger_flags_list} if challenger_flags_list else {}),
                                "background": False,
                            },
                        )
                        esc_job = run_job(ledger=ledger, job_id=esc_job["id"])
                        step_record["escalation_review_job"] = esc_job

                # Lightweight retry: when all findings are minor/nit and code-level,
                # create review_fix tasks inline, run them via Codex, then re-run
                # the last shell_exec actions — skipping the Planner entirely.
                if lr_enabled and lr_max_consecutive > 0:
                    rr = _extract_review_result(job)
                    findings = rr.get("findings") or []
                    if not isinstance(findings, list):
                        findings = []
                    findings = [f for f in findings if isinstance(f, dict)]
                    if _is_lightweight_fixable(findings):
                        shell_actions = _get_last_shell_actions(iter_out)
                        if shell_actions and lightweight_retry_streak < lr_max_consecutive:
                            # Create and run inline fix tasks for code-level findings.
                            project = get_project(ledger, project_id)
                            for fidx, f in enumerate(findings[: cfg.max_fix_tasks_per_review]):
                                fix_task = create_task(
                                    ledger=ledger,
                                    project_id=project_id,
                                    task_type="review_fix",
                                    spec={
                                        "source": "lightweight_retry",
                                        "finding_index": fidx,
                                        "stage": stage,
                                        "severity": f.get("severity"),
                                        "category": f.get("category"),
                                        "message": f.get("message"),
                                        "target_paths": f.get("target_paths") or [],
                                        "suggested_fix": f.get("suggested_fix"),
                                    },
                                )
                                fix_out = run_task(ledger=ledger, project=project, task=fix_task)
                                step_record["fix_tasks_ran"].append(fix_out["task"])

                            lightweight_retry_streak += 1
                            rerun_shell_actions = shell_actions
                            step_record["lightweight_retry"] = {
                                "scheduled": True,
                                "streak": int(lightweight_retry_streak),
                                "max_consecutive": int(lr_max_consecutive),
                            }
                        else:
                            step_record["lightweight_retry"] = {
                                "scheduled": False,
                                "streak": int(lightweight_retry_streak),
                                "max_consecutive": int(lr_max_consecutive),
                                "skipped_reason": "max_consecutive_reached" if lightweight_retry_streak >= lr_max_consecutive else "no_shell_actions",
                            }
                            lightweight_retry_streak = 0
                    else:
                        lightweight_retry_streak = 0

            # Overwrite step log with review + fixes.
            project = get_project(ledger, project_id)
            put_artifact(
                ledger=ledger,
                project=project,
                relative_path=step_rel,
                content=json.dumps(step_record, ensure_ascii=False, indent=2) + "\n",
                mode="overwrite",
                kind="agent_step_json",
            )

        # --- Auto stage update ---
        if auto_stage_enabled and not dry_run:
            stage_requested = (iter_out.get("project_stage") or {}).get("requested")
            if isinstance(stage_requested, str) and stage_requested.strip():
                stage_requested = stage_requested.strip()
                prev_stage = str(get_project(ledger, project_id).get("stage") or "")
                if stage_requested != prev_stage:
                    backward = _is_stage_backward(prev_stage, stage_requested)
                    if backward:
                        # Backward: apply unconditionally.
                        set_project_stage(ledger, project_id, stage_requested)
                        step_record["stage_update"] = {
                            "applied": True,
                            "from": prev_stage,
                            "to": stage_requested,
                            "direction": "backward",
                        }
                    else:
                        # Forward: apply only if review passed.
                        # When dual_on_hard is active, use worst-of-both-reviews.
                        review_rec = ""
                        if step_record.get("review_job"):
                            rr = _extract_review_result(step_record["review_job"])
                            review_rec = str(rr.get("recommendation") or "").strip().lower()
                        if step_record.get("escalation_review_job"):
                            esc_rr = _extract_review_result(step_record["escalation_review_job"])
                            esc_rec = str(esc_rr.get("recommendation") or "").strip().lower()
                            # Worst-of policy: if escalation rejects, block stage advance.
                            if esc_rec and esc_rec not in auto_stage_apply_on_set:
                                review_rec = esc_rec
                        if review_rec in auto_stage_apply_on_set:
                            set_project_stage(ledger, project_id, stage_requested)
                            step_record["stage_update"] = {
                                "applied": True,
                                "from": prev_stage,
                                "to": stage_requested,
                                "direction": "forward",
                                "trigger": f"review_{review_rec}",
                            }
                        else:
                            step_record["stage_update"] = {
                                "applied": False,
                                "from": prev_stage,
                                "requested": stage_requested,
                                "direction": "forward",
                                "reason": f"review_recommendation={review_rec or 'none'}",
                            }

        # Write step.json again if stage was updated (stage update happens after the previous write).
        if step_record.get("stage_update"):
            project = get_project(ledger, project_id)
            put_artifact(
                ledger=ledger,
                project=project,
                relative_path=step_rel,
                content=json.dumps(step_record, ensure_ascii=False, indent=2) + "\n",
                mode="overwrite",
                kind="agent_step_json",
            )

        steps.append(step_record)

        if rerun_shell_actions is not None:
            # Force a lightweight retry iteration next, even if the plan suggested stopping.
            continue

        plan = iter_out.get("plan") or {}
        if isinstance(plan, dict) and bool(plan.get("should_stop")):
            stopped_reason = str(plan.get("stop_reason") or "should_stop")
            break

    return {
        "project_id": project_id,
        "objective": objective,
        "steps": steps,
        "decision_log": decision_log,
        "stopped_reason": stopped_reason,
        "dry_run": dry_run,
        "config": cfg.__dict__,
    }
