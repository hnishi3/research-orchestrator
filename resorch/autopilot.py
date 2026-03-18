from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.artifacts import put_artifact
from resorch.ledger import Ledger
from resorch.jobs import create_job, poll_job as poll_job_fn, run_job
from resorch.projects import get_project
from resorch.tasks import create_task, list_tasks, run_task
from resorch.utils import read_text, utc_now_iso
from resorch.visual_inspection import get_visual_inspection_status

log = logging.getLogger(__name__)


from resorch.autopilot_config import (  # noqa: E402
    ReviewRecommendation,
    _load_yaml,
    load_review_policy,
    load_pivot_policy,
    load_plan_schema,
)


from resorch.autopilot_git import (  # noqa: E402
    _git,
    _parse_numstat,
    _ensure_git_baseline,
    _is_review_excluded,
    _list_git_changed_paths,
    compute_git_change_summary,
    _REVIEW_EXCLUDE_PREFIXES,
    _REVIEW_EXCLUDE_SEGMENTS,
)


from resorch.autopilot_review import (  # noqa: E402
    _parse_iso_z,
    compute_failure_streaks,
    _detect_external_fetch,
    _any_path_matches,
    _list_ready_stage_transitions,
    _claims_created_since,
    _any_stalled_jobs,
    _list_pending_external_jobs,
    recommend_review_from_policy,
)


from resorch.autopilot_planner import (  # noqa: E402
    _load_context_files,
    _pick_recent_paths,
    _default_planner_context_files,
    _load_playbook_context,
    _compact_json_schema,
    _build_planner_prompt,
    generate_plan_openai,
    generate_plan_claude,
    generate_plan_codex,
)


from resorch.autopilot_action import (  # noqa: E402
    _normalize_action_spec,
    _validate_normalized_action_spec,
    _validate_action_spec,
    _validate_plan_action_semantics,
    _repair_planned_action_for_runtime,
    _repair_plan_actions_for_runtime,
    _EMBEDDED_PYTHON_RE,
    _SHELL_PROMOTE_LINE_THRESHOLD,
    _WORKSPACE_SCRIPT_RE,
    _should_promote_to_codex,
    PRE_EXEC_REVIEW_INSTRUCTIONS,
    PRE_EXEC_REVIEW_INSTRUCTIONS_CODEX,
    _render_pre_exec_review_instructions,
    _maybe_inject_pre_exec_review,
    _promote_shell_to_codex,
    _inject_shell_init,
    _inject_shell_init_into_codex,
)


from resorch.autopilot_pivot import (  # noqa: E402
    _nested_get,
    _as_float,
    _pivot_no_improvement_trigger,
)


from resorch.autopilot_digests import (  # noqa: E402
    _as_compact_task_ref,
    summarize_pre_exec_reviews,
    summarize_codex_exec_statuses,
    _update_pdca_digests,
    _update_exploration_log,
    _write_last_errors,
    _write_last_challenger,
)


def _extract_review_result(job: Dict[str, Any]) -> Dict[str, Any]:
    result = job.get("result") if isinstance(job, dict) else None
    if not isinstance(result, dict):
        return {}
    review_result = result.get("review_result")
    if isinstance(review_result, dict):
        return review_result
    # Fallback path: when Claude CLI fails schema validation, the system
    # retries via an OpenAI reformatter.  The result lives under
    # result.fallback_job_result.review_result.
    fb = result.get("fallback_job_result")
    if isinstance(fb, dict):
        review_result = fb.get("review_result")
        if isinstance(review_result, dict):
            return review_result
    return {}


def _run_review_fix_tasks_created_since(
    *,
    ledger: Ledger,
    project: Dict[str, Any],
    created_at: str,
    max_tasks: int,
) -> List[Dict[str, Any]]:
    project_id = str(project.get("id") or "")
    fixes = [
        t
        for t in list_tasks(ledger, project_id=project_id)
        if t.get("type") == "review_fix" and t.get("status") == "created" and str(t.get("updated_at") or "") >= created_at
    ]
    ran: List[Dict[str, Any]] = []
    for t in fixes[: max(0, int(max_tasks))]:
        out = run_task(ledger=ledger, project=project, task=t)
        ran.append(out["task"])
    return ran


def _run_post_step_verifier_best_effort(
    *,
    workspace: Path,
    ledger: Ledger,
    project_id: str,
) -> Dict[str, Any]:
    """Run post-step verifier in fail-open mode."""
    try:
        from resorch.verifier_loop import run_post_step_verification

        return run_post_step_verification(workspace=workspace, ledger=ledger, project_id=project_id)
    except Exception as e:  # noqa: BLE001
        log.warning("Post-step verification failed: %s", e)
        return {}


def run_autopilot_iteration(
    *,
    ledger: Ledger,
    project_id: str,
    objective: str,
    model: str,
    iteration: int,
    dry_run: bool,
    max_actions: int,
    background: bool,
    config: Optional[Dict[str, Any]] = None,
    rerun_mode: bool = False,
    rerun_actions: Optional[List[Dict[str, Any]]] = None,
    retry_mode: bool = False,
    retry_actions: Optional[List[Dict[str, Any]]] = None,
    reuse_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    started_at = utc_now_iso()
    project = get_project(ledger, project_id)
    prev_stage = str(project.get("stage") or "")

    pending_before = _list_pending_external_jobs(ledger, project_id=project_id)
    if pending_before and not dry_run:
        # Best-effort: poll external compute jobs before planning.
        for j in pending_before:
            try:
                poll_job_fn(ledger=ledger, job_id=str(j["id"]))
            except Exception:  # noqa: BLE001
                # Keep the loop running even if polling fails for one job.
                continue

    pending_after = _list_pending_external_jobs(ledger, project_id=project_id)
    if pending_after:
        ts = utc_now_iso().replace(":", "").replace("Z", "Z")
        stop_reason = f"external_jobs_pending count={len(pending_after)}"
        plan = {
            "plan_id": f"wait-{ts}",
            "project_id": str(project_id),
            "iteration": int(iteration),
            "objective": str(objective),
            "self_confidence": 1.0,
            "evidence_strength": 1.0,
            "actions": [],
            "should_stop": True,
            "stop_reason": stop_reason,
            "next_stage": prev_stage,
        }
        plan_path = f"notes/autopilot/plan-{iteration:03d}-{ts}.json"
        put_artifact(
            ledger=ledger,
            project=project,
            relative_path=plan_path,
            content=json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
            mode="overwrite",
            kind="autopilot_plan_json",
        )

        workspace = Path(project["repo_path"]).resolve()
        policy = load_review_policy(ledger.paths.root, workspace=workspace)
        default_targets = list((policy.get("targets") or {}).get("default") or [])
        default_targets = [str(x) for x in default_targets if x]
        git_summary = compute_git_change_summary(workspace)
        streaks = compute_failure_streaks(ledger, project_id=project_id)

        recommendation = ReviewRecommendation(level="none", reasons=[stop_reason], targets=default_targets)
        _update_pdca_digests(
            ledger=ledger,
            project=project,
            iteration=iteration,
            started_at=started_at,
            plan_artifact_path=plan_path,
            tasks_created=[],
            tasks_ran=[],
            git_change_summary=git_summary,
            review_recommendation=recommendation,
        )
        verifier_result = _run_post_step_verifier_best_effort(
            workspace=workspace,
            ledger=ledger,
            project_id=project_id,
        )

        return {
            "plan": plan,
            "plan_artifact_path": plan_path,
            "planner_meta": {"skipped": True, "reason": stop_reason, "pending_jobs": pending_after},
            "goal_alignment": {"enabled": False, "aligned": True, "drift_summary": None, "method": "skipped"},
            "interpretation_challenger": {"enabled": False},
            "started_at": started_at,
            "project_stage": {"before": prev_stage, "after": prev_stage, "changed": False, "requested": None},
            "tasks_created": [],
            "tasks_ran": [],
            "codex_exec_summary": summarize_codex_exec_statuses([]),
            "pre_exec_review_summary": summarize_pre_exec_reviews([]),
            "git_change_summary": git_summary,
            "failure_streaks": streaks,
            "stalled_jobs": [],
            "ready_stage_transitions": [],
            "review_recommendation": {"level": recommendation.level, "reasons": recommendation.reasons, "targets": recommendation.targets},
            "pending_external_jobs": pending_after,
            "verifier_result": verifier_result,
        }

    policy_now = load_review_policy(ledger.paths.root, workspace=Path(project["repo_path"]).resolve())
    vis = get_visual_inspection_status(policy=policy_now, workspace=Path(project["repo_path"]).resolve())
    if vis.enabled and vis.pending_figures:
        ts = utc_now_iso().replace(":", "").replace("Z", "Z")
        stop_reason = f"visual_inspection_required count={len(vis.pending_figures)} marker={vis.marker_path}"
        plan = {
            "plan_id": f"visual-{ts}",
            "project_id": str(project_id),
            "iteration": int(iteration),
            "objective": str(objective),
            "self_confidence": 1.0,
            "evidence_strength": 1.0,
            "actions": [],
            "should_stop": True,
            "stop_reason": stop_reason,
            "next_stage": prev_stage,
        }
        plan_path = f"notes/autopilot/plan-{iteration:03d}-{ts}.json"
        put_artifact(
            ledger=ledger,
            project=project,
            relative_path=plan_path,
            content=json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
            mode="overwrite",
            kind="autopilot_plan_json",
        )

        workspace = Path(project["repo_path"]).resolve()
        default_targets = list((policy_now.get("targets") or {}).get("default") or [])
        default_targets = [str(x) for x in default_targets if x]
        git_summary = compute_git_change_summary(workspace)
        streaks = compute_failure_streaks(ledger, project_id=project_id)

        recommendation = ReviewRecommendation(level="hard", reasons=[stop_reason], targets=default_targets)
        _update_pdca_digests(
            ledger=ledger,
            project=project,
            iteration=iteration,
            started_at=started_at,
            plan_artifact_path=plan_path,
            tasks_created=[],
            tasks_ran=[],
            git_change_summary=git_summary,
            review_recommendation=recommendation,
        )
        verifier_result = _run_post_step_verifier_best_effort(
            workspace=workspace,
            ledger=ledger,
            project_id=project_id,
        )

        return {
            "plan": plan,
            "plan_artifact_path": plan_path,
            "planner_meta": {"skipped": True, "reason": stop_reason, "pending_figures": vis.pending_figures},
            "goal_alignment": {"enabled": False, "aligned": True, "drift_summary": None, "method": "skipped"},
            "interpretation_challenger": {"enabled": False},
            "started_at": started_at,
            "project_stage": {"before": prev_stage, "after": prev_stage, "changed": False, "requested": None},
            "tasks_created": [],
            "tasks_ran": [],
            "codex_exec_summary": summarize_codex_exec_statuses([]),
            "pre_exec_review_summary": summarize_pre_exec_reviews([]),
            "git_change_summary": git_summary,
            "failure_streaks": streaks,
            "stalled_jobs": [],
            "ready_stage_transitions": [],
            "review_recommendation": {"level": recommendation.level, "reasons": recommendation.reasons, "targets": recommendation.targets},
            "pending_visual_inspection": {"marker_path": vis.marker_path, "pending_figures": vis.pending_figures},
            "verifier_result": verifier_result,
        }

    workspace = Path(project["repo_path"]).resolve()

    # Snapshot workspace so git diff reflects only THIS iteration's changes.
    if not dry_run:
        _ensure_git_baseline(workspace, iteration=iteration)

    # --- Goal alignment check (fail-open) ---
    goal_cfg = policy_now.get("goal_alignment") or {}
    goal_enabled = bool(goal_cfg.get("enabled", False)) if isinstance(goal_cfg, dict) else False
    goal_alignment: Dict[str, Any] = {"enabled": goal_enabled, "aligned": True, "drift_summary": None, "method": "skipped"}
    if not rerun_mode and not retry_mode and reuse_plan is None and goal_enabled and not dry_run:
        try:
            source_file = str(goal_cfg.get("source_file") or "notes/problem.md")
            src = Path(source_file)
            if src.is_absolute():
                src_path = src.resolve()
            else:
                src_path = (workspace / src).resolve()
            src_path.relative_to(workspace)

            rq_text = read_text(src_path).strip()
            if rq_text:
                from resorch.goal_alignment import check_goal_alignment

                ar = check_goal_alignment(
                    research_question=rq_text,
                    recent_objectives=[objective],
                    provider=str(goal_cfg.get("provider") or "claude_code_cli"),
                    model=str(goal_cfg.get("model") or "haiku"),
                    workspace_dir=workspace,
                    reasoning_effort=goal_cfg.get("reasoning_effort"),
                )
                goal_alignment = {
                    "enabled": True,
                    "aligned": bool(ar.aligned),
                    "drift_summary": ar.drift_summary,
                    "method": ar.method,
                }
        except Exception:  # noqa: BLE001
            goal_alignment = {"enabled": goal_enabled, "aligned": True, "drift_summary": None, "method": "skipped"}
    elif (rerun_mode or retry_mode or reuse_plan is not None) and goal_enabled:
        goal_alignment["skipped_reason"] = "do_phase_retry" if retry_mode else ("reuse_last_plan" if reuse_plan is not None else "lightweight_retry")

    if retry_mode:
        ts = utc_now_iso().replace(":", "").replace("Z", "Z")
        plan = {
            "plan_id": f"retry-{ts}",
            "project_id": str(project_id),
            "iteration": int(iteration),
            "objective": str(objective),
            "self_confidence": 1.0,
            "evidence_strength": 1.0,
            "actions": [a for a in (retry_actions or []) if isinstance(a, dict)],
            "should_stop": False,
            "stop_reason": None,
            "next_stage": prev_stage,
        }
        planner_meta: Dict[str, Any] = {"skipped": True, "reason": "do_phase_retry", "retry_mode": True}
    elif rerun_mode:
        ts = utc_now_iso().replace(":", "").replace("Z", "Z")
        plan = {
            "plan_id": f"rerun-{ts}",
            "project_id": str(project_id),
            "iteration": int(iteration),
            "objective": str(objective),
            "self_confidence": 1.0,
            "evidence_strength": 1.0,
            "actions": [a for a in (rerun_actions or []) if isinstance(a, dict)],
            "should_stop": False,
            "stop_reason": None,
            "next_stage": prev_stage,
        }
        planner_meta = {"skipped": True, "reason": "lightweight_retry", "rerun_mode": True}
    elif reuse_plan is not None:
        # --reuse-last-plan: use the provided plan directly, skipping the Planner.
        ts = utc_now_iso().replace(":", "").replace("Z", "Z")
        plan = dict(reuse_plan)
        plan["plan_id"] = f"reuse-{ts}"
        plan["iteration"] = int(iteration)
        plan["project_id"] = str(project_id)
        plan["objective"] = str(objective)
        # Clear execution-state fields from previous run to avoid stale stop signals.
        plan.pop("should_stop", None)
        plan.pop("stop_reason", None)
        planner_meta = {"skipped": True, "reason": "reuse_last_plan", "reuse_plan": True}
        log.info("Reusing plan from previous run (skipped Planner). Actions: %d", len(plan.get("actions") or []))
    else:
        # Determine planner provider from config.
        planner_provider = "claude_code_cli"
        if isinstance(config, dict):
            planner_provider = str(config.get("planner_provider", "claude_code_cli")).strip().lower()

        if planner_provider == "claude_code_cli":
            plan, planner_meta = generate_plan_claude(
                ledger=ledger,
                project_id=project_id,
                objective=objective,
                model=model,
                iteration=iteration,
                max_actions=max_actions,
                config=config,
                goal_alignment=goal_alignment,
            )
        elif planner_provider == "codex_cli":
            plan, planner_meta = generate_plan_codex(
                ledger=ledger,
                project_id=project_id,
                objective=objective,
                model=model,
                iteration=iteration,
                max_actions=max_actions,
                config=config,
                goal_alignment=goal_alignment,
            )
        else:
            plan, planner_meta = generate_plan_openai(
                ledger=ledger,
                project_id=project_id,
                objective=objective,
                model=model,
                iteration=iteration,
                max_actions=max_actions,
                background=background,
                config=config,
                goal_alignment=goal_alignment,
            )

    plan = _repair_plan_actions_for_runtime(plan)

    next_stage = plan.get("next_stage")
    stage_request: Optional[str] = None
    if isinstance(next_stage, str):
        next_stage = next_stage.strip()
        if next_stage and next_stage != prev_stage:
            # Treat as a suggestion only. Stage updates are a meaningful decision
            # and should be applied explicitly (e.g., after review / human check).
            stage_request = next_stage

    ts = utc_now_iso().replace(":", "").replace("Z", "Z")
    plan_path = f"notes/autopilot/plan-{iteration:03d}-{ts}.json"
    put_artifact(
        ledger=ledger,
        project=project,
        relative_path=plan_path,
        content=json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        mode="overwrite",
        kind="autopilot_plan_json",
    )

    created: List[Dict[str, Any]] = []
    ran: List[Dict[str, Any]] = []

    workspace = Path(project["repo_path"]).resolve()
    policy = policy_now  # reuse the policy loaded once at iteration start (L259)

    review_phases = policy.get("review_phases") or {}
    if not isinstance(review_phases, dict):
        review_phases = {}
    pre_exec_cfg = review_phases.get("code_review_gate") or review_phases.get("pre_exec") or {}
    if not isinstance(pre_exec_cfg, dict):
        pre_exec_cfg = {}
    pre_exec_enabled = bool(pre_exec_cfg.get("enabled", False)) and not rerun_mode and not retry_mode
    pre_exec_review: Dict[str, Any] = {"enabled": pre_exec_enabled, "jobs": [], "fix_tasks_ran": [], "final_recommendation": None}

    actions = [a for a in (plan.get("actions") or []) if isinstance(a, dict)]

    # Auto-promote long shell_exec with embedded Python → codex_exec.
    actions = [
        _promote_shell_to_codex(a) if _should_promote_to_codex(a) else a
        for a in actions
    ]

    if not pre_exec_enabled:
        # Backward-compatible: run actions in plan order.
        for a in actions:
            task_type = str(a.get("task_type") or "").strip()
            if rerun_mode and task_type != "shell_exec":
                continue
            spec = a.get("spec") or {}
            if not isinstance(spec, dict):
                continue
            spec_norm = _normalize_action_spec(
                task_type=task_type,
                spec=spec,
                action_title=str(a.get("title") or ""),
            )
            spec_err = _validate_normalized_action_spec(task_type=task_type, spec=spec_norm)
            if spec_err:
                log.error("Skipping invalid planned action (%s): %s", task_type, spec_err)
                continue
            spec_norm = _maybe_inject_pre_exec_review(spec=spec_norm, task_type=task_type, policy=policy)
            shell_init = (config or {}).get("shell_init")
            if shell_init:
                if task_type == "shell_exec":
                    spec_norm = _inject_shell_init(spec_norm, shell_init)
                elif task_type == "codex_exec":
                    spec_norm = _inject_shell_init_into_codex(spec_norm, shell_init)

            task = create_task(ledger=ledger, project_id=project_id, task_type=task_type, spec=spec_norm)
            created.append(task)
            if not dry_run:
                try:
                    ran.append(run_task(ledger=ledger, project=project, task=task)["task"])
                except Exception as exc:  # noqa: BLE001
                    log.error("task %s (%s) failed: %s", task.get("id"), task_type, exc)
                    try:
                        ledger.update_task_status(task["id"], "failed")
                    except Exception:  # noqa: BLE001
                        log.warning("Failed to mark task %s as failed in ledger", task.get("id"))
    else:
        codex_actions = [a for a in actions if str(a.get("task_type") or "").strip() == "codex_exec"]
        shell_actions = [a for a in actions if str(a.get("task_type") or "").strip() == "shell_exec"]

        # Phase 1: codex_exec (code generation).
        for a in codex_actions:
            spec = a.get("spec") or {}
            if not isinstance(spec, dict):
                continue
            spec_norm = _normalize_action_spec(
                task_type="codex_exec",
                spec=spec,
                action_title=str(a.get("title") or ""),
            )
            spec_err = _validate_normalized_action_spec(task_type="codex_exec", spec=spec_norm)
            if spec_err:
                log.error("Skipping invalid planned codex_exec action: %s", spec_err)
                continue
            spec_norm = _maybe_inject_pre_exec_review(spec=spec_norm, task_type="codex_exec", policy=policy)
            shell_init = (config or {}).get("shell_init")
            if shell_init:
                spec_norm = _inject_shell_init_into_codex(spec_norm, shell_init)
            task = create_task(ledger=ledger, project_id=project_id, task_type="codex_exec", spec=spec_norm)
            created.append(task)
            if not dry_run:
                try:
                    ran.append(run_task(ledger=ledger, project=project, task=task)["task"])
                except Exception as exc:  # noqa: BLE001
                    log.error("codex_exec task %s failed: %s", task.get("id"), exc)
                    try:
                        ledger.update_task_status(task["id"], "failed")
                    except Exception:  # noqa: BLE001
                        log.warning("Failed to mark task %s as failed in ledger", task.get("id"))
                    # Append failed task to ran so downstream logic sees the failure.
                    try:
                        ran.append(get_task(ledger, task["id"]))
                    except Exception:  # noqa: BLE001
                        ran.append({"id": task.get("id"), "type": "codex_exec", "status": "failed"})

        # Detect codex all-fail: skip remaining phases to avoid wasting review calls.
        codex_ran_ok = any(t.get("status") == "success" for t in ran if t.get("type") == "codex_exec")
        codex_all_failed = bool(codex_actions) and not codex_ran_ok and not dry_run and (ran or codex_actions)

        # Phase 2: pre-exec code review (+ optional fix + retry).
        blocked = False
        if codex_all_failed:
            pre_exec_review["skipped_reason"] = "do_phase_codex_all_failed"
            blocked = True
            log.warning("All codex_exec tasks failed — skipping pre-exec review and shell_exec.")
        elif not dry_run and codex_actions:
            targets = _list_git_changed_paths(workspace)
            if targets:
                max_fix_retries = int(pre_exec_cfg.get("max_fix_retries", 1) or 0)
                max_fix_tasks = int(config.get("max_fix_tasks_per_review", 10) if isinstance(config, dict) else 10)

                for attempt in range(max_fix_retries + 1):
                    created_at = utc_now_iso()

                    provider = str(pre_exec_cfg.get("provider") or "claude_code_cli")
                    job_kind = str(pre_exec_cfg.get("kind") or "code_review")
                    job_spec: Dict[str, Any] = {
                        "stage": "pre_exec",
                        "mode": str(pre_exec_cfg.get("mode") or "balanced"),
                        "targets": targets,
                        "questions": [str(x) for x in (pre_exec_cfg.get("questions") or []) if x]
                        or ["Is this safe and correct to execute? List blockers first."],
                        "system_prompt_file": str(pre_exec_cfg.get("system_prompt_file") or "prompts/reviewer_code.md"),
                        "background": False,
                    }
                    if pre_exec_cfg.get("model"):
                        job_spec["model"] = pre_exec_cfg.get("model")

                    job = create_job(ledger=ledger, project_id=project_id, provider=provider, kind=job_kind, spec=job_spec)
                    job = run_job(ledger=ledger, job_id=job["id"])

                    review_result = _extract_review_result(job)
                    rec = str(review_result.get("recommendation") or "").strip().lower()
                    stored_path = _nested_get(job, "result.ingested.stored_path")
                    pre_exec_review["jobs"].append(
                        {
                            "job_id": job.get("id"),
                            "provider": job.get("provider"),
                            "kind": job.get("kind"),
                            "status": job.get("status"),
                            "stored_path": stored_path,
                            "recommendation": rec or None,
                        }
                    )
                    pre_exec_review["final_recommendation"] = rec or None

                    if rec not in {"major", "reject"}:
                        break

                    # Always run fix tasks for the findings just created,
                    # even on the last attempt — otherwise they stay "created" forever.
                    fixes_ran = _run_review_fix_tasks_created_since(
                        ledger=ledger,
                        project=project,
                        created_at=created_at,
                        max_tasks=max_fix_tasks,
                    )
                    pre_exec_review["fix_tasks_ran"].extend(fixes_ran)
                    ran.extend(fixes_ran)

                    if attempt >= max_fix_retries:
                        blocked = True
                        plan["should_stop"] = True
                        plan["stop_reason"] = f"pre_exec_code_review_{rec or 'reject'}"
                        break

                    targets = _list_git_changed_paths(workspace)
                    if not targets:
                        blocked = True
                        plan["should_stop"] = True
                        plan["stop_reason"] = "pre_exec_code_review_no_targets"
                        break
            else:
                pre_exec_review["skipped_reason"] = "no_git_or_no_changes"
        else:
            pre_exec_review["skipped_reason"] = "dry_run_or_no_codex_actions"

        # Phase 3: shell_exec (execution) only if pre-exec review didn't block.
        if not blocked:
            for a in shell_actions:
                spec = a.get("spec") or {}
                if not isinstance(spec, dict):
                    continue
                spec_norm = _normalize_action_spec(
                    task_type="shell_exec",
                    spec=spec,
                    action_title=str(a.get("title") or ""),
                )
                spec_err = _validate_normalized_action_spec(task_type="shell_exec", spec=spec_norm)
                if spec_err:
                    log.error("Skipping invalid planned shell_exec action: %s", spec_err)
                    continue
                shell_init = (config or {}).get("shell_init")
                if shell_init:
                    spec_norm = _inject_shell_init(spec_norm, shell_init)
                task = create_task(ledger=ledger, project_id=project_id, task_type="shell_exec", spec=spec_norm)
                created.append(task)
                if not dry_run:
                    try:
                        ran.append(run_task(ledger=ledger, project=project, task=task)["task"])
                    except Exception as exc:  # noqa: BLE001
                        log.error("shell_exec task %s failed: %s", task.get("id"), exc)
                        try:
                            ledger.update_task_status(task["id"], "failed")
                        except Exception:  # noqa: BLE001
                            log.warning("Failed to mark task %s as failed in ledger", task.get("id"))
                        # Append failed task to ran so downstream logic sees the failure.
                        try:
                            ran.append(get_task(ledger, task["id"]))
                        except Exception:  # noqa: BLE001
                            ran.append({"id": task.get("id"), "type": "shell_exec", "status": "failed"})

    default_targets = list((policy.get("targets") or {}).get("default") or [])
    default_targets = [str(x) for x in default_targets if x]

    # --- Do-phase failure detection (unified for both paths) ---
    _codex_in_plan = [a for a in actions if str(a.get("task_type") or "").strip() == "codex_exec"]
    _codex_ran_tasks = [t for t in ran if t.get("type") == "codex_exec"]
    _codex_any_ok = any(t.get("status") == "success" for t in _codex_ran_tasks)
    do_phase_failed = bool(_codex_in_plan) and not _codex_any_ok and not dry_run and bool(_codex_ran_tasks)
    # Also detect rate-limited specifically.
    _codex_rate_limited = any(t.get("status") == "rate_limited" for t in _codex_ran_tasks)
    if do_phase_failed:
        log.warning(
            "Do-phase failed: %d/%d codex_exec tasks failed (rate_limited=%s). "
            "Skipping post-Do phases to save Planner calls.",
            len(_codex_ran_tasks), len(_codex_in_plan), _codex_rate_limited,
        )

    # --- Interpretation Challenger (fail-open) ---
    challenger_cfg = policy.get("interpretation_challenger") or {}
    if not isinstance(challenger_cfg, dict):
        challenger_cfg = {}
    challenger_enabled = bool(challenger_cfg.get("enabled", False))
    challenger_escalate_on = str(challenger_cfg.get("escalate_on") or "high").strip().lower()
    challenger_result: Dict[str, Any] = {"enabled": challenger_enabled, "skipped_reason": None}
    if do_phase_failed:
        challenger_result["skipped_reason"] = "do_phase_failed"
    elif challenger_enabled and not dry_run:
        any_exec_success = any(
            t.get("status") == "success" and t.get("type") in {"shell_exec", "codex_exec"}
            for t in ran
        )
        if any_exec_success:
            from resorch.interpretation_challenger import maybe_challenge_interpretation_from_workspace

            cr = maybe_challenge_interpretation_from_workspace(
                workspace_dir=workspace,
                provider=str(challenger_cfg.get("provider") or "claude_code_cli"),
                model=str(challenger_cfg.get("model") or "sonnet"),
                system_prompt_file=str(challenger_cfg.get("system_prompt_file") or "prompts/challenger.md"),
                reasoning_effort=challenger_cfg.get("reasoning_effort"),
            )
            if cr is None:
                challenger_result["skipped_reason"] = "no_scoreboard"
            else:
                challenger_result = {
                    "enabled": True,
                    "overall_concern_level": cr.overall_concern_level,
                    "flags": cr.flags,
                    "checks": [{"item": c.item, "status": c.status, "reason": c.reason} for c in cr.checks],
                }
        else:
            challenger_result["skipped_reason"] = "no_exec_success"
    elif challenger_enabled:
        challenger_result["skipped_reason"] = "dry_run"

    git_summary = compute_git_change_summary(workspace)
    streaks = compute_failure_streaks(ledger, project_id=project_id)

    stall_minutes = 0
    soft_triggers = policy.get("soft_triggers") or {}
    if isinstance(soft_triggers, dict) and soft_triggers.get("stall_minutes") is not None:
        try:
            stall_minutes = int(soft_triggers.get("stall_minutes") or 0)
        except (TypeError, ValueError):
            stall_minutes = 0
    stalled_jobs = _any_stalled_jobs(ledger, project_id=project_id, stall_minutes=stall_minutes) if stall_minutes else []
    ready_stage_transitions = _list_ready_stage_transitions(ledger, project_id=project_id)
    if stage_request:
        ready_stage_transitions.insert(0, {"name": f"project_stage_request:{prev_stage}->{stage_request}", "decision": "manual"})

    recommendation = recommend_review_from_policy(
        policy=policy,
        plan_self_confidence=plan.get("self_confidence"),
        plan_evidence_strength=plan.get("evidence_strength"),
        git_changed_lines=int(git_summary.get("changed_lines", 0)),
        git_changed_files=int(git_summary.get("changed_files", 0)),
        git_changed_paths=[str(x) for x in (git_summary.get("changed_paths") or []) if x],
        failure_streak_any=int(streaks.get("any", 0)),
        failure_streak_same_task=int(streaks.get("same_task", 0)),
        ready_stage_transitions=ready_stage_transitions,
        claim_created=_claims_created_since(ledger, project_id=project_id, since_iso=started_at),
        external_fetch_detected=_detect_external_fetch(ran),
        stalled_jobs=stalled_jobs,
        default_targets=default_targets,
        plan_suggested=plan.get("suggested_review") if isinstance(plan.get("suggested_review"), dict) else None,
        stage_transition_requested=bool(stage_request),
    )

    if goal_alignment.get("enabled") and not bool(goal_alignment.get("aligned", True)):
        level = recommendation.level
        if level == "none":
            level = "soft"
        drift = str(goal_alignment.get("drift_summary") or "").strip()
        reason = "goal_alignment_drift" + (f": {drift}" if drift else "")
        reasons = list(recommendation.reasons)
        reasons.append(reason)
        recommendation = ReviewRecommendation(level=level, reasons=reasons, targets=list(recommendation.targets))

    if (
        challenger_result.get("enabled")
        and str(challenger_result.get("overall_concern_level") or "").strip().lower() == challenger_escalate_on
        and challenger_escalate_on in {"medium", "high"}
    ):
        level = recommendation.level
        if level == "none":
            level = "soft"
        flags = challenger_result.get("flags") if isinstance(challenger_result.get("flags"), list) else []
        flags_txt = ",".join([str(x) for x in flags[:6] if x])
        reason = f"interpretation_challenger(concern={challenger_escalate_on}" + (f" flags={flags_txt}" if flags_txt else "") + ")"
        reasons = list(recommendation.reasons)
        reasons.append(reason)
        recommendation = ReviewRecommendation(level=level, reasons=reasons, targets=list(recommendation.targets))

    pivot = _pivot_no_improvement_trigger(repo_root=ledger.paths.root, workspace=workspace)
    if pivot:
        pivot_level, pivot_reason = pivot
        level = recommendation.level
        if pivot_level == "hard" and level != "hard":
            level = "hard"
        elif level == "none":
            level = "soft"
        reasons = list(recommendation.reasons)
        reasons.append(pivot_reason)
        recommendation = ReviewRecommendation(level=level, reasons=reasons, targets=list(recommendation.targets))

    # Execution starvation detection: if primary_metric is still null after 2+
    # iterations, the project is likely over-engineering without running experiments.
    if iteration >= 2:
        _sb_path = (workspace / "results" / "scoreboard.json").resolve()
        _pm_val = None
        try:
            _sb_raw = read_text(_sb_path)
            _sb_data = json.loads(_sb_raw) if _sb_raw.strip() else {}
            _pm = _sb_data.get("primary_metric") if isinstance(_sb_data, dict) else None
            _cur = _pm.get("current") if isinstance(_pm, dict) else None
            _pm_val = _cur.get("mean") if isinstance(_cur, dict) else _cur
        except Exception:
            pass
        if _pm_val is None:
            level = recommendation.level
            if level != "hard":
                level = "soft"
            reasons = list(recommendation.reasons)
            reasons.append(
                f"execution_starvation(iteration={iteration}, primary_metric=null): "
                f"project has run {iteration} iterations without producing real experimental results. "
                "Planner should prioritise running the core computation immediately."
            )
            recommendation = ReviewRecommendation(level=level, reasons=reasons, targets=list(recommendation.targets))

    # Metric revision detection: if plan notes contain "metric_revision:", escalate to hard review.
    _plan_notes = str(plan.get("notes") or "") if isinstance(plan, dict) else ""
    if "metric_revision:" in _plan_notes.lower():
        level = "hard"
        reasons = list(recommendation.reasons)
        reasons.append("metric_revision_detected: plan proposes changing the primary metric (auto-escalated to hard review)")
        recommendation = ReviewRecommendation(level=level, reasons=reasons, targets=list(recommendation.targets))

    _update_pdca_digests(
        ledger=ledger,
        project=project,
        iteration=iteration,
        started_at=started_at,
        plan_artifact_path=plan_path,
        tasks_created=created,
        tasks_ran=ran,
        git_change_summary=git_summary,
        review_recommendation=recommendation,
    )

    _update_exploration_log(
        ledger=ledger,
        project=project,
        iteration=iteration,
        plan=plan,
    )

    # --- Write notes/last_errors.md for next Planner (error_block source) ---
    _write_last_errors(workspace=workspace, tasks_ran=ran, ledger=ledger)

    # --- Write notes/last_challenger.md for next Planner (challenger_block source) ---
    _write_last_challenger(workspace=workspace, challenger_result=challenger_result)
    verifier_result = _run_post_step_verifier_best_effort(
        workspace=workspace,
        ledger=ledger,
        project_id=project_id,
    )
    codex_exec_summary = summarize_codex_exec_statuses(ran)
    pre_exec_review_summary = summarize_pre_exec_reviews(ran)

    return {
        "plan": plan,
        "plan_artifact_path": plan_path,
        "planner_meta": planner_meta,
        "pre_exec_review": pre_exec_review,
        "goal_alignment": goal_alignment,
        "interpretation_challenger": challenger_result,
        "started_at": started_at,
        "project_stage": {
            "before": prev_stage,
            "after": str(project.get("stage") or ""),
            "changed": False,
            "requested": stage_request,
        },
        "tasks_created": created,
        "tasks_ran": ran,
        "codex_exec_summary": codex_exec_summary,
        "pre_exec_review_summary": pre_exec_review_summary,
        "git_change_summary": git_summary,
        "failure_streaks": streaks,
        "stalled_jobs": stalled_jobs,
        "ready_stage_transitions": ready_stage_transitions,
        "review_recommendation": {
            "level": recommendation.level,
            "reasons": recommendation.reasons,
            "targets": recommendation.targets,
            "deferred_due_to_do_phase_failure": do_phase_failed,
        },
        "verifier_result": verifier_result,
        "do_phase_failed": do_phase_failed,
        "do_phase_rate_limited": _codex_rate_limited if do_phase_failed else False,
    }
