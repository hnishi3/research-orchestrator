from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator

from resorch.autopilot_config import load_plan_schema
from resorch.autopilot_action import (
    _repair_plan_actions_for_runtime,
    _validate_plan_action_semantics,
)
from resorch.autopilot_review import _parse_iso_z
from resorch.ledger import Ledger
from resorch.openai_tools import extract_json_object_from_response, run_response_to_completion_with_fallback
from resorch.projects import get_project
from resorch.providers.openai import OpenAIClient
from resorch.utils import read_text, utc_now_iso

log = logging.getLogger(__name__)


def _load_context_files(workspace: Path, *, rel_paths: List[str], max_chars: int) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for rel in rel_paths:
        p = (workspace / rel).resolve()
        try:
            p.relative_to(workspace)
        except ValueError:
            continue
        if not p.exists() or not p.is_file():
            continue
        try:
            txt = read_text(p)
        except OSError:
            continue
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "\n…(truncated)…\n"
        out.append((rel, txt))
    return out


def _pick_recent_paths(workspace: Path, *, pattern: str, limit: int) -> List[str]:
    candidates = sorted((workspace.glob(pattern)), key=lambda p: p.name, reverse=True)
    out: List[str] = []
    for p in candidates[: max(0, limit)]:
        try:
            rel = p.resolve().relative_to(workspace.resolve())
        except ValueError:
            continue
        if not p.is_file():
            continue
        out.append(rel.as_posix())
    return out


def _default_planner_context_files(workspace: Path) -> List[str]:
    base = [
        "topic_brief.md",
        "notes/problem.md",
        "notes/method.md",
        "notes/predecessor_summary.md",
        "results/scoreboard.json",
        "notes/analysis_digest.md",
        "notes/foundation_survey.md",
        "notes/stagnation_report.md",
        "notes/exploration_log.md",
        "reviews/last_review_summary.md",
        "paper/manuscript.md",
        "notes/autopilot/verifier_last.md",
        "notes/autopilot/verifier_last.json",
    ]
    base.append("constraints.yaml")
    # Include workspace-level config files (e.g., pilot_afm.yaml).
    base += _pick_recent_paths(workspace, pattern="configs/*.yaml", limit=3)
    base += _pick_recent_paths(workspace, pattern="playbook/*.yaml", limit=5)
    # Include scripts so the Planner knows what code already exists.
    base += _pick_recent_paths(workspace, pattern="scripts/*.py", limit=10)
    base += _pick_recent_paths(workspace, pattern="evidence/*.json", limit=5)
    base += _pick_recent_paths(workspace, pattern="notes/autopilot/plan-*.json", limit=2)
    base += _pick_recent_paths(workspace, pattern="reviews/RESP-*.json", limit=1)
    # Drop duplicates while preserving order.
    seen: set[str] = set()
    out: List[str] = []
    for rel in base:
        if rel in seen:
            continue
        seen.add(rel)
        out.append(rel)
    return out


def _load_playbook_context(ledger: Ledger, project: Dict[str, Any]) -> str:
    domain = str(project.get("domain", "") or "").strip()
    rows = list(ledger.list_playbook_entries(limit=200))

    def _updated_at_ts(row: Dict[str, Any]) -> float:
        dt = _parse_iso_z(str(row.get("updated_at") or ""))
        return dt.timestamp() if dt is not None else float("-inf")

    if domain:
        domain_lower = domain.lower()

        def _topic_match(row: Dict[str, Any]) -> bool:
            return domain_lower in str(row.get("topic") or "").lower()

        rows.sort(
            key=lambda r: (
                not _topic_match(r),
                -_updated_at_ts(r),
                str(r.get("id") or ""),
            )
        )
        matched = [r for r in rows if _topic_match(r)]
        rows = matched if matched else rows
        rows = rows[:10]
    else:
        rows.sort(key=lambda r: (-_updated_at_ts(r), str(r.get("id") or "")))
        rows = rows[:5]
    if not rows:
        return ""

    lines: List[str] = ["Playbook lessons (from completed projects):"]
    for row in rows:
        entry_id = str(row.get("id") or "").strip()
        if not entry_id:
            continue

        rule: Dict[str, Any] = {}
        raw_rule = row.get("rule_json")
        if isinstance(raw_rule, str) and raw_rule.strip():
            try:
                parsed = json.loads(raw_rule)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                rule = parsed

        summary = ""
        for key in ("summary", "rule", "lesson", "name", "title"):
            value = rule.get(key)
            if isinstance(value, str) and value.strip():
                summary = value.strip()
                break
        if not summary:
            steps = rule.get("steps")
            if isinstance(steps, list):
                for step in steps:
                    if isinstance(step, str) and step.strip():
                        summary = step.strip()
                        break
                    if isinstance(step, dict):
                        for k in ("summary", "description", "rule", "name", "title", "step"):
                            value2 = step.get(k)
                            if isinstance(value2, str) and value2.strip():
                                summary = value2.strip()
                                break
                        if summary:
                            break
        if not summary:
            topic = str(row.get("topic") or "").strip()
            summary = f"topic={topic}" if topic else "No summary provided."

        summary = " ".join(summary.split())
        if len(summary) > 200:
            summary = summary[:197] + "..."
        lines.append(f"- {entry_id}: {summary}")

    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n\n"


def _compact_json_schema(workspace: Path, max_files: int = 3) -> str:
    """Return a compact, token-light schema summary of key JSON files."""

    def _describe(v: Any, depth: int = 0) -> str:  # noqa: ANN401
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return f"int (e.g. {v})"
        if isinstance(v, float):
            return f"float (e.g. {v:.4g})"
        if isinstance(v, str):
            short = v[:40] + "..." if len(v) > 40 else v
            return f'str ("{short}")'
        if v is None:
            return "null"
        if isinstance(v, list):
            return f"list[{len(v)} items]"
        if isinstance(v, dict) and depth < 1:
            inner = ", ".join(
                f"{k}: {_describe(sv, depth + 1)}" for k, sv in list(v.items())[:6]
            )
            if len(v) > 6:
                inner += ", ..."
            return "{" + inner + "}"
        if isinstance(v, dict):
            return f"dict[{len(v)} keys]"
        return type(v).__name__

    # Candidate files ordered by importance.
    candidates = [
        "results/scoreboard.json",
    ]
    # Add latest metrics.json from results/runs/
    runs_dir = workspace / "results" / "runs"
    if runs_dir.is_dir():
        metrics = sorted(runs_dir.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for m in metrics[:1]:
            candidates.append(str(m.relative_to(workspace)))

    lines: List[str] = []
    count = 0
    for rel in candidates:
        fp = workspace / rel
        if not fp.is_file():
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        lines.append(f"  {rel}:")
        for k, v in list(data.items())[:12]:
            lines.append(f"    {k}: {_describe(v)}")
        if len(data) > 12:
            lines.append(f"    ... ({len(data) - 12} more keys)")
        count += 1
        if count >= max_files:
            break

    if not lines:
        return ""
    header = (
        "\nWorkspace data schemas (auto-generated — field names and types from actual files):\n"
        "  Use these types when reading/writing these files.  In particular, bare numeric\n"
        "  fields (e.g., auroc_micro: float) are NOT wrapped in {value: ...} dicts.\n"
    )
    return header + "\n".join(lines) + "\n"


def _build_planner_prompt(
    *,
    ledger: Ledger,
    project_id: str,
    objective: str,
    iteration: int,
    max_actions: int = 6,
    max_context_chars_per_file: int = 60_000,
    context_files: Optional[List[str]] = None,
    goal_alignment: Optional[Dict[str, Any]] = None,
    provider: str = "openai",
) -> Tuple[str, Dict[str, Any], "Draft202012Validator"]:
    """Build the planner prompt, JSON schema, and validator.

    The prompt is provider-agnostic except for a few small differences:
    - Output mechanism: OpenAI uses function calling (submit_plan);
      Claude/Codex use plain JSON output conforming to the schema.
    - Web search tool name: OpenAI uses web_search; Claude uses WebSearch/WebFetch.

    Returns (base_prompt, schema_dict, validator).
    """
    project = get_project(ledger, project_id)
    workspace = Path(project["repo_path"]).resolve()

    schema = load_plan_schema(ledger.paths.root)
    validator = Draft202012Validator(schema)
    schema_txt = json.dumps(schema, ensure_ascii=False, indent=2)

    files = _load_context_files(
        workspace,
        rel_paths=list(context_files or _default_planner_context_files(workspace)),
        max_chars=max_context_chars_per_file,
    )
    playbook_block = _load_playbook_context(ledger, project)

    recent_tasks = ledger.list_tasks(project_id=project_id)[:10]
    recent_tasks_summary = []
    for t in recent_tasks:
        entry: Dict[str, Any] = {
            "id": t.get("id"),
            "type": t.get("type"),
            "status": t.get("status"),
            "updated_at": t.get("updated_at"),
        }
        # Attach the latest task-run summary (Codex memo) if available.
        try:
            runs = ledger._exec(
                "SELECT exit_code, last_message_path FROM task_runs WHERE task_id = ? ORDER BY started_at DESC LIMIT 1",
                (t["id"],),
            ).fetchall()
            if runs:
                entry["exit_code"] = runs[0]["exit_code"]
                lmp = runs[0].get("last_message_path")
                if lmp:
                    lmp_path = Path(lmp).resolve()
                    # Boundary check: only read from logs dir or workspace.
                    _logs_root = ledger.paths.logs_dir.resolve()
                    _ws_root = workspace.resolve()
                    _in_bounds = False
                    try:
                        lmp_path.relative_to(_logs_root)
                        _in_bounds = True
                    except ValueError:
                        try:
                            lmp_path.relative_to(_ws_root)
                            _in_bounds = True
                        except ValueError:
                            pass
                    if _in_bounds and lmp_path.is_file():
                        memo = lmp_path.read_text(encoding="utf-8", errors="replace")[:2000]
                        entry["summary"] = memo
        except Exception:
            pass
        recent_tasks_summary.append(entry)

    # Build a lightweight workspace file tree so the Planner knows what exists.
    _TREE_SKIP = {".git", "__pycache__", ".cache", "node_modules", ".orchestrator"}
    tree_lines: List[str] = []
    for dirpath_str, dirnames, filenames in os.walk(workspace):
        dirnames[:] = [d for d in dirnames if d not in _TREE_SKIP]
        dp = Path(dirpath_str)
        rel_dir = dp.relative_to(workspace)
        depth = len(rel_dir.parts)
        if depth > 3:
            dirnames.clear()
            continue
        indent = "  " * depth
        for fname in sorted(filenames):
            fp = dp / fname
            try:
                size = fp.stat().st_size
            except OSError:
                size = 0
            tree_lines.append(f"{indent}{rel_dir / fname}  ({size:,} bytes)")
    file_tree_txt = "\n".join(tree_lines[:200])  # cap at 200 entries

    # --- Separate review files from reference material ---
    _REVIEW_RELS = {"reviews/last_review_summary.md"}
    review_files = [(rel, txt) for rel, txt in files if rel in _REVIEW_RELS]
    reference_files = [(rel, txt) for rel, txt in files if rel not in _REVIEW_RELS]

    # Detect if the latest review is major/reject so we can highlight it.
    _review_is_major = False
    _review_section = ""
    for _rel, _txt in review_files:
        _low = _txt.lower()
        if "recommendation: major" in _low or "recommendation: reject" in _low:
            _review_is_major = True
        _review_section += _txt.rstrip() + "\n"

    # Also try to pick up a RESP-*.json that may have been loaded.
    for _rel, _txt in files:
        if _rel.startswith("reviews/RESP-") and _rel.endswith(".json"):
            if _rel not in _REVIEW_RELS:
                reference_files = [(_r, _t) for _r, _t in reference_files if _r != _rel]
                _review_section += f"\n--- {_rel} (raw review JSON) ---\n{_txt}\n"

    # Build the review-first prompt.
    review_block = ""
    if _review_is_major and _review_section.strip():
        review_block = (
            "============================================================\n"
            "UNRESOLVED REVIEW FINDINGS  (recommendation: major/reject)\n"
            "============================================================\n"
            "The research reviewer found major issues that MUST be addressed.\n"
            "Read the findings below carefully BEFORE looking at reference materials.\n\n"
            + _review_section.strip()
            + "\n\n"
            "YOUR OBLIGATIONS:\n"
            "- Your plan MUST address every major finding listed above.\n"
            "- In the 'notes' field, explain how each major finding is handled.\n"
            "- Do NOT plan new experiments on the current approach until the\n"
            "  methodology issues above are resolved or explicitly reframed.\n"
            "- If the findings require fundamental changes that cannot be done\n"
            "  in one iteration, set should_stop=true and explain what the\n"
            "  human PI needs to decide.\n"
            "============================================================\n\n"
        )
    elif _review_section.strip():
        review_block = (
            "--- Latest review (minor/accept — for your awareness) ---\n"
            + _review_section.strip()
            + "\n---\n\n"
        )

    # --- Goal alignment drift warning ---
    drift_block = ""
    if goal_alignment and goal_alignment.get("enabled") and not bool(goal_alignment.get("aligned", True)):
        drift_summary = str(goal_alignment.get("drift_summary") or "").strip()
        if drift_summary:
            drift_block = (
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "GOAL DRIFT DETECTED\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "The goal-alignment checker found that recent work has DRIFTED\n"
                "from the original research question.\n\n"
                f"Drift summary: {drift_summary}\n\n"
                "YOUR OBLIGATIONS:\n"
                "- Re-read notes/problem.md (the original research question).\n"
                "- Your plan MUST steer back toward the original question.\n"
                "- In the 'notes' field, explain how this plan corrects the drift.\n"
                "- If the drift is justified (deliberate pivot), say so explicitly\n"
                "  and explain why the new direction is better.\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n"
            )

    # --- PI decisions (binding human guidance) ---
    pi_block = ""
    pi_path = workspace / "notes" / "pi_decisions.md"
    if pi_path.exists():
        pi_text = pi_path.read_text(encoding="utf-8", errors="replace").strip()
        if pi_text:
            pi_block = (
                "============================================================\n"
                "PI DECISIONS (BINDING — you MUST follow these)\n"
                "============================================================\n"
                + pi_text
                + "\n============================================================\n\n"
            )

    # --- Stagnation report (metric plateau detected) ---
    stagnation_block = ""
    stag_path = workspace / "notes" / "stagnation_report.md"
    if stag_path.exists():
        stag_text = stag_path.read_text(encoding="utf-8", errors="replace").strip()
        if stag_text:
            stagnation_block = (
                "============================================================\n"
                "STAGNATION REPORT (metric plateau detected — auto-generated)\n"
                "============================================================\n"
                + stag_text
                + "\n\n"
                "YOUR OBLIGATIONS:\n"
                "- The primary metric has plateaued for multiple consecutive iterations.\n"
                "- If ci_overlap_current_vs_best is true OR best_sample_warning is present,\n"
                "  the recorded best is NOT a reliable target — do NOT optimize toward it.\n"
                "- You MUST choose one of: (a) revise the target to a realistic value,\n"
                "  (b) change your approach fundamentally, or (c) advance to the next\n"
                "  project stage. Repeating the same strategy is NOT acceptable.\n"
                "- Explain your choice in the plan's 'notes' field.\n"
                "============================================================\n\n"
            )

    # --- Error block (execution failures from previous iteration) ---
    error_block = ""
    err_path = workspace / "notes" / "last_errors.md"
    if err_path.exists():
        err_text = err_path.read_text(encoding="utf-8", errors="replace").strip()
        if err_text:
            error_block = (
                "============================================================\n"
                "EXECUTION ERRORS (from previous iteration — auto-generated)\n"
                "============================================================\n"
                + err_text
                + "\n\n"
                "YOUR OBLIGATIONS:\n"
                "- The previous iteration had task failures. The stderr output above\n"
                "  shows the ACTUAL error messages from the failed commands.\n"
                "- Read the error messages carefully and fix the SPECIFIC issue.\n"
                "- Do NOT retry the same command with identical parameters.\n"
                "- Common fixes: wrong CLI flags, missing files, wrong paths,\n"
                "  invalid arguments, missing dependencies.\n"
                "============================================================\n\n"
            )

    # --- Challenger block (interpretation concerns from previous iteration) ---
    challenger_block = ""
    chal_path = workspace / "notes" / "last_challenger.md"
    if chal_path.exists():
        chal_text = chal_path.read_text(encoding="utf-8", errors="replace").strip()
        if chal_text:
            challenger_block = (
                "============================================================\n"
                "INTERPRETATION CHALLENGER FLAGS (auto-generated)\n"
                "============================================================\n"
                + chal_text
                + "\n\n"
                "YOUR OBLIGATIONS:\n"
                "- An independent checker flagged concerns about the experimental\n"
                "  results or methodology. Review each flag above.\n"
                "- If mock/test data appears in the scoreboard, remove it.\n"
                "- If results are unreliable, do NOT build on them.\n"
                "============================================================\n\n"
            )

    # --- Verifier block (deterministic post-step checks from previous iteration) ---
    verifier_block = ""
    verifier_path = workspace / "notes" / "autopilot" / "verifier_last.json"
    if verifier_path.exists():
        try:
            verifier_raw = verifier_path.read_text(encoding="utf-8", errors="replace")
            verifier_obj = json.loads(verifier_raw) if verifier_raw.strip() else {}
            if not isinstance(verifier_obj, dict):
                verifier_obj = {}
        except (OSError, json.JSONDecodeError):
            verifier_obj = {}

        verdict = str(verifier_obj.get("verdict") or "").strip().lower()
        fail_items = verifier_obj.get("fail_items")
        if not isinstance(fail_items, list):
            fail_items = []
        fail_items = [str(x) for x in fail_items if str(x).strip()]

        needs_human_items = verifier_obj.get("needs_human_items")
        if not isinstance(needs_human_items, list):
            needs_human_items = []
        needs_human_items = [str(x) for x in needs_human_items if str(x).strip()]

        if verdict in {"fail", "needs_human"} or fail_items or needs_human_items:
            fail_lines = "\n".join(f"- {x}" for x in fail_items[:20]) or "- (none)"
            needs_lines = "\n".join(f"- {x}" for x in needs_human_items[:20]) or "- (none)"
            verifier_block = (
                "============================================================\n"
                "POST-STEP VERIFIER FINDINGS (deterministic checks)\n"
                "============================================================\n"
                f"- previous_verdict: {verdict or 'unknown'}\n"
                "- Fail items (MUST fix before new speculative work):\n"
                f"{fail_lines}\n"
                "- Needs human (track for submission-time verification):\n"
                f"{needs_lines}\n\n"
                "YOUR OBLIGATIONS:\n"
                "- Your plan MUST include concrete actions that resolve fail items above.\n"
                "- Do NOT ignore fail items; treat them as blockers for this iteration.\n"
                "- For needs_human items, preserve evidence and mark what requires PI/human review.\n"
                "============================================================\n\n"
            )

    # Determine whether this iteration needs explicit exploration of alternatives.
    need_alternatives = (
        iteration == 0
        or stagnation_block != ""
        or challenger_block != ""
        or error_block != ""
    )
    alternatives_block = ""
    if need_alternatives:
        alternatives_block = (
            "\nExploration requirement (ACTIVE for this iteration):\n"
            "- Before committing to your plan, consider 2-3 genuinely different approaches.\n"
            "- For each alternative, assess expected gain, risk, and why you prefer or reject it.\n"
            "- Populate the 'alternatives_considered' array in your output.\n"
            "- If notes/exploration_log.md lists previously rejected approaches, check whether\n"
            "  the rejection reasons still apply given current results. Conditions may have changed.\n"
            "- Do NOT generate token alternatives just to fill the field. Each alternative must be\n"
            "  a realistically viable approach you would consider if the chosen plan fails.\n\n"
        )

    base_prompt = (
        "You are the Planner for a local-first research orchestrator.\n"
        "Your job: produce the next plan as a JSON object "
        + ("by calling the function `submit_plan`.\n"
           "The function arguments MUST conform to the JSON Schema below.\n"
           if provider == "openai" else
           "conforming to the JSON Schema below.\n")
        + f"- project_id must be {project_id}\n"
        f"- iteration must be {iteration}\n"
        f"- actions length must be <= {max_actions}\n"
        "\nPlanning guidelines:\n"
        "- Review existing scripts/ before planning; prefer patching over recreating.\n"
        "- If the previous iteration was blocked, address the blocker explicitly.\n"
        "- If there are unresolved review findings (above), prioritize scientific findings\n"
        "  (category=method, analysis, novelty) over infrastructure findings (reproducibility,\n"
        "  writing, citations, cache hygiene). When primary_metric needs improvement, defer\n"
        "  documentation-only fixes to a later iteration.\n"
        "- If goal drift was detected (above), correct it or justify the new direction.\n"
        "- If PI decisions are present (above), they are BINDING: follow them unless physically impossible.\n"
        "- Monitor your own self_confidence across iterations. If it drops below 0.3,\n"
        "  explicitly consider whether a pivot is needed and explain in the 'notes' field.\n"
        "- Then check analysis_digest.md 'Next actions' for additional priorities.\n"
        "- Review notes/exploration_log.md for alternatives considered and paths not taken.\n"
        "  Avoid repeating failed or superseded approaches unless conditions have changed.\n"
        + alternatives_block
        + "- You SHOULD set next_stage when the current stage's objectives are met.\n"
        "  Do not leave next_stage as null indefinitely; advancing the stage helps\n"
        "  calibrate reviews and watchdog behavior.\n"
        "- Respect environment constraints in constraints.yaml; do not plan tasks requiring unavailable tools.\n"
        "- Action spec keys: shell_exec → {\"command\": \"...\"}, codex_exec → {\"prompt\": \"...\"}. Do NOT use 'script', 'cmd', or 'instructions'.\n"
        "- Task-type selection rule:\n"
        "  * shell_exec: ONLY for short commands (~15 lines max) — invoking an existing script,\n"
        "    file operations, pip install, simple data queries.  Never embed multi-page Python\n"
        "    scripts as inline heredocs.\n"
        "  * codex_exec: For ANY non-trivial code generation, data processing, metric computation,\n"
        "    or analysis.  Describe the GOAL + inputs/outputs and let Codex read the actual files\n"
        "    and implement.  Codex can inspect workspace files at runtime; you (the Planner) cannot.\n"
        "    This prevents schema mismatches — e.g., assuming a JSON field is a dict when it is a float.\n"
        "\nResearch-phase obligations (literature search via web search):\n"
        "- You have access to "
        + ("WebSearch and WebFetch tools" if provider == "claude_code_cli" else "web_search")
        + ". Use it PROACTIVELY for EVERY iteration — not only for literature surveys.\n"
        "  Verify citations, check for newer methods, and confirm baseline numbers via web search.\n"
        "- Researching prior work is YOUR responsibility as the Planner, not the executor's.\n"
        "  The executor (Codex) has NO web access — it can only use its training data, which\n"
        "  may be outdated or hallucinate citations.\n"
        "- Before generating the plan, search for:\n"
        "  (a) Key recent papers, reviews, and benchmarks on the topic.\n"
        "  (b) Existing databases, tools, and catalogs that already cover part of the objective.\n"
        "  (c) Known limitations, controversies, or open questions in the field.\n"
        "  (d) Specific author names, publication years, DOIs, and quantitative results.\n"
        "- Embed your findings (citations, key data, URLs) directly into codex_exec prompts\n"
        "  so the executor can write them into workspace documents with real references.\n"
        "  BAD: 'Write a literature-grounded survey of X.'\n"
        "  GOOD: 'Write a survey of X. Key references to include: Smith et al. (2024) showed Y\n"
        "         (DOI: ...), the Z database (url) covers A but not B, ...'\n"
        "- Include a summary of your web search findings in the plan's 'notes' field for audit.\n"
        "- Populate the optional 'literature_findings' array with individual citations/findings\n"
        "  (e.g., ['Smith et al. (2024) DOI:10.xxx — showed that ...', ...]).\n"
        "- Exception to execution-first policy: When the project is in a research/survey phase\n"
        "  (e.g., Phase 1 in problem.md is qualitative), the literature survey IS the real work.\n"
        "  Do not skip it to produce premature quantitative metrics.\n"
        "\nExecution-first policy (CRITICAL — anti-over-engineering):\n"
        "- If primary_metric.current.mean is null (no real experimental results yet), your HIGHEST PRIORITY\n"
        "  is to produce a REAL result, however small. A single smoke-test run producing a non-null metric is\n"
        "  worth more than any amount of infrastructure, validation, or test code.\n"
        "- Do NOT build comprehensive infrastructure before running your first experiment.\n"
        "  The correct order is: (1) minimal script that runs 1-3 targets end-to-end → (2) compute primary\n"
        "  metric from real outputs → (3) THEN iterate on infrastructure/validation/robustness.\n"
        "- If you have been running for 2+ iterations with primary_metric still null, you MUST include a\n"
        "  shell_exec action that directly executes the core computation (e.g., prediction, download,\n"
        "  simulation) in THIS plan. No more 'preparation' iterations.\n"
        "- Infrastructure-only plans (no shell_exec producing real data) are acceptable ONLY for iteration 0.\n"
        "\nPDCA requirements (research loop):\n"
        "- Treat `results/scoreboard.json` and `notes/analysis_digest.md` as required artifacts.\n"
        "- If your plan runs experiments/analysis or produces measurable results, include an action (usually last)\n"
        "  that updates:\n"
        "  - `results/scoreboard.json.primary_metric.current.mean` to a numeric value (and, if known, std/n_runs/ci_95).\n"
        "  - (Optional) `results/scoreboard.json.metrics` for additional metrics (e.g., RMSD, TM-score).\n"
        "  - `notes/analysis_digest.md` filling in Results + Next experiments with concrete numbers + links to artifacts.\n"
        "- Pivot triggers (configs/pivot_policy.yaml) rely on `primary_metric.current.mean`; if it is missing/null,\n"
        "  explicitly write why and what is needed to compute it.\n"
        "\nMetric revision & stage transition rules:\n"
        "- You MAY revise primary_metric (name, target, direction) or advance the project stage when\n"
        "  scientifically justified. Include 'metric_revision:' in the plan's notes field with the old\n"
        "  and new metric names/targets (e.g., 'metric_revision: usable_figure_rate→inter_rater_kappa').\n"
        "- Justification MUST be evidence-based (theoretical ceiling, data distribution, stage completion),\n"
        "  NOT mere inability to reach the current target. Metric revisions trigger hard review.\n"
        "- When proposing next_stage, also define the new stage's primary metric in an action that updates\n"
        "  scoreboard.json accordingly. Do not leave the next stage without a measurable primary metric.\n"
        "- If a stagnation report is present (above), you MUST act on it: do NOT repeat the same strategy.\n"
        + _compact_json_schema(workspace)
        + "\nJSON Schema:\n"
        + schema_txt
        + "\n\nProject context:\n"
        + json.dumps(
            {
                "project_id": project_id,
                "title": project.get("title"),
                "stage": project.get("stage"),
                "recent_tasks": recent_tasks_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nWorkspace file tree:\n"
        + file_tree_txt
        + "\n\n"
        + drift_block
        + pi_block
        + stagnation_block
        + error_block
        + challenger_block
        + verifier_block
        + review_block
        + playbook_block
        + "Reference materials (workspace files):\n"
        + "\n".join([f"\n--- FILE: {rel} ---\n{txt}\n" for rel, txt in reference_files])
        + "\nObjective:\n"
        + objective.strip()
        + "\n"
    )

    return base_prompt, schema, validator


def generate_plan_openai(
    *,
    ledger: Ledger,
    project_id: str,
    objective: str,
    model: str,
    iteration: int,
    max_actions: int = 6,
    max_context_chars_per_file: int = 60_000,
    context_files: Optional[List[str]] = None,
    max_retries: int = 2,
    background: bool = True,
    config: Optional[Dict[str, Any]] = None,
    goal_alignment: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base_prompt, schema, validator = _build_planner_prompt(
        ledger=ledger,
        project_id=project_id,
        objective=objective,
        iteration=iteration,
        max_actions=max_actions,
        max_context_chars_per_file=max_context_chars_per_file,
        context_files=context_files,
        goal_alignment=goal_alignment,
        provider="openai",
    )

    tool_name = "submit_plan"
    tool_desc = "Submit a single AutopilotPlan JSON object."
    tool_variant = {
        "tools": [{"type": "function", "name": tool_name, "description": tool_desc, "parameters": schema}],
        "tool_choice": {"type": "function", "name": tool_name},
    }

    client = OpenAIClient.from_env()
    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        prompt = base_prompt
        if last_err:
            prompt = (
                base_prompt
                + "\n\nYour previous output did not match the schema.\n"
                + "Fix and re-submit by calling submit_plan again.\n\nSchema error:\n"
                + last_err
            )

        base_payload: Dict[str, Any] = {"model": model, "input": prompt, "background": bool(background)}
        reasoning = config.get("planner_reasoning_effort") if isinstance(config, dict) else None
        if reasoning:
            base_payload["reasoning"] = {"effort": str(reasoning)}
        # Enable web search so Planner can look up recent papers / APIs / datasets.
        web_search_tool = {"type": "web_search_preview"}
        combined_tools = [web_search_tool] + list(tool_variant.get("tools") or [])
        combined_variant = {**tool_variant, "tools": combined_tools}
        payload_variants = [{**base_payload, **combined_variant}]
        resp = run_response_to_completion_with_fallback(client=client, payload_variants=payload_variants)

        usage = resp.get("usage") if isinstance(resp, dict) else None
        usage_obj: Dict[str, Any] = usage if isinstance(usage, dict) else {}
        total_tokens = usage_obj.get("total_tokens")
        try:
            total_tokens_i = int(total_tokens) if total_tokens is not None else None
        except (TypeError, ValueError):
            total_tokens_i = None

        planner_meta = {
            "response_id": resp.get("id") if isinstance(resp, dict) else None,
            "status": resp.get("status") if isinstance(resp, dict) else None,
            "usage": usage_obj,
            "usage_total_tokens": total_tokens_i,
        }

        plan = extract_json_object_from_response(resp, function_name=tool_name)
        plan = _repair_plan_actions_for_runtime(plan)
        errors = sorted(validator.iter_errors(plan), key=lambda e: e.json_path)
        if not errors:
            semantic_errors = _validate_plan_action_semantics(plan)
            if semantic_errors:
                if attempt >= max_retries:
                    raise ValueError(f"Planner action validation failed: {semantic_errors[0]}")
                last_err = semantic_errors[0]
                continue
            return plan, planner_meta
        if attempt >= max_retries:
            raise ValueError(f"Planner output schema validation failed: {errors[0].message}")
        last_err = errors[0].message

    raise SystemExit("generate_plan_openai failed unexpectedly (exhausted retries).")


def generate_plan_claude(
    *,
    ledger: Ledger,
    project_id: str,
    objective: str,
    model: str,
    iteration: int,
    max_actions: int = 6,
    max_context_chars_per_file: int = 60_000,
    context_files: Optional[List[str]] = None,
    max_retries: int = 2,
    config: Optional[Dict[str, Any]] = None,
    goal_alignment: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate a plan using Claude Code CLI (subscription-based, no API cost)."""
    from resorch.providers.claude_code_cli import (
        ClaudeCodeCliConfig,
        ClaudeCodeCliError,
        extract_structured_output,
        run_claude_code_print_json,
    )

    base_prompt, schema, validator = _build_planner_prompt(
        ledger=ledger,
        project_id=project_id,
        objective=objective,
        iteration=iteration,
        max_actions=max_actions,
        max_context_chars_per_file=max_context_chars_per_file,
        context_files=context_files,
        goal_alignment=goal_alignment,
        provider="claude_code_cli",
    )

    project = get_project(ledger, project_id)
    workspace = Path(project["repo_path"]).resolve()

    timeout = 1800
    if isinstance(config, dict):
        try:
            timeout = int(config.get("planner_timeout", 1800))
        except (TypeError, ValueError):
            timeout = 1800

    cli_model = str(model or "opus").strip()

    cfg = ClaudeCodeCliConfig(
        model=cli_model,
        timeout_sec=timeout,
        tools="Read,WebSearch,WebFetch",
        allowed_tools="Read,WebSearch,WebFetch",
        no_session_persistence=True,
    )

    system_prompt = (
        "You are the Planner for a local-first research orchestrator. "
        "You MUST respond with a single JSON object conforming to the provided schema. "
        "Do NOT include any text outside the JSON object."
    )

    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        prompt = base_prompt
        if last_err:
            prompt = (
                base_prompt
                + "\n\nYour previous output did not match the schema.\n"
                + "Fix and re-submit a corrected JSON plan.\n\nSchema error:\n"
                + last_err
            )

        try:
            cli_json = run_claude_code_print_json(
                prompt=prompt,
                system_prompt=system_prompt,
                json_schema=schema,
                workspace_dir=workspace,
                config=cfg,
            )
        except ClaudeCodeCliError as e:
            if attempt >= max_retries:
                raise ValueError(f"Claude planner failed after {max_retries + 1} attempts: {e}") from e
            last_err = str(e)
            continue

        plan = extract_structured_output(cli_json)
        plan = _repair_plan_actions_for_runtime(plan)

        planner_meta: Dict[str, Any] = {
            "provider": "claude_code_cli",
            "model": cli_model,
            "attempt": attempt,
            "usage": {},
            "usage_total_tokens": None,
        }

        errors = sorted(validator.iter_errors(plan), key=lambda e: e.json_path)
        if not errors:
            semantic_errors = _validate_plan_action_semantics(plan)
            if semantic_errors:
                if attempt >= max_retries:
                    raise ValueError(f"Claude planner action validation failed: {semantic_errors[0]}")
                last_err = semantic_errors[0]
                continue
            return plan, planner_meta
        if attempt >= max_retries:
            raise ValueError(f"Claude planner output schema validation failed: {errors[0].message}")
        last_err = errors[0].message

    raise SystemExit("generate_plan_claude failed unexpectedly (exhausted retries).")


def generate_plan_codex(
    *,
    ledger: Ledger,
    project_id: str,
    objective: str,
    model: str,
    iteration: int,
    max_actions: int = 6,
    max_context_chars_per_file: int = 60_000,
    context_files: Optional[List[str]] = None,
    max_retries: int = 2,
    config: Optional[Dict[str, Any]] = None,
    goal_alignment: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate a plan using Codex CLI."""
    from resorch.providers.codex_cli import (
        CodexCliConfig,
        CodexCliError,
        extract_structured_output,
        run_codex_exec_print_json,
    )

    base_prompt, schema, validator = _build_planner_prompt(
        ledger=ledger,
        project_id=project_id,
        objective=objective,
        iteration=iteration,
        max_actions=max_actions,
        max_context_chars_per_file=max_context_chars_per_file,
        context_files=context_files,
        goal_alignment=goal_alignment,
        provider="codex_cli",
    )

    project = get_project(ledger, project_id)
    workspace = Path(project["repo_path"]).resolve()

    timeout = 1800
    if isinstance(config, dict):
        try:
            timeout = int(config.get("planner_timeout", 1800))
        except (TypeError, ValueError):
            timeout = 1800

    cli_model = str(model or "").strip()
    if cli_model.lower() in {"haiku", "sonnet", "opus"}:
        # Existing config templates default to Claude aliases. For Codex provider,
        # omit incompatible aliases and let Codex pick its default model.
        cli_model = ""

    system_prompt = (
        "You are the Planner for a local-first research orchestrator. "
        "You MUST respond with a single JSON object conforming to the provided schema. "
        "Do NOT include any text outside the JSON object."
    )

    overrides: List[str] = []
    reasoning = config.get("planner_reasoning_effort") if isinstance(config, dict) else None
    if reasoning:
        overrides.append(f'model_reasoning_effort="{reasoning}"')

    cfg = CodexCliConfig(
        model=cli_model or None,
        timeout_sec=timeout,
        sandbox="danger-full-access",
        config_overrides=overrides,
    )

    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        prompt = base_prompt
        if last_err:
            prompt = (
                base_prompt
                + "\n\nYour previous output did not match the schema.\n"
                + "Fix and re-submit a corrected JSON plan.\n\nSchema error:\n"
                + last_err
            )
        prompt = f"{system_prompt}\n\n{prompt}"

        try:
            cli_json = run_codex_exec_print_json(
                prompt=prompt,
                json_schema=schema,
                workspace_dir=workspace,
                config=cfg,
            )
        except CodexCliError as e:
            if attempt >= max_retries:
                raise ValueError(f"Codex planner failed after {max_retries + 1} attempts: {e}") from e
            last_err = str(e)
            continue

        plan = extract_structured_output(cli_json)
        plan = _repair_plan_actions_for_runtime(plan)

        usage_obj = cli_json.get("usage") if isinstance(cli_json, dict) else None
        if not isinstance(usage_obj, dict):
            usage_obj = {}
        total_tokens = usage_obj.get("total_tokens")
        try:
            total_tokens_i = int(total_tokens) if total_tokens is not None else None
        except (TypeError, ValueError):
            total_tokens_i = None

        planner_meta: Dict[str, Any] = {
            "provider": "codex_cli",
            "model": cli_model or None,
            "attempt": attempt,
            "usage": usage_obj,
            "usage_total_tokens": total_tokens_i,
        }

        errors = sorted(validator.iter_errors(plan), key=lambda e: e.json_path)
        if not errors:
            semantic_errors = _validate_plan_action_semantics(plan)
            if semantic_errors:
                if attempt >= max_retries:
                    raise ValueError(f"Codex planner action validation failed: {semantic_errors[0]}")
                last_err = semantic_errors[0]
                continue
            return plan, planner_meta
        if attempt >= max_retries:
            raise ValueError(f"Codex planner output schema validation failed: {errors[0].message}")
        last_err = errors[0].message

    raise SystemExit("generate_plan_codex failed unexpectedly (exhausted retries).")
