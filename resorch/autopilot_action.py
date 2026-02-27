from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def _normalize_action_spec(
    *,
    task_type: str,
    spec: Dict[str, Any],
    action_title: Optional[str] = None,
) -> Dict[str, Any]:
    out = dict(spec)
    # Planner may use "instructions" but tasks.py expects "prompt".
    if "instructions" in out and "prompt" not in out:
        out["prompt"] = out.pop("instructions")
    # Planner may use "commands" (list) or "cmd" but shell_exec expects "command".
    if "commands" in out and "command" not in out:
        cmds = out.pop("commands")
        if isinstance(cmds, list):
            out["command"] = " && ".join(str(c) for c in cmds)
            out.setdefault("shell", True)
        else:
            out["command"] = cmds
    if "cmd" in out and "command" not in out:
        out["command"] = out.pop("cmd")
    # Planner may use "script" for shell_exec instead of "command".
    if "script" in out and "command" not in out:
        out["command"] = out.pop("script")
        out.setdefault("shell", True)
    # Common Planner synonym drift for codex_exec prompt.
    if task_type == "codex_exec" and "prompt" not in out:
        for alias in ("instruction", "request", "task", "goal", "objective", "message", "content"):
            val = out.get(alias)
            if isinstance(val, str) and val.strip():
                out["prompt"] = val.strip()
                break
    # Last-resort repair for codex_exec: derive a minimal prompt from action title.
    if task_type == "codex_exec":
        prompt = out.get("prompt")
        prompt_file = out.get("prompt_file")
        has_prompt = isinstance(prompt, str) and bool(prompt.strip())
        has_prompt_file = isinstance(prompt_file, str) and bool(prompt_file.strip())
        if not has_prompt and not has_prompt_file:
            title = str(action_title or "").strip()
            if title:
                out["prompt"] = (
                    f"{title}\n\n"
                    "Use workspace files as source of truth. Implement the task with minimal changes,"
                    " run relevant checks, and return concise results."
                )
    # Ensure multi-line or compound commands use shell=True.
    if task_type == "shell_exec" and "command" in out:
        cmd_val = out["command"]
        if isinstance(cmd_val, str) and ("\n" in cmd_val or "&&" in cmd_val or ";" in cmd_val):
            out.setdefault("shell", True)
    # Planner-generated codex_exec tasks need write access; default in tasks.py is read-only.
    # Use danger-full-access because workspace-write triggers Landlock which may not be
    # available on all kernels (causes silent failures with exit code 0).
    if task_type == "codex_exec" and "sandbox" not in out:
        out["sandbox"] = "danger-full-access"
    return out


def _validate_normalized_action_spec(*, task_type: str, spec: Dict[str, Any]) -> Optional[str]:
    """Return validation error message for a normalized action spec, else None."""
    if task_type == "shell_exec":
        cmd = spec.get("command")
        if isinstance(cmd, str) and cmd.strip():
            return None
        if isinstance(cmd, list) and len(cmd) > 0:
            return None
        return "shell_exec requires non-empty spec.command"

    if task_type == "codex_exec":
        prompt = spec.get("prompt")
        prompt_file = spec.get("prompt_file")
        if isinstance(prompt, str) and prompt.strip():
            return None
        if isinstance(prompt_file, str) and prompt_file.strip():
            return None
        return "codex_exec requires non-empty spec.prompt or spec.prompt_file"

    return None


def _validate_action_spec(
    *,
    task_type: str,
    spec: Any,
    action_title: Optional[str] = None,
) -> Optional[str]:
    """Validate a raw action spec with compatibility normalization."""
    if not isinstance(spec, dict):
        return "spec must be an object"
    normalized = _normalize_action_spec(task_type=task_type, spec=spec, action_title=action_title)
    return _validate_normalized_action_spec(task_type=task_type, spec=normalized)


def _validate_plan_action_semantics(plan: Dict[str, Any]) -> List[str]:
    """Validate action spec semantics not expressible in the JSON Schema."""
    actions = plan.get("actions")
    if not isinstance(actions, list):
        return []

    errors: List[str] = []
    for i, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        task_type = str(action.get("task_type") or "").strip()
        err = _validate_action_spec(
            task_type=task_type,
            spec=action.get("spec"),
            action_title=str(action.get("title") or ""),
        )
        if err:
            errors.append(f"actions[{i}]: {err}")
    return errors


def _repair_planned_action_for_runtime(action: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort repair for planner action drift before validation/execution."""
    out = dict(action)
    task_type = str(out.get("task_type") or "").strip()
    title = str(out.get("title") or "").strip()
    spec_raw = out.get("spec")
    spec = spec_raw if isinstance(spec_raw, dict) else {}
    spec_norm = _normalize_action_spec(task_type=task_type, spec=spec, action_title=title)

    # If shell_exec command is missing, salvage by converting to codex_exec.
    if task_type == "shell_exec":
        cmd = spec_norm.get("command")
        cmd_ok = (isinstance(cmd, str) and bool(cmd.strip())) or (isinstance(cmd, list) and len(cmd) > 0)
        if not cmd_ok:
            prompt_src = ""
            for key in ("prompt", "instructions", "instruction", "request", "task", "goal", "objective", "message", "content"):
                v = spec.get(key)
                if isinstance(v, str) and v.strip():
                    prompt_src = v.strip()
                    break
            if not prompt_src:
                for key in ("script", "cmd", "command"):
                    v = spec.get(key)
                    if isinstance(v, str) and v.strip():
                        prompt_src = v.strip()
                        break
            if not prompt_src:
                prompt_src = title
            if prompt_src:
                repaired_spec = _normalize_action_spec(
                    task_type="codex_exec",
                    spec={"prompt": prompt_src},
                    action_title=title,
                )
                out["task_type"] = "codex_exec"
                out["spec"] = repaired_spec
                log.info(
                    "Auto-repaired invalid shell_exec -> codex_exec (missing command, title=%r)",
                    title[:120],
                )
                return out

    out["spec"] = spec_norm
    return out


def _repair_plan_actions_for_runtime(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Apply best-effort repairs to all actions in a planner output."""
    actions = plan.get("actions")
    if not isinstance(actions, list):
        return plan
    out = dict(plan)
    out["actions"] = [
        _repair_planned_action_for_runtime(a) if isinstance(a, dict) else a
        for a in actions
    ]
    return out


# ---------------------------------------------------------------------------
# Auto-promote shell_exec -> codex_exec
# ---------------------------------------------------------------------------
_EMBEDDED_PYTHON_RE = re.compile(
    r"<<['\"]?(?:PY|PYTHON|EOF)"   # heredoc markers
    r"|python3?\s+-c\s"            # python -c '...'
    r"|python3?\s+-\s*<<"          # python - <<EOF
    r"|python3?\s+<<",             # python <<EOF
    re.IGNORECASE,
)
_SHELL_PROMOTE_LINE_THRESHOLD = 40

# Condition B: command invokes a workspace Python script whose CLI interface
# the Planner cannot inspect.  Matches  python scripts/foo.py ,
# python3 tools/bar.py , etc. but NOT  python -m pytest  or  python -c '...' .
_WORKSPACE_SCRIPT_RE = re.compile(
    r"python3?\s+(?!-)[a-zA-Z0-9_./]+\.py\b",
)


def _should_promote_to_codex(action: Dict[str, Any]) -> bool:
    """Return True if a shell_exec action should be auto-promoted to codex_exec.

    Promotion triggers (OR):
      A) command > ``_SHELL_PROMOTE_LINE_THRESHOLD`` lines **and** contains
         embedded Python (heredoc or ``python -c``).
      B) command invokes a workspace Python script (``python <path>.py``).
         The Planner cannot read the script so it guesses CLI args -- Codex can.
    """
    if str(action.get("task_type") or "").strip() != "shell_exec":
        return False
    spec = action.get("spec") or {}
    if not isinstance(spec, dict):
        return False
    # Check all possible command field names (before normalization).
    cmd = spec.get("command") or spec.get("script") or spec.get("cmd") or ""
    cmds = spec.get("commands")
    if cmds and not cmd:
        if isinstance(cmds, list):
            cmd = "\n".join(str(c) for c in cmds)
        else:
            cmd = str(cmds)
    if not isinstance(cmd, str) or not cmd.strip():
        return False

    # Condition B: workspace script invocation (any length).
    if _WORKSPACE_SCRIPT_RE.search(cmd):
        return True

    # Condition A: long inline Python.
    if len(cmd.strip().splitlines()) <= _SHELL_PROMOTE_LINE_THRESHOLD:
        return False
    return bool(_EMBEDDED_PYTHON_RE.search(cmd))


# ---------------------------------------------------------------------------
# Pre-exec review gate (injected into codex_exec prompts)
# ---------------------------------------------------------------------------

PRE_EXEC_REVIEW_INSTRUCTIONS = """\

--- PRE-EXEC REVIEW GATE ---
Before running any Python script that computes metrics or produces results,
call:
  claude --print --no-session-persistence --model {model} \
    --tools Read --allowedTools Read \
    -p "Read notes/problem.md and notes/method.md for context, then read \
    the script at [SCRIPT_PATH]. Check scientific correctness ONLY \
    (not code bugs — you handle those yourself): \
    (1) Label direction: is higher/lower correct for the metric? \
    (2) Train/test leakage: does training data leak into evaluation? \
    (3) Baseline fairness: are baselines given equal treatment? \
    (4) Misleadingly positive results: any cherry-picking or selective reporting? \
    (5) Statistical test choice: is the test appropriate for the data? \
    Your ENTIRE response must be a single line starting with PASS or FAIL. \
    Format: PASS: <reason> or FAIL: <reason>. No preamble, no markdown, no headers."
Replace [SCRIPT_PATH] with the actual script path. If FAIL, fix before running.
Applies to analysis/metrics scripts only — skip for helpers, installs, downloads.
--- END PRE-EXEC REVIEW GATE ---
"""

PRE_EXEC_REVIEW_INSTRUCTIONS_CODEX = """\

--- PRE-EXEC REVIEW GATE ---
Before running any Python script that computes metrics or produces results,
call:
  codex exec --sandbox read-only --skip-git-repo-check --cd . {model_flag}"Read notes/problem.md and notes/method.md for context, then read \
  the script at [SCRIPT_PATH]. Check scientific correctness ONLY \
  (not code bugs — you handle those yourself): \
  (1) Label direction: is higher/lower correct for the metric? \
  (2) Train/test leakage: does training data leak into evaluation? \
  (3) Baseline fairness: are baselines given equal treatment? \
  (4) Misleadingly positive results: any cherry-picking or selective reporting? \
  (5) Statistical test choice: is the test appropriate for the data? \
  Your ENTIRE response must be a single line starting with PASS or FAIL. \
  Format: PASS: <reason> or FAIL: <reason>. No preamble, no markdown, no headers."
Replace [SCRIPT_PATH] with the actual script path. If FAIL, fix before running.
Applies to analysis/metrics scripts only — skip for helpers, installs, downloads.
--- END PRE-EXEC REVIEW GATE ---
"""


def _render_pre_exec_review_instructions(*, provider: str, model: str) -> str:
    provider_norm = str(provider or "").strip().lower()
    if provider_norm == "codex_cli":
        model_flag = f"--model {model} " if model else ""
        return PRE_EXEC_REVIEW_INSTRUCTIONS_CODEX.format(model_flag=model_flag)
    return PRE_EXEC_REVIEW_INSTRUCTIONS.format(model=model or "haiku")


def _maybe_inject_pre_exec_review(
    *,
    spec: Dict[str, Any],
    task_type: str,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Append pre-exec review gate instructions to codex_exec prompts.

    Returns a (possibly modified) copy of *spec*.  No-op if:
      - task_type is not codex_exec
      - pre_exec_review is not enabled in policy
      - spec has no prompt field
    """
    if task_type != "codex_exec":
        return spec
    review_cfg = policy.get("pre_exec_review") or policy.get("science_review") or {}
    if not isinstance(review_cfg, dict) or not review_cfg.get("enabled", False):
        return spec
    prompt = spec.get("prompt")
    if not prompt or not isinstance(prompt, str):
        return spec
    provider = str(review_cfg.get("provider") or "claude_code_cli").strip().lower()
    if provider not in {"claude_code_cli", "codex_cli"}:
        provider = "claude_code_cli"
    model_default = "gpt-5.3-codex" if provider == "codex_cli" else "haiku"
    model = str(review_cfg.get("model") or model_default).strip()
    if provider == "codex_cli" and model.lower() in {"haiku", "sonnet", "opus"}:
        # Handle old Claude-style aliases gracefully in Codex mode.
        model = "gpt-5.3-codex"
    out = dict(spec)
    out["prompt"] = prompt.rstrip() + "\n\n" + _render_pre_exec_review_instructions(provider=provider, model=model)
    return out


def _promote_shell_to_codex(action: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a shell_exec action to codex_exec.

    The original command is included in the Codex prompt as a *reference
    implementation* with an explicit warning to verify schemas.
    """
    title = action.get("title") or action.get("description") or "Execute the task"
    spec = action.get("spec") or {}
    cmd = spec.get("command") or spec.get("script") or spec.get("cmd") or ""
    cmds = spec.get("commands")
    if cmds and not cmd:
        if isinstance(cmds, list):
            cmd = "\n".join(str(c) for c in cmds)
        else:
            cmd = str(cmds)

    new_spec = {
        "prompt": (
            f"{title}\n\n"
            "The Planner provided the following reference script.  "
            "DO NOT execute it blindly — verify all file paths, JSON schemas, "
            "and field types against the actual workspace files before running.  "
            "In particular, check whether numeric fields (e.g. auroc_micro) are "
            "bare floats vs dicts with a 'value' key.\n\n"
            f"```bash\n{cmd}\n```"
        ),
        "sandbox": "danger-full-access",
    }
    out = dict(action)
    out["task_type"] = "codex_exec"
    out["spec"] = new_spec
    out["_auto_promoted_from_shell"] = True
    log.info(
        "Auto-promoted shell_exec → codex_exec (%d lines, title=%r)",
        len(cmd.strip().splitlines()),
        title[:80],
    )
    return out
