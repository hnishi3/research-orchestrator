# Changelog

## 2026-02-08: PDCA Cycle Fixes (9 issues)

Based on observations from the `structbio-opus-free` agent run (steps 000-012, ~20 hours).
Full investigation: see `memory/pdca_findings_v2.md`.

### CRITICAL

- **Interpretation Challenger now fires on codex_exec** (`autopilot.py`)
  - Previously only triggered on `shell_exec` success, but auto-promotion converts
    all tasks to `codex_exec` before execution. Challenger was completely dead.
  - Fix: Check for both `shell_exec` and `codex_exec` in the trigger condition.

### HIGH

- **Escalation (dual review) now fires with same provider** (`agent_loop.py`)
  - Guard `esc_provider != provider` blocked escalation when primary and escalation
    used the same provider (the default config). Escalation never ran.
  - Fix: Removed the same-provider guard. `dual_on_hard: true` is sufficient intent signal.

- **Hard review no longer fires every iteration** (`autopilot.py`)
  - Two always-true triggers: `paper_files_changed` (all untracked files matched)
    and `stage_transition_ready` (transitions perpetually "ready" when stage stuck).
  - Fix A: Added `_ensure_git_baseline()` — auto-commits workspace state at iteration
    start so `git diff` only reflects current iteration's changes.
  - Fix B: Added `stage_transition_requested` parameter — hard gate only fires when
    Planner explicitly requests a stage change via `next_stage`.

### MEDIUM

- **step.json now includes stage_update data** (`agent_loop.py`)
  - Stage update happened after the final step.json write, so `stage_update` field
    was missing from the persisted file.
  - Fix: Added a third write after stage update when `step_record["stage_update"]` exists.

- **Watchdog exempts writing/validation stages** (`agent_loop.py`)
  - Unchanged metric watchdog force-stopped agent during legitimate writing/validation
    phases where metrics are stable by design.
  - Fix: Added `exempt_stages` config (default: writing, validation, revision, final).
    Configurable via `pivot_policy.yaml` → `metric_watchdog.exempt_stages`.

- **Review severity now stage-aware** (`prompts/reviewer_research.md`)
  - 20/26 intake reviews rated "major" because rubric didn't account for stage.
  - Fix: Added stage-aware calibration guidance (intake=lenient, analysis=standard,
    writing=strict).

### LOW

- **Planner WebSearch instruction strengthened** (`autopilot.py`)
  - Changed from "use PROACTIVELY when objective involves literature survey" to
    "use PROACTIVELY for EVERY iteration".

- **Low confidence pivot guidance** (`autopilot.py`)
  - Added instruction: when `self_confidence < 0.3`, explicitly consider pivot.

- **Stage progression guidance** (`autopilot.py`)
  - Added instruction: "You SHOULD set next_stage when current stage objectives are met."

### Tests

- Added `tests/test_pdca_fixes.py` with 7 new tests covering Issues 1-5.
