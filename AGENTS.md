# AGENTS.md (for Codex)

## Mission
Implement and maintain the research-orchestrator system in this repository.
The system must reduce manual copy/paste between ChatGPT (planning) and Codex (execution) by making **artifacts + ledger** the source of truth.

## Research loops (PDCA)
- **Micro (minutes–1h)**: code/config → run → update `results/scoreboard.json` + `notes/analysis_digest.md` → review → fix
- **Meso (hours–days; HPC wait)**: submit external job → wait/poll → ingest results → next step (agent may stop and resume via cron)
- **Macro (across projects)**: accumulate successes/failures/pivots in Idea Bank + Playbook to get better over time

## Invariants
- **Artifact-first**: important state must live in workspace files and the ledger; chat logs are not source-of-truth.
- **Claim → Evidence**: any strong conclusion should link to evidence (URL/log/figure/repro command).
- **Digest discipline**: after each iteration, update `results/scoreboard.json` and `notes/analysis_digest.md` (or record why it cannot be updated yet).

## Working agreements
- Make the smallest safe change that satisfies the task.
- Prefer readability and reproducibility over cleverness.
- Never hard-code secrets. Use `.env` and document required env vars in README.
- After changing code, run unit tests and (if available) a minimal end-to-end demo command.
- Write or update docs for any user-visible behavior.

## Repository conventions
- All persistent state goes to `./.orchestrator/` by default (SQLite DB, logs).
- Project workspaces live under `./workspaces/<project_id>/` unless configured otherwise.
- Generated artifacts must be registered in the ledger and placed under the workspace tree.

## Safety
- Default to read-only execution and escalate permissions only when necessary.
- Avoid destructive shell commands and large refactors unless explicitly requested.

## Deliverables checklist (for each milestone)
- [ ] Code implemented
- [ ] Tests added/updated
- [ ] README updated (usage + examples)
- [ ] Demo run succeeds

## Roadmap (backlog)
- **M1 Idea Bank v0.1**: lineage edges + spawn/park/revive/graph CLI (implemented)
- **M2 HPC backend v0.1**: `constraints.yaml` compute backend (local/slurm), submit/poll, stop-and-resume behavior (future)
- **M3 Multi-metric scoreboard**: extend scoreboard schema + pivot policy for tradeoffs (future)
- **M4 Data mounts + DB inventory**: external data references + DB version tracking (no auto-download by default) (future)
- **M5 Pipeline integration**: Nextflow/Snakemake as black box + summary→scoreboard convention (future)
- **M6 Visual inspection gate**: require human confirmation for non-scalar validity (e.g., 3D structures) (implemented)
- **M7 Cohort mode**: run multiple “student agents” with dedupe/locking + lab-meeting artifact (future)
