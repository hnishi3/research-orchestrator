# Research-Orchestrator (resorch)

A local-first, multi-LLM research agent system that coordinates:

- **Planner** (OpenAI GPT / Claude Code CLI / Codex CLI) — generates experiment plans,
- **Executor** (Codex CLI) — runs code and experiments in a sandbox,
- **Reviewer** (OpenAI / Claude Code CLI / Codex CLI / Anthropic API) — reviews results, triggers pivots,

to iterate through PDCA (Plan-Do-Check-Act) cycles across multiple research projects.

## Prerequisites

| Tool | Purpose | Required? |
|------|---------|-----------|
| Python 3.10+ | Runtime | Yes |
| [Codex CLI](https://github.com/openai/codex) (`codex`) | Task executor | Yes |
| `OPENAI_API_KEY` | OpenAI Planner / Reviewer | Yes (if using OpenAI planner) |
| [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude`) | Planner / Reviewer (subscription-based, no API cost) | Optional |
| `ANTHROPIC_API_KEY` | Anthropic API Reviewer | Optional |

The entry point is the `./orchestrator` script at the repo root. No `pip install` step is needed; ensure the dependencies are available in your Python environment.

## Quickstart

```bash
# 1. Initialize state
./orchestrator init

# 2. Create a project
./orchestrator project new --title "My Research"
./orchestrator project list

# 3. Run the agent loop (multi-iteration PDCA)
./orchestrator agent run \
  --project my-research \
  --objective "Implement baseline and evaluate on test set." \
  --max-steps 5
```

## Workspace Structure

`./orchestrator project new` creates a workspace under `workspaces/<project_id>/`:

```
workspaces/<project_id>/
├── notes/
│   ├── problem.md            # Research question, hypothesis, metrics
│   ├── method.md             # Approach, baselines, ablations
│   ├── analysis_digest.md    # Rolling human-readable log
│   ├── exploration_log.md    # Rejected approaches & alternatives (rolling)
│   └── autopilot/            # Plan JSONs + verifier outputs
├── ideas/                    # Idea bank (JSONL)
├── results/
│   ├── scoreboard.json       # Machine-readable metrics (drives pivot triggers)
│   ├── fig/                  # Figures (triggers visual inspection if enabled)
│   └── raw/
├── runs/
│   ├── agent/                # Agent loop iteration outputs
│   └── smoke/                # Feasibility smoke test results
├── reviews/
│   └── last_review_summary.md
├── configs/                  # Workspace-level config overrides (see Configuration)
├── src/
├── experiments/
├── data/
├── paper/
│   └── manuscript.md
├── evidence/
├── claims/
└── jobs/
```

Key files:
- **`notes/problem.md`** is critical — the goal alignment checker reads it before every iteration.
- **`results/scoreboard.json`** drives pivot detection. The `primary_metric.current.mean` value is compared across runs.

## Core Workflow: Agent Loop

The **agent loop** (`agent run`) is the primary workflow. Each iteration:

1. **Goal alignment** — verifies the objective hasn't drifted from `notes/problem.md` (Claude Sonnet, fail-open)
2. **Planner** — reads scoreboard, digest, exploration log, and review history; generates a plan with up to N actions. On key iterations (first, stagnation, errors), requires `alternatives_considered` for exploration diversity
3. **Pre-exec review gate** — before executing analysis scripts, validates scientific correctness via a 5-item checklist (fail-open). Supports `claude_code_cli` (default, subscription) and `codex_cli` providers. Configurable in `review_policy.yaml` → `pre_exec_review`
4. **Executor** — Codex CLI runs each action (`codex_exec` or `shell_exec`). Complex shell commands are auto-promoted to `codex_exec` (>40 lines with embedded Python, or workspace script invocations)
5. **Post-exec review** — evaluates results, creates `review_fix` tasks if needed. On hard gates, runs dual review (primary + escalation, worst-of policy)
6. **Interpretation challenger** — sanity-checks updated scoreboard metrics (Claude Sonnet, fail-open)
7. **Post-step verifier** — rule-based verification checklist for scientific soundness; outputs to `notes/autopilot/verifier_last.{md,json}`
8. **Pivot check** — if primary metric hasn't improved, triggers a review or stops

Note: **Pre-exec code review** (Claude Opus) exists but is **disabled by default** — the pre-exec review gate and auto-promotion make it largely redundant.

```bash
# Basic
./orchestrator agent run --project <id> --objective "<text>" --max-steps 10

# With custom config (e.g., Claude planner)
./orchestrator agent run --project <id> --objective "<text>" \
  --max-steps 10 --config workspaces/<id>/configs/agent_loop.yaml

# Skip Planner on first iteration (reuse previous plan)
./orchestrator agent run --project <id> --objective "<text>" \
  --max-steps 3 --reuse-last-plan

# Dry run (plan only, no execution)
./orchestrator agent run --project <id> --objective "<text>" --max-steps 1 --dry-run
```

### Autopilot (single iteration)

Run one plan-execute cycle without the full loop:

```bash
./orchestrator autopilot run --project <id> \
  --objective "Implement the next smallest feature."
```

### Cohort mode (multiple agents)

Run N independent "student agents" and generate a lab-meeting summary:

```bash
./orchestrator cohort run --project <id> --n 3 \
  --objective "Propose 3 distinct approaches to improve the primary metric." \
  --max-steps 2
```

## Configuration

Three YAML config files control behavior. Config resolution uses auto-discovery:

1. **Explicit `--config` path** (if provided on CLI)
2. **Workspace-level**: `workspaces/<id>/configs/<file>.yaml`
3. **Global fallback**: `configs/<file>.yaml`
4. **Built-in defaults**

### `configs/agent_loop.yaml` — Planner & Loop

```yaml
planner:
  provider: openai            # "openai" | "claude_code_cli" | "codex_cli"
  model: gpt-5.2-pro          # Provider-specific model name
  background: true            # OpenAI background mode (async)
  max_actions: 6
  reasoning_effort: high
  timeout: 1800               # Planner call timeout (seconds)

review:
  max_fix_tasks_per_review: 10
  questions:
    - Are the conclusions supported by the current results and evidence?
    - What are the top 1-3 highest-information next experiments, and why?
    # ...
```

**Default**: Claude Code CLI (`provider: claude_code_cli`, `model: opus`) — subscription-based, no per-token API cost.

**Provider options**:
- `claude_code_cli` — Claude Code CLI (subscription). Models: `haiku`, `sonnet`, `opus`.
- `codex_cli` — Codex CLI (subscription). Models: `gpt-5.2`, `gpt-5.3-codex`, etc. Supports `reasoning_effort` (`low`/`medium`/`high`/`xhigh`).
- `openai` — OpenAI Responses API (per-token cost, requires `OPENAI_API_KEY`). Also supports `reasoning_effort` and `background` (async).

**Codex-only example** (for users who only have Codex CLI):
```yaml
planner:
  provider: codex_cli
  model: gpt-5.2
  reasoning_effort: xhigh
  max_actions: 6
```

Override per-project via `--config` flag or by placing a copy in `workspaces/<id>/configs/agent_loop.yaml`.

### `configs/review_policy.yaml` — Multi-layer Review

| Layer | When | Default Provider | Purpose |
|-------|------|------------------|---------|
| `goal_alignment` | Before Planner | Claude Sonnet (CLI) | Drift detection vs `problem.md` (fail-open) |
| `pre_exec_review` | Before script execution | Claude Sonnet (CLI) | 5-item science checklist (fail-open) |
| `code_review_gate` | Before Executor | Claude Opus (CLI) | Code review of planned actions (**disabled by default**) |
| `post_exec` (primary) | After Executor | Claude Opus (CLI) | Result evaluation |
| `post_exec` (escalation) | Hard gates | Claude Opus (CLI) | Dual review (worst-of policy) |
| `interpretation_challenger` | Scoreboard update | Claude Sonnet (CLI) | Sanity-check metrics (fail-open) |
| `post_step_verifier` | After each step | Rule-based | Verification checklist (no LLM cost) |
| `lightweight_retry` | Minor findings | (no model) | Skip Planner, re-run with fixes |

Key sections:

```yaml
# --- Reviewer models (post-exec primary / escalation / reformatter) ---
# Default: Claude Code CLI (subscription, no API cost).
reviewers:
  primary:
    provider: claude_code_cli     # "claude_code_cli" | "openai" | "codex_cli"
    model: opus
  escalation:
    provider: claude_code_cli     # Used for hard gates (dual review)
    model: opus
  reformatter:
    provider: claude_code_cli
    model: sonnet

# For OpenAI reviewers (per-token cost):
#   primary:  { provider: openai, model: gpt-5.2-pro, reasoning_effort: medium }
#   escalation: { provider: openai, model: gpt-5.2-pro, reasoning_effort: high }

# --- Hard / soft triggers ---
hard_gates:
  on_stage_transition: true
  on_claim_create: true
  on_paper_artifact_change: { paths: ["paper/**"] }
  on_dependency_change: { paths: ["requirements*.txt", "pyproject.toml"] }
  on_external_fetch: true

soft_triggers:
  fail_streak: { same_task: 2, any: 3, force_stop_after: 6 }
  stall_minutes: 30
  cost_guard: { window_minutes: 15, max_total_tokens: 200000 }

requires_visual_inspection: false   # stop when new figures appear

# --- Pre-exec review (science checklist, injected into codex_exec prompts) ---
pre_exec_review:
  enabled: true
  model: sonnet                     # Claude model (subscription, no API cost)
  # provider: claude_code_cli       # or "codex_cli" for GPT-based review
```

**Codex-only example** (all reviewers and guardrails via Codex CLI):
```yaml
reviewers:
  primary:  { provider: codex_cli, model: gpt-5.2, reasoning_effort: xhigh }
  escalation: { provider: codex_cli, model: gpt-5.2, reasoning_effort: xhigh }
  reformatter: { provider: codex_cli, model: gpt-5.2, reasoning_effort: medium }
goal_alignment:
  provider: codex_cli
  model: gpt-5.2
  reasoning_effort: medium
interpretation_challenger:
  provider: codex_cli
  model: gpt-5.2
  reasoning_effort: medium
pre_exec_review:
  provider: codex_cli
  model: gpt-5.2
```

**Workspace override**: place `review_policy.yaml` in `workspaces/<id>/configs/` — auto-loaded, takes precedence over `configs/review_policy.yaml`.

### `configs/pivot_policy.yaml` — Stall Detection

```yaml
no_improvement:
  metric_path: primary_metric.current.mean
  direction: maximize
  window_runs: 3
  use_ci_overlap: true          # overlapping 95% CIs = "no improvement"
  force_stop_after: 3           # halt after N consecutive triggers

metric_watchdog:
  null_metric_force_stop_after: 3
  unchanged_metric_force_stop_after: 3
```

### `configs/stage_transitions.yaml` — Stage Gates

Defines conditions for project stage transitions (e.g., intake → exploration → experiment → analysis → writing).

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | Required when using OpenAI provider (planner, reviewer, or jobs) | (none — not needed for default config) |
| `ANTHROPIC_API_KEY` | Required when using `provider: anthropic` | (none — not needed for `claude_code_cli`) |
| `OPENAI_PLANNER_MODEL` | Override planner model (env var fallback) | config value takes precedence |
| `PLANNER_PROVIDER` | Override planner provider (env var fallback) | config value takes precedence |
| `RESORCH_ROOT` | Override repo root path | (auto-detect via `AGENTS.md` + `resorch/`) |
| `RESORCH_WEBHOOK_TOKEN` | Webhook authentication | (optional) |
| `OPENAI_WEBHOOK_SECRET` | Standard Webhooks signature verification | (optional) |

## CLI Reference

### Projects

```bash
./orchestrator project new --title "Demo Project"
./orchestrator project list
./orchestrator project open <project_id>
./orchestrator project set-stage <project_id> <stage>

# Create a successor project (inherits data/src/configs via symlinks)
./orchestrator project create-successor --predecessor <id> \
  [--id <new_id>] [--title <title>] [--inherit data src configs]
```

The successor gets a `notes/predecessor_summary.md` with the predecessor's scoreboard and analysis digest, automatically loaded into the Planner's context.

### Tasks

```bash
./orchestrator task create --project <id> --type codex_exec \
  --spec-json '{"prompt":"Say hello.","sandbox":"read-only"}'
./orchestrator task run <task_id>
./orchestrator task status <task_id>
```

Task types: `codex_exec` (Codex sandbox), `shell_exec` (local shell), `review_fix` (auto-created from reviews).

Notes:
- `codex_exec` accepts `spec.prompt` or `spec.prompt_file`
- Default sandbox: `read-only`; `review_fix` defaults to `workspace-write`
- The runner appends `schemas/task_result.schema.json` for structured output
- The runner injects `-c model_reasoning_effort="high"` unless overridden in `spec.config_overrides`

### Reviews

```bash
./orchestrator review request --project <id> --stage intake --target notes/problem.md
./orchestrator review redteam --project <id>
./orchestrator review ingest --result /path/to/review_result.json
```

`review ingest` auto-creates `review_fix` tasks and writes `reviews/last_review_summary.md`.

### Artifacts

```bash
./orchestrator artifact list --project <id>
./orchestrator artifact get <artifact_id>
./orchestrator artifact put --project <id> --kind <kind> --path <file_path>
```

Artifacts are automatically registered by most commands (idea import, smoke ingest, topic commit, etc.). Use `artifact put` to manually register files.

### Jobs (Background / Polling)

```bash
# OpenAI response
./orchestrator job create --project <id> --provider openai --kind response \
  --spec-json '{"payload":{"model":"gpt-5.2","input":"Summarize.","background":true}}'
./orchestrator job run <job_id>
./orchestrator job poll <job_id>

# Deep research (o3-deep-research)
./orchestrator job create --project <id> --provider openai --kind deep_research \
  --spec-json '{"query":"Key papers about X.","artifact_path":"notes/deep_research.md"}'

# Claude Code CLI review (subscription, no API cost)
./orchestrator job create --project <id> --provider claude_code_cli --kind review \
  --spec-json '{"stage":"intake","targets":["notes/problem.md"],"questions":["Clear?"]}'

# Anthropic API review
./orchestrator job create --project <id> --provider anthropic --kind review \
  --spec-json '{"stage":"intake","targets":["notes/problem.md"]}'

# OpenAI review
./orchestrator job create --project <id> --provider openai --kind review \
  --spec-json '{"stage":"intake","targets":["notes/problem.md"],"model":"gpt-5.2-pro"}'

# Local/Slurm compute
./orchestrator job create --project <id> --provider compute --kind compute \
  --spec-json '{"cd":".","command":"python train.py"}'
```

List and inspect jobs:

```bash
./orchestrator job list --project <id>
./orchestrator job get <job_id>
```

Configure compute backend in `constraints.yaml` (`compute.backend: local|slurm`).
When external compute jobs are pending, the agent loop pauses (`should_stop=true`).

### Topic Engine & Idea Bank

The Topic Engine manages research topic selection through a 6-stage pipeline. Each stage has CLI commands for the manual steps; automated LLM-driven stages are planned but not yet implemented.

#### Stage Pipeline

| Stage | Name | CLI Support | Automation |
|-------|------|-------------|------------|
| 0 | Constraints Capture | `constraints init` | Manual (fill YAML template) |
| 1 | Idea Generation | `idea import` | **Not implemented** — create JSONL externally, then import. Prompt template at `prompts/idea_generation.md` |
| 2 | Novelty Check | `evidence add/list` | **Not implemented** — run Deep Research jobs manually, add evidence per idea |
| 3 | Smoke Test | `smoke ingest/list` | **Not implemented** — run experiments manually, ingest results |
| 4 | Value & Scoring | `idea score`, `idea dedupe` | Rubric-weighted scoring (`--provider arithmetic`, default) or LLM evaluation (`--provider claude`) |
| 5 | Commit | `topic commit`, `topic brief` | Generates structured `topic_brief.md` from idea record |

Stage gates are evaluated via `stage check` (see `configs/stage_transitions.yaml`):

```bash
./orchestrator stage check --project <id>
```

#### Idea CRUD & Import

Ideas follow the `IdeaRecord` schema (`schemas/idea.schema.json`). Required fields: `id`, `title`, `one_sentence_claim`, `contribution_type`, `target_venues`, `novelty_statement`, `evaluation_plan` (datasets/metrics/baselines/ablations), `feasibility`, `risks`, `evidence`.

```bash
# Import ideas from JSONL (one IdeaRecord per line)
./orchestrator idea import --project <id> --input ideas/ideas.jsonl

# List / get / update status
./orchestrator idea list --project <id>
./orchestrator idea get <idea_id>
./orchestrator idea set-status <idea_id> --status selected
# Status values: candidate | active | rejected | smoke_passed | selected | parked | done
```

#### Scoring & Deduplication

Scoring applies rubric weights (`rubrics/idea_score_rubric.yaml`) to per-axis scores (novelty, feasibility, impact, clarity, reusability, risk_penalty; 0-5 each). Two providers:

- `--provider arithmetic` (default) — uses pre-filled scores in the JSONL
- `--provider claude` — LLM evaluates each axis and fills scores automatically

```bash
./orchestrator idea score --project <id> \
  --rubric rubrics/idea_score_rubric.yaml --output ideas/ranked.jsonl

# LLM-powered scoring
./orchestrator idea score --project <id> --provider claude --output ideas/ranked.jsonl
```

Deduplication uses lexical + Jaccard similarity (threshold 0.9) to merge near-duplicates:

```bash
./orchestrator idea dedupe --project <id> \
  --input ideas/ideas.jsonl --output ideas/deduped.jsonl
```

#### Idea Bank (Lifecycle Management)

**Park** — shelve an idea with reason and unblock conditions:

```bash
./orchestrator idea park <id> --reason "Need more compute" --unblock "GPU available"
```

**Revive** — unpark into a new project (creates child idea + lineage edge):

```bash
./orchestrator idea revive <id> --new-project <project> --title "Next Phase"
```

**Spawn** — create a child idea via mutation operator:

```bash
# Operators: narrow, broaden, reframe, baseline_add, metric_swap
./orchestrator idea spawn --parent <id> --operator baseline_add --baseline "NewBaseline"
./orchestrator idea spawn --parent <id> --operator reframe --reframe "Alternative framing"
```

**Link** — connect two ideas with a relation:

```bash
./orchestrator idea link --src <id1> --dst <id2> --relation reframe --reason "Alt approach"
```

**Graph** — visualize idea lineage:

```bash
./orchestrator idea graph --project <id> --format dot   # or json
```

#### Smoke Tests

Run experiments externally, then ingest structured results (`schemas/smoke_test_result.schema.json`). On `pass` verdict, the idea status auto-updates to `smoke_passed`:

```bash
./orchestrator smoke ingest --project <id> --result tmp_smoke.json
./orchestrator smoke list --project <id>
```

#### Topic Brief & Commit

Generate a structured Markdown brief from an idea record (title ideas, claim, evaluation plan, feasibility, risks, 2-week plan):

```bash
./orchestrator topic brief --project <id> --idea <idea_id>          # preview
./orchestrator topic commit --project <id> --idea <idea_id>         # brief + set selected
```

#### Topic Engine Search Loop

`topic engine` runs an automated multi-round loop: generate ideas (LLM), deduplicate, score, and activate the top-k:

```bash
./orchestrator topic engine --project <id> --rounds 3 --top-k 5 --dry-run
```

#### Not Yet Implemented

The following features are described in the design (`docs/TOPIC_ENGINE.md`) but not yet automated:

- **`idea generate`** — standalone LLM-driven bulk idea generation (separate from `topic engine`). Currently requires external creation + `idea import`.
- **Automated novelty search** — per-idea web search / Deep Research to find related work. Currently manual: run jobs, then `evidence add`.
- **Auto stage transitions** — `stage check` evaluates gates but does not auto-apply pass/reject in the agent loop.

#### Commit & Launch

Commit an idea and create a ready-to-run project in one step:

```bash
./orchestrator idea commit-and-launch --project <id> --idea <idea_id>
```

This runs `topic commit` + `project new --idea-id`, wiring the topic brief into the new workspace.

### Benchmarks

Run external benchmark suites (PaperBench, AIRS, ReplicatorBench) via adapter:

```bash
./orchestrator bench list --suite paperbench --external-path /path/to/PaperBench
./orchestrator bench run --suite paperbench --task <task_id> \
  --external-path /path/to/PaperBench --max-steps 20
./orchestrator bench run --suite paperbench --task <task_id> --dry-run
```

### Portfolio

Run one prioritization cycle across all active projects:

```bash
./orchestrator portfolio cycle
```

### Verification

```bash
./orchestrator verify checklist --project <id>      # PI verification checklist
./orchestrator verify consistency --project <id>     # Manuscript consistency check
./orchestrator verify submission --project <id>      # Submission release gate
```

### Evidence & Claims

```bash
./orchestrator evidence add --project <id> --kind paper \
  --title "Title" --url "https://..." --summary "Why it matters"
./orchestrator evidence list --project <id>
./orchestrator evidence get <evidence_id>

./orchestrator claim new --project <id> --statement "We claim X" --evidence <eid>
```

Evidence kinds: `paper`, `blog`, `doc`, `dataset`, `benchmark`, `repo`, `other`.

### Constraints & DB

```bash
./orchestrator constraints init --project <id>     # create constraints.yaml template
./orchestrator db ensure --project <id>             # check DB inventory (no auto-download)
```

### Visual Inspection Gate

When `requires_visual_inspection: true`:

```bash
./orchestrator visual status --project <id>
./orchestrator visual approve --project <id> --note "Looks reasonable"
```

### Pipeline Integration

Ingest pipeline outputs into PDCA artifacts:

```bash
./orchestrator summary ingest --project <id> --path results/summary.json
```

### Webhooks

```bash
./orchestrator webhook serve --host 127.0.0.1 --port 8787
```

Receives events at `POST /openai`. Supports `RESORCH_WEBHOOK_TOKEN` (header auth) and `OPENAI_WEBHOOK_SECRET` (Standard Webhooks signatures).

### Playbook

```bash
./orchestrator playbook put --file playbook/playbook_entry_template.yaml
./orchestrator playbook list
./orchestrator playbook get <entry_id>

# Auto-extract a playbook entry from project artifacts (analysis digest, scoreboard, reviews)
./orchestrator playbook extract --project <id> [--mode compact|full]
```

Playbook entries are YAML records of patterns/anti-patterns (trigger, steps, pitfalls, evidence).

### MCP Server (experimental)

```bash
python mcp_server/server.py --repo-root .
```

See `docs/CODEX_MCP_SETUP.md` and `.codex/config.toml.example`.

### Tests

```bash
pytest -q   # 450+ tests
```

## Key Files

| Path | Description |
|------|-------------|
| `SPEC.md` | Full specification and implementation roadmap |
| `AGENTS.md` | Instructions for Codex in this repo |
| `configs/agent_loop.yaml` | Planner & review loop config |
| `configs/review_policy.yaml` | Review triggers, guardrails, multi-layer providers |
| `configs/pivot_policy.yaml` | Stall detection & pivot triggers |
| `configs/stage_transitions.yaml` | Stage gate rules |
| `configs/default.rules` | Codex rules file |
| `schemas/` | JSON Schemas (task_result, scoreboard, idea, smoke_test, etc.) |
| `prompts/` | System prompts (reviewer_code, reviewer_research, challenger, redteam) |
| `review/` | Review packet template & result schema |
| `rubrics/` | Stage gate rubric examples |
| `playbook/` | Structured learning templates |

## Documentation

| Doc | Description |
|-----|-------------|
| `docs/TOPIC_ENGINE.md` | Topic selection process & stage gates |
| `docs/MODEL_ORCHESTRATION.md` | Multi-model coordination design |
| `docs/ERROR_HANDLING.md` | Error handling patterns |
| `docs/PROVENANCE_AND_CITATIONS.md` | Evidence & citation tracking |
| `docs/API_WEB_SEARCH.md` | Web search via OpenAI API |
| `docs/CODEX_MCP_SETUP.md` | MCP server setup for Codex |
| `docs/RELATED_WORK.md` | Prior work & design takeaways |
| `docs/PAPER_IDEA.md` | Paper outline & evaluation plan |
