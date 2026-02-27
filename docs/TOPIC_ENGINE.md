# Topic Engine — Research Topic Selection Module

The Topic Engine replaces ad-hoc topic selection with a structured pipeline:
**generate candidates → discard early → validate feasibility → commit**.

Goals:
1. Maximize research velocity — reject weak topics in 48 hours rather than wasting 2 weeks.
2. Make topic selection explainable — the ledger records why each topic was chosen or rejected.

---

## Pipeline (Stages and Gates)

### Stage 0: Constraints Capture

**Input (human writes once)**
- Research domain / target venues (candidates are fine)
- Available compute (GPU, cloud, budget, deadline)
- Data constraints (public only / institutional data / no PII)
- Risk constraints (safety, ethics, licensing)
- Desired contribution type (method / system / dataset / analysis / survey)

**Output**: `constraints.yaml` in the workspace

**CLI**: `orchestrator constraints init --project <id>`

**Gate**: If constraints are incomplete, generate minimal clarifying questions.

---

### Stage 1: Idea Generation

**Goal**: Generate many candidates so that bad ones can be discarded early.

**Recommended approach** (three lenses to reduce bias):
1. **Literature-gap driven** — holes in existing work
2. **Capability-driven** — what is realistically executable
3. **Asset-driven** — leverage existing code / data / benchmarks

**Output**: 20–60 `IdeaRecord` entries (JSONL)

**Rule**: One candidate = one claim. Not "seems promising" but "what would the paper demonstrate?" in one sentence.

**CLI**: `orchestrator idea import --project <id> --input ideas.jsonl`

**Automation**: `orchestrator topic engine --project <id> --rounds 3` runs a multi-round loop (generate → deduplicate → score → activate top-k).

---

### Stage 2: Novelty & Positioning Check

**Tools**: OpenAI `web_search` / Deep Research (store results as Evidence)

**What to do**:
- For each idea, find the 5 most relevant prior works.
- Write a one-line differentiation (if you can't articulate the difference, discard the idea).

**Output**: `idea.novelty.evidence[]` (URL, retrieval date, excerpt, summary, relevance)

**Gate**: "No articulated difference" or "identical work exists" → auto-reject.

**CLI**: `orchestrator evidence add`, `orchestrator evidence list`

**Note**: Automated novelty search is not yet implemented. Currently manual: run Deep Research jobs, then `evidence add`.

---

### Stage 3: Feasibility Smoke Test

**Goal**: Prevent "it didn't work" after 2 weeks of effort.

**Minimum tasks**:
- Reproduce baseline (or approximate implementation, small config is fine)
- Run data acquisition → preprocessing → 1 epoch end-to-end
- Confirm that the primary metric is measurable

**Output**: `runs/smoke/<idea_id>/` with logs, commit hash, seed, environment

**Gate**: If no measurable number within 48 hours → reject by default (exception: survey / analysis topics).

**CLI**: `orchestrator smoke ingest --project <id> --result result.json`

---

### Stage 4: Value Check (Scoring)

**Evaluation axes** (0–5 each):
- Does it fit the target venue's scope?
- Are baselines and comparisons clear?
- Even if it fails, is there a publishable "academic lesson" (negative results)?

**Scoring**: Rubric-weighted sum using `rubrics/idea_score_rubric.yaml`.

Default weights:
| Axis | Weight |
|------|--------|
| Novelty | 0.25 |
| Feasibility | 0.25 |
| Impact | 0.20 |
| Clarity | 0.15 |
| Reusability | 0.10 |
| Risk (penalty) | 0.05 |

Feasibility is weighted heavily because "experiment won't run" is fatal.

**Providers**:
- `--provider arithmetic` (default) — uses pre-filled scores in JSONL
- `--provider claude` — LLM evaluates each axis automatically

**CLI**: `orchestrator idea score --project <id> --rubric rubrics/idea_score_rubric.yaml`

**Gate**: Weak evidence links → send to "additional experiment" or reject.

---

### Stage 5: Commit

**Output** (this defines the "topic"):
- `topic_brief.md` (1–2 pages)
  - 3 title candidates
  - One-sentence claim
  - Expected key figure
  - Evaluation plan (baselines / ablations / metrics)
  - Risk / ethics / licensing
  - 2-week work plan

**CLI**:
```bash
orchestrator topic brief --project <id> --idea <idea_id>     # preview
orchestrator topic commit --project <id> --idea <idea_id>    # brief + set selected
```

---

## IdeaRecord Required Fields

See `schemas/idea.schema.json` for the full schema.

- `id`: Unique identifier
- `title`: Working title
- `one_sentence_claim`: One-sentence assertion (stated as fact)
- `contribution_type`: method / system / dataset / analysis / survey
- `target_venues`: Candidate venues
- `novelty_statement`: One-line differentiation from existing work
- `evaluation_plan`: Metrics, baselines, datasets, ablations
- `feasibility`: Resource estimates (GPU hours, etc.)
- `risks`: Ethics, safety, licensing, reproducibility
- `evidence`: Prior work references (required)

---

## Review Integration

**Fixed gates** (external reviewer is invoked at these points):
- After Stage 2: Novelty / positioning red-team
- After Stage 3: Feasibility / experiment design red-team
- Before Stage 5: PC (program committee) role — enumerate rejection reasons

**Red-team review**: `orchestrator review redteam --project <id>` generates a review packet using `prompts/reviewer_redteam.md` as rubric.

---

## Cross-Project Learning

At project completion, update the Playbook and inject lessons into the next topic selection cycle:

- Venues that worked + reviewer feedback → playbook entries
- Common rejection reasons (weak novelty, missing comparisons, bad setup) → playbook entries
- Reusable baseline implementations, data preprocessing → inherited via `project create-successor`

**CLI**: `orchestrator playbook extract --project <id>`, `orchestrator playbook put --file entry.yaml`

---

## Add-on Templates

- Stage gate rules: `configs/stage_transitions.yaml`
- Smoke test schema: `schemas/smoke_test_result.schema.json`
- Idea deduplication: `resorch/idea_dedupe.py` (FTS5-based, CLI: `orchestrator idea dedupe`)
