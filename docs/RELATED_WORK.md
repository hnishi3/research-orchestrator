# Related Work — AI Research Automation Systems

Design lessons from prior systems, selected for practical relevance to the research-orchestrator architecture.

> Goal: Know "which idea to borrow from where" at a glance. Prioritizes implementation utility over completeness.

---

## 1) End-to-End Research Agents (Idea → Experiment → Paper)

### The AI Scientist (Sakana AI, 2024) — arXiv:2408.06292
- **What it does**: Given a direction + seed codebase, runs the full pipeline: idea generation → literature search → experiment iteration → figure generation → paper writing → automated review.
- **Design takeaways**:
  - **Connect research phases via artifacts** (logs, figures, notes) rather than chat context.
  - **Feed automated reviews back into the next iteration** (use acceptance-style rubrics).

### The AI Scientist-v2 (Yamada et al., 2025) — arXiv:2504.08066
- **What's new**: Reduces template dependency, generalizes across domains. Introduces an Experiment Manager + progressive agentic tree search. VLM feedback for figure refinement.
- **Design takeaways**:
  - **Tree-structured exploration** beats linear iteration — evaluate branches and prune.
  - **Dedicated "experiment progress manager"** prevents the LLM from losing focus.

### Kosmos (Mitchener et al., 2025) — arXiv:2511.02824
- **What's notable**: 12-hour continuous runs, parallel data analysis and literature search. Uses a "world model" (structured memory) for consistency. Report claims are traced back to code and primary sources.
- **Design takeaways**:
  - **Structured memory / world model** is effective for long-horizon consistency.
  - **Enforced claim → evidence links** make paper writing less fragile.

### SafeScientist (Zhu et al., 2025) — EMNLP / NeurIPS
- **What's notable**: Addresses science-specific risks (hazardous materials, biosafety) with input monitoring, cooperative monitoring, tool monitoring, and ethics review. Evaluated on SciSafetyBench.
- **Design takeaways**:
  - **Safety design is a publishable contribution** for research automation (especially for web fetch and tool execution).

### "Why LLMs Aren't Scientists Yet" (Trehan & Chopra, 2026) — arXiv:2601.03315
- **What's useful**: Catalogs failure modes of autonomous research (implementation drift, memory degradation, premature success claims, weak experiment design) with real examples.
- **Design takeaways**:
  - **Turn failure modes into gate conditions** (e.g., success claims require matching reproduction logs).

---

## 2) Software Engineering Multi-Agent Systems

### MetaGPT (FoundationAgents, 2023–) — "Software Company as Multi-Agent System"
- **What it does**: Assigns roles (PM, Architect, Engineer) with SOPs (standard operating procedures) to flow requirements → design → implementation.
- **Design takeaways**:
  - **Explicit roles + artifact schemas (SOPs)** stabilize multi-model / multi-project operations.

### AutoGen (Wu et al., ICLR 2024 submission)
- **What it does**: Composes "conversable agents" and programs their conversation patterns.
- **Design takeaways**:
  - **Codifying inter-agent conversation design** improves operational reproducibility.

### SWE-agent (Yang et al., 2024) — arXiv:2405.15793
- **Key finding**: Agent-Computer Interface (ACI) design — how the agent explores, edits, and executes — matters more than prompt engineering alone.
- **Design takeaways**:
  - **Work environment interface design is the bottleneck**, not just prompts.

---

## 3) Tool Use and Long-Horizon Interaction

### ReAct (Yao et al., 2022) — arXiv:2210.03629
- **What it does**: Interleaves reasoning (thought) and action (tool call) to reduce hallucination and update plans.
- **Design takeaways**:
  - **Think → Act → Observe → Update** as the minimal design unit.

### Reflexion (Shinn et al., 2023) — arXiv:2303.11366
- **What it does**: Reflects on failures in natural language and applies lessons to the next trial (verbal RL).
- **Design takeaways**:
  - **Playbook updates at project completion** are a natural fit for the Reflexion pattern.

### AskToAct (Zhang et al., 2025) — arXiv:2503.01940
- **Key insight**: Self-correcting clarification improves tool use for ambiguous user requests.
- **Design takeaways**:
  - **Deep Research API doesn't ask follow-ups**, so build a clarification loop externally.

### ToolSandbox (Lu et al., 2024/2025) — arXiv:2408.04682
- **Key contribution**: Evaluation framework for stateful, multi-turn tool use (covers realistic edge cases).
- **Design takeaways**:
  - **Include stateful traps in your evaluation** to strengthen publishability.

---

## 4) Benchmarks and Evaluation

- **PaperBench** (Starace et al., 2025) — arXiv:2504.01848: Evaluates "paper reproduction" by decomposing it into rubric items.
- **GAIA** (Mialon et al., 2023 / ICLR 2024) — arXiv:2311.12983: Evaluates real-world assistant ability (web + tools + reasoning).

---

## 5) Checklist for This Orchestrator

- [ ] Manage experiments as **tree-structured exploration** (keep best node)
- [ ] **Structured memory**: enforce claim → evidence storage
- [ ] **Block premature success claims** until reproduction logs are complete
- [ ] **Safety design for web fetch + tool execution** (allowlist + monitoring + reviewer gate)
- [ ] Use **PaperBench / GAIA / ToolSandbox** as reference for evaluation task design
