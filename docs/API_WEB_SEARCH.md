# API Web Search — Reproducing "Strong Search" via OpenAI API

## TL;DR

- The OpenAI **Responses API** provides a `web_search` tool for search + cited summaries.
- **Deep Research** (`o3-deep-research`, `o4-mini-deep-research`) is also available via API for long-form research reports.
- The API version of Deep Research **does not ask clarifying questions** (unlike the ChatGPT UI), so the orchestrator must handle ambiguity resolution externally.

---

## 1) `web_search` Tool (Responses API)

### Features

- `web_search` feeds search results into the model's context.
- Responses include **citations**.
- Add `include: ["web_search_call.action.sources"]` to get source URLs in the response.
- Use `filters.allowed_domains` to restrict search to specific domains.
- Control live vs. cached results with `external_web_access`.

### Minimal Example (JavaScript)

```js
import OpenAI from "openai";
const client = new OpenAI();

const resp = await client.responses.create({
  model: "gpt-5.2",
  include: ["web_search_call.action.sources"],
  tools: [{ type: "web_search" }],
  input: "List 3 key papers on AI-driven research automation (2025-2026) with one-sentence summaries and citations."
});

console.log(resp.output_text);
```

### Domain-Restricted Search

```js
const resp = await client.responses.create({
  model: "gpt-5.2",
  include: ["web_search_call.action.sources"],
  tools: [{
    type: "web_search",
    filters: { allowed_domains: ["developers.openai.com", "platform.openai.com"] }
  }],
  input: [{ type: "text", text: "Summarize the latest Codex changelog entries about web_search settings, with citations." }]
});
```

### Disabling Live Fetch (Safety-First)

```js
const resp = await client.responses.create({
  model: "gpt-5.2",
  tools: [{ type: "web_search", external_web_access: false }],
  input: "What is the sunrise time in Tokyo today? Include a citation."
});
```

---

## 2) Deep Research (Responses API)

### Features

- A research-specialized reasoning model that produces long-form reports with citations and structure.
- The API version **does not ask follow-up questions** — provide purpose, scope, time range, exclusions, and output format upfront.

### Minimal Example (Python)

```python
from openai import OpenAI
client = OpenAI()

resp = client.responses.create(
    model="o3-deep-research",
    tools=[{"type": "web_search_preview"}],
    input=(
        "You are a research assistant. Survey AI-driven research automation systems "
        "(AI Scientist, Kosmos, PaperBench, SafeScientist). Extract 5 design patterns "
        "and propose how each could be integrated into a local-first research orchestrator. "
        "Include citations."
    ),
)
print(resp.output_text)
```

---

## 3) Codex CLI and Network Access

- **Codex Cloud** agents have **network disabled by default**. Enable per-environment with domain allowlists (to mitigate prompt injection / data leakage / license contamination risks).
- **Codex CLI / IDE Extension** has web search enabled by default (cached mode).

**Recommended split**:
- Use **API `web_search` / Deep Research** for research citations and evidence gathering (reproducible, auditable).
- Use **Codex's built-in web search** for implementation lookups (convenient, live if needed).

---

## 4) Orchestrator Integration Notes

### Fixing Evidence to the Ledger

- Store web search results as **URL + excerpt + retrieval date + summary** in `evidence/`.
- Store claims in `claims/` and require `claim → evidence` links.
- See: `orchestrator evidence add`, `orchestrator claim new`.

### Handling Ambiguity

Since the Deep Research API does not ask clarifying questions, resolve ambiguity externally:

1. **1st pass**: Detect missing requirements (list ambiguities).
2. **2nd pass**: Ask the user minimal clarifying questions.
3. **3rd pass**: Submit to Deep Research / `web_search`.
