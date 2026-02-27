# MCP Server (Experimental)

This directory contains a lightweight **Model Context Protocol (MCP)** stdio server.
It exposes the orchestrator's ledger and artifacts so that external tools (e.g., Codex, IDE extensions) can access the same data layer.

> The implementation is intentionally minimal and avoids external SDK dependencies.
> The stable contract is defined in `mcp_server/tool_manifest.yaml`.

## Minimum Interface (MVP)

- `search`: Given a text query, return matching items (ledger records / artifacts).
- `fetch`: Given an `id` returned by `search`, return the full content.

## Logical ID Design

IDs encode the data source unambiguously:

- `ledger:tasks/<task_id>`
- `ledger:projects/<project_id>`
- `artifact:<project_id>/<path>` — e.g., `artifact:my-project/paper/draft.md`

## Available Tools

When exposed as MCP tools, the following are supported:

- `ledger.list_projects`, `ledger.list_tasks(project_id, status)`, `ledger.get_task(task_id)`
- `artifact.list(project_id, prefix)`, `artifact.get(project_id, path)`, `artifact.put(project_id, path, content, mode)`
- `review.create_packet(project_id, stage, targets[])`
- `idea.list(project_id, status)`, `idea.get(idea_id)`, `idea.set_status(idea_id, status)`
- `smoke.ingest(project_id, result_path)`, `smoke.list(project_id, idea_id)`
- `topic.brief(project_id, idea_id, output_path, set_selected)`
- `evidence.add(project_id, kind, title, url, summary, ...)`, `evidence.list(project_id, idea_id)`, `evidence.get(evidence_id)`

## Running

```bash
python mcp_server/server.py --repo-root .
```

Quick test (list tools):

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python mcp_server/server.py --repo-root .
```

**Safety**: The `task.run` tool is **disabled by default**. Set `RESORCH_MCP_ALLOW_TASK_RUN=1` to enable it.

See also: `docs/CODEX_MCP_SETUP.md` for Codex integration.
