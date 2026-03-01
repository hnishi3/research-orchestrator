# Security Policy

## Execution Model

Research-orchestrator delegates code execution to external tools (Codex CLI, shell).
The agent loop may run arbitrary code within project workspaces.

### Sandbox modes

| Mode | Permissions | Default for |
|------|-------------|-------------|
| `read-only` | No filesystem writes | Manual `task create` |
| `workspace-write` | Writes within workspace only | `review_fix` |
| `danger-full-access` | Unrestricted | Agent loop tasks |

`danger-full-access` is the agent loop default because Codex CLI's Landlock sandbox
can block legitimate workspace reads on some kernel configurations.

**Recommendation**: Run in an isolated environment (container, VM, dedicated user).

### MCP server

The bundled MCP server (`mcp_server/server.py`) disables `task.run` by default.
Set `RESORCH_MCP_ALLOW_TASK_RUN=1` to enable remote task execution.

## Reporting a Vulnerability

1. **Do not** open a public GitHub issue.
2. Use GitHub's private vulnerability reporting, or contact the maintainers directly.

We aim to acknowledge within 72 hours and provide a fix within 14 days for critical issues.

## Scope

This is research software for local or trusted environments.
It is not designed for multi-tenant or internet-facing deployment.
