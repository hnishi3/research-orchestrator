# Codex ↔ MCP server setup (template)

This repo includes a lightweight MCP-ish stdio server at:
- `mcp_server/server.py`

## Option A: Project-scoped config (recommended)
1) Create `.codex/config.toml` in the repository root:
   ```toml
   [mcp_servers.resorch]
   command = "python"
   args = ["mcp_server/server.py", "--repo-root", "<REPO_ROOT>"]
   ```
2) Replace `<REPO_ROOT>` with the absolute path to this repo
3) Ensure the project is marked as trusted in Codex

## Option B: Run the server manually
Run:

```bash
python mcp_server/server.py --repo-root .
```

Then connect from an MCP client that supports stdio transport.

## Notes / open issues
- Codex CLI versions differ. Some docs mention `codex mcp add/list/get`, but the CLI in this environment only exposes `codex mcp` as “run Codex as an MCP server”. If your Codex doesn’t support `mcp_servers` config, you’ll need the newer workflow or a different client.
- The stable interface contract for this repo is `mcp_server/tool_manifest.yaml`.
