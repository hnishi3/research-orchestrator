from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from resorch.utils import extract_json_object


class ClaudeCodeCliError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClaudeCodeCliConfig:
    executable: str = "claude"
    timeout_sec: int = 900
    model: Optional[str] = "opus"
    tools: str = "Read"
    allowed_tools: str = "Read"
    no_session_persistence: bool = True
    permission_mode: Optional[str] = None


def run_claude_code_print_json(
    *,
    prompt: str,
    system_prompt: Optional[str],
    json_schema: Optional[Dict[str, Any]],
    workspace_dir: Path,
    config: Optional[ClaudeCodeCliConfig] = None,
) -> Dict[str, Any]:
    """Run Claude Code CLI in print mode and return parsed JSON output.

    This is intended for local/subscription usage (not Anthropic API key usage).
    To avoid accidental API billing, ANTHROPIC_API_KEY is removed from the
    subprocess environment.
    """

    cfg = config or ClaudeCodeCliConfig()

    exe = shutil.which(cfg.executable) or cfg.executable
    cmd = [
        exe,
        "--print",
        "--output-format",
        "json",
        "--add-dir",
        str(workspace_dir),
    ]
    if cfg.tools:
        cmd.extend(["--tools", cfg.tools])
    if cfg.allowed_tools:
        cmd.extend(["--allowedTools", cfg.allowed_tools])
    if cfg.no_session_persistence:
        cmd.append("--no-session-persistence")
    if cfg.permission_mode:
        cmd.extend(["--permission-mode", cfg.permission_mode])
    if cfg.model:
        cmd.extend(["--model", cfg.model])
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    if json_schema is not None:
        cmd.extend(["--json-schema", json.dumps(json_schema, ensure_ascii=False)])
    # Pass prompt via stdin to avoid OS "Argument list too long" errors
    # when the prompt includes large file contents.

    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("CLAUDECODE", None)  # Allow nested Claude Code CLI calls from within a Claude Code session

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workspace_dir),
            text=True,
            capture_output=True,
            input=prompt,
            env=env,
            timeout=int(cfg.timeout_sec),
            check=False,
        )
    except FileNotFoundError as e:
        raise ClaudeCodeCliError(f"Claude Code CLI executable not found: {cfg.executable}") from e
    except subprocess.TimeoutExpired as e:
        raise ClaudeCodeCliError(f"Claude Code CLI timed out after {cfg.timeout_sec}s") from e

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise ClaudeCodeCliError(f"Claude Code CLI failed (exit {proc.returncode}): {stderr}")

    stdout = (proc.stdout or "").strip()
    if not stdout:
        raise ClaudeCodeCliError("Claude Code CLI produced empty output.")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise ClaudeCodeCliError("Claude Code CLI output was not valid JSON.") from e


def extract_structured_output(cli_json: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the structured output JSON object from a CLI JSON response."""

    structured = cli_json.get("structured_output")
    if isinstance(structured, dict):
        return structured

    # Fallback: some output formats may only include a text result.
    result = cli_json.get("result")
    if isinstance(result, dict):
        structured2 = result.get("structured_output")
        if isinstance(structured2, dict):
            return structured2
        result = result.get("result")

    if isinstance(result, str) and result.strip():
        out = extract_json_object(result)
        if isinstance(out, dict):
            return out

    raise ClaudeCodeCliError("Could not extract structured_output from Claude Code CLI JSON.")
