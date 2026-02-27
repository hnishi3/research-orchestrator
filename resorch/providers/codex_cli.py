from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.utils import extract_json_object


class CodexCliError(RuntimeError):
    pass


@dataclass(frozen=True)
class CodexCliConfig:
    executable: str = "codex"
    timeout_sec: int = 1800
    model: Optional[str] = None
    # Codex CLI read-only sandbox prevents reading workspace files due to Landlock.
    sandbox: str = "danger-full-access"
    ephemeral: bool = True
    config_overrides: List[str] = field(default_factory=list)


def run_codex_exec_print_json(
    *,
    prompt: str,
    json_schema: Optional[Dict[str, Any]],
    workspace_dir: Path,
    config: Optional[CodexCliConfig] = None,
) -> Dict[str, Any]:
    """Run Codex CLI in non-interactive JSONL mode and return parsed output.

    This executes `codex exec --json` and uses `--output-schema` +
    `--output-last-message` so callers can consume a structured final JSON.
    """

    cfg = config or CodexCliConfig()
    exe = shutil.which(cfg.executable) or cfg.executable

    schema_fd, schema_tmp = tempfile.mkstemp(prefix="resorch-codex-schema-", suffix=".json")
    last_fd, last_tmp = tempfile.mkstemp(prefix="resorch-codex-last-", suffix=".txt")
    os.close(schema_fd)
    os.close(last_fd)
    schema_path = Path(schema_tmp)
    last_message_path = Path(last_tmp)
    try:
        schema_payload = _normalize_schema_for_codex(json_schema if isinstance(json_schema, dict) else {"type": "object"})
        schema_path.write_text(json.dumps(schema_payload, ensure_ascii=False), encoding="utf-8")

        cmd = [
            exe,
            "exec",
            "--json",
            "--sandbox",
            str(cfg.sandbox),
            "--cd",
            str(workspace_dir),
            "--skip-git-repo-check",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(last_message_path),
        ]
        if cfg.ephemeral:
            cmd.append("--ephemeral")
        if cfg.model:
            cmd.extend(["--model", str(cfg.model)])
        for ov in cfg.config_overrides:
            cmd.extend(["-c", str(ov)])

        env = dict(os.environ)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace_dir),
                text=True,
                capture_output=True,
                input=prompt if prompt.endswith("\n") else f"{prompt}\n",
                env=env,
                timeout=int(cfg.timeout_sec),
                check=False,
            )
        except FileNotFoundError as e:
            raise CodexCliError(f"Codex CLI executable not found: {cfg.executable}") from e
        except subprocess.TimeoutExpired as e:
            raise CodexCliError(f"Codex CLI timed out after {cfg.timeout_sec}s") from e

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise CodexCliError(f"Codex CLI failed (exit {proc.returncode}): {stderr}")

        stdout = proc.stdout or ""
        events: List[Dict[str, Any]] = []
        usage: Dict[str, Any] = {}
        event_errors: List[str] = []
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            events.append(event)
            if event.get("type") == "turn.completed":
                u = event.get("usage")
                if isinstance(u, dict):
                    usage = u
            if event.get("type") == "error":
                msg = event.get("message")
                if isinstance(msg, str) and msg.strip():
                    event_errors.append(msg.strip())
            if event.get("type") == "turn.failed":
                err = event.get("error")
                if isinstance(err, dict):
                    msg = err.get("message")
                    if isinstance(msg, str) and msg.strip():
                        event_errors.append(msg.strip())

        last_text = ""
        try:
            last_text = last_message_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            last_text = ""

        if event_errors:
            raise CodexCliError(f"Codex CLI returned turn failure: {event_errors[0]}")

        if not last_text:
            raise CodexCliError("Codex CLI produced empty final output.")

        return {
            "structured_output": _parse_final_json(last_text),
            "usage": usage,
            "events": events,
        }
    finally:
        try:
            schema_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            last_message_path.unlink(missing_ok=True)
        except OSError:
            pass


def extract_structured_output(cli_json: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured JSON output from Codex runner result."""

    structured = cli_json.get("structured_output")
    if isinstance(structured, dict):
        return structured

    result = cli_json.get("result")
    if isinstance(result, str) and result.strip():
        out = extract_json_object(result)
        if isinstance(out, dict):
            return out

    raise CodexCliError("Could not extract structured_output from Codex CLI JSON.")


def _parse_final_json(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    out = extract_json_object(text)
    if isinstance(out, dict):
        return out
    raise CodexCliError("Codex CLI final output was not a JSON object.")


def _normalize_schema_for_codex(schema: Any) -> Any:
    """Normalize JSON Schema for Codex structured output constraints.

    Codex/OpenAI structured outputs require object schemas to explicitly set
    additionalProperties=false. Normalize recursively so nested objects also
    satisfy this requirement.
    """
    if isinstance(schema, list):
        return [_normalize_schema_for_codex(x) for x in schema]
    if not isinstance(schema, dict):
        return schema

    out: Dict[str, Any] = {}
    for k, v in schema.items():
        if k in {"properties", "patternProperties"} and isinstance(v, dict):
            out[k] = {str(pk): _normalize_schema_for_codex(pv) for pk, pv in v.items()}
        elif k in {"items", "contains", "not", "if", "then", "else", "additionalProperties"}:
            out[k] = _normalize_schema_for_codex(v)
        elif k in {"allOf", "anyOf", "oneOf", "prefixItems"} and isinstance(v, list):
            out[k] = [_normalize_schema_for_codex(x) for x in v]
        else:
            out[k] = _normalize_schema_for_codex(v)

    is_object_like = out.get("type") == "object" or "properties" in out
    if is_object_like:
        # Required by Codex/OpenAI structured outputs for object schemas.
        out["additionalProperties"] = False
        props = out.get("properties")
        if not isinstance(props, dict):
            props = {}
            out["properties"] = props
        # Strict mode requires required[] to include every property key.
        out["required"] = list(props.keys())

    return out
