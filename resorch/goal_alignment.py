from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from resorch.providers.anthropic import AnthropicClient, extract_text
from resorch.providers.claude_code_cli import ClaudeCodeCliConfig, extract_structured_output, run_claude_code_print_json
from resorch.providers.codex_cli import (
    CodexCliConfig,
    extract_structured_output as codex_extract_structured_output,
    run_codex_exec_print_json,
)
from resorch.utils import extract_json_object


ALIGNMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "aligned": {"type": "boolean"},
        "drift_summary": {"type": ["string", "null"]},
    },
    "required": ["aligned"],
}

ALIGNMENT_PROMPT_TEMPLATE = (
    "Compare the research question (A) with the latest iteration objectives (B) below.\n"
    "Is B making progress toward A?\n\n"
    "## A. Research Question\n{research_question}\n\n"
    "## B. Latest Objectives\n{objectives_text}\n\n"
    'Respond in JSON: {{"aligned": bool, "drift_summary": "Explain drift in 1-2 sentences if any, otherwise null"}}'
)


@dataclass(frozen=True)
class AlignmentResult:
    aligned: bool
    drift_summary: Optional[str]  # None if aligned
    method: str  # "claude_code_cli" | "codex_cli" | "anthropic" | "skipped"


def check_goal_alignment(
    *,
    research_question: str,
    recent_objectives: list[str],
    provider: str = "claude_code_cli",
    model: str = "haiku",
    workspace_dir: Optional[Path] = None,
    reasoning_effort: Optional[str] = None,
) -> AlignmentResult:
    """Check goal alignment via the configured provider.

    provider="claude_code_cli": uses Claude Code CLI (subscription, no per-token cost)
    provider="codex_cli": uses Codex CLI
    provider="anthropic": uses Anthropic Messages API (per-token cost)

    Returns AlignmentResult. On any error, returns aligned=True (fail-open).
    """
    if not research_question.strip():
        return AlignmentResult(aligned=True, drift_summary=None, method="skipped")

    objectives_text = "\n".join(f"- {o}" for o in recent_objectives if str(o).strip())
    prompt = ALIGNMENT_PROMPT_TEMPLATE.format(
        research_question=research_question.strip(),
        objectives_text=objectives_text,
    )

    provider_norm = str(provider or "").strip().lower()
    try:
        if provider_norm == "anthropic":
            result = _call_anthropic(prompt=prompt, model=model)
            method = "anthropic"
        elif provider_norm == "codex_cli":
            result = _call_codex_cli(prompt=prompt, model=model, workspace_dir=workspace_dir, reasoning_effort=reasoning_effort)
            method = "codex_cli"
        else:
            result = _call_claude_code_cli(prompt=prompt, model=model, workspace_dir=workspace_dir)
            method = "claude_code_cli"

        aligned = bool(result.get("aligned", True))
        drift_summary = result.get("drift_summary")
        if drift_summary is not None and not isinstance(drift_summary, str):
            drift_summary = str(drift_summary)
        drift_summary = drift_summary.strip() if isinstance(drift_summary, str) else None
        if not drift_summary:
            drift_summary = None
        return AlignmentResult(aligned=aligned, drift_summary=drift_summary, method=method)
    except Exception:  # noqa: BLE001
        return AlignmentResult(aligned=True, drift_summary=None, method="skipped")


def _call_claude_code_cli(*, prompt: str, model: str, workspace_dir: Optional[Path]) -> Dict[str, Any]:
    if workspace_dir is None:
        raise ValueError("workspace_dir is required for provider=claude_code_cli")
    cfg = ClaudeCodeCliConfig(model=str(model) if model else None, timeout_sec=300, tools="Read", allowed_tools="Read")
    cli_json = run_claude_code_print_json(
        prompt=prompt,
        system_prompt=None,
        json_schema=ALIGNMENT_SCHEMA,
        workspace_dir=workspace_dir,
        config=cfg,
    )
    out = extract_structured_output(cli_json)
    if not isinstance(out, dict):
        raise ValueError("Claude Code structured_output was not an object.")
    return out


def _call_codex_cli(*, prompt: str, model: str, workspace_dir: Optional[Path], reasoning_effort: Optional[str] = None) -> Dict[str, Any]:
    if workspace_dir is None:
        raise ValueError("workspace_dir is required for provider=codex_cli")
    codex_model = str(model or "").strip()
    if codex_model.lower() in {"haiku", "sonnet", "opus"}:
        codex_model = ""
    overrides: list[str] = []
    if reasoning_effort:
        overrides.append(f'model_reasoning_effort="{reasoning_effort}"')
    # Codex CLI read-only sandbox prevents reading workspace files due to Landlock.
    cfg = CodexCliConfig(
        model=codex_model or None,
        timeout_sec=300,
        sandbox="danger-full-access",
        config_overrides=overrides,
    )
    cli_json = run_codex_exec_print_json(
        prompt=prompt,
        json_schema=None,  # schema example already in prompt; skip --output-schema for speed
        workspace_dir=workspace_dir,
        config=cfg,
    )
    out = codex_extract_structured_output(cli_json)
    if not isinstance(out, dict):
        raise ValueError("Codex structured_output was not an object.")
    return out


def _call_anthropic(*, prompt: str, model: str) -> Dict[str, Any]:
    """Call Anthropic Messages API directly and parse JSON from the text response."""
    m = str(model or "").strip()
    if m and "/" not in m and not m.startswith("claude-"):
        # Convenience: allow shorthand like "haiku"/"sonnet"/"opus".
        m = f"claude-{m}-4-5"
    client = AnthropicClient.from_env()
    resp = client.messages_create(
        model=m or "claude-haiku-4-5",
        system="You are a strict goal alignment checker. Output only a JSON object.",
        user=prompt,
        max_tokens=256,
        temperature=0.0,
    )
    text = extract_text(resp)
    if not text:
        raise ValueError("Anthropic response had no text content.")
    return extract_json_object(text)
