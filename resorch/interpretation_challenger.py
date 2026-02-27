from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.providers.anthropic import AnthropicClient, extract_text
from resorch.providers.claude_code_cli import ClaudeCodeCliConfig, extract_structured_output, run_claude_code_print_json
from resorch.providers.codex_cli import (
    CodexCliConfig,
    extract_structured_output as codex_extract_structured_output,
    run_codex_exec_print_json,
)
from resorch.utils import extract_json_object


@dataclass(frozen=True)
class ChallengerCheck:
    item: str  # "statistical_reliability" | "baseline_strength" | ...
    status: str  # "ok" | "needs_review" | "insufficient_info"
    reason: str


@dataclass(frozen=True)
class ChallengerResult:
    checks: List[ChallengerCheck]
    flags: List[str]  # items with status != "ok"
    overall_concern_level: str  # "low" | "medium" | "high"


CHALLENGER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "status": {"type": "string", "enum": ["ok", "needs_review", "insufficient_info"]},
                    "reason": {"type": "string"},
                },
                "required": ["item", "status", "reason"],
            },
        },
        "flags": {"type": "array", "items": {"type": "string"}},
        "overall_concern_level": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["checks", "flags", "overall_concern_level"],
}


def maybe_challenge_interpretation_from_workspace(
    *,
    workspace_dir: Path,
    provider: str = "claude_code_cli",
    model: str = "sonnet",
    system_prompt_file: str = "prompts/challenger.md",
) -> Optional[ChallengerResult]:
    scoreboard_path = (workspace_dir / "results" / "scoreboard.json").resolve()
    if not scoreboard_path.exists():
        return None
    try:
        scoreboard_json = scoreboard_path.read_text(encoding="utf-8")
    except OSError:
        return None

    def _safe_read(rel: str) -> str:
        path = (workspace_dir / rel).resolve()
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    return challenge_interpretation(
        scoreboard_json=scoreboard_json,
        analysis_digest=_safe_read("notes/analysis_digest.md"),
        problem_md=_safe_read("notes/problem.md"),
        provider=provider,
        model=model,
        workspace_dir=workspace_dir,
        system_prompt_file=system_prompt_file,
    )


def challenge_interpretation(
    *,
    scoreboard_json: str,
    analysis_digest: str,
    problem_md: str,
    provider: str = "claude_code_cli",
    model: str = "sonnet",
    workspace_dir: Optional[Path] = None,
    system_prompt_file: str = "prompts/challenger.md",
) -> ChallengerResult:
    """Challenge experimental results via the configured provider.

    provider="claude_code_cli": uses Claude Code CLI (subscription, no per-token cost)
    provider="codex_cli": uses Codex CLI
    provider="anthropic": uses Anthropic Messages API (per-token cost)

    On error, returns a ChallengerResult with concern_level="low" (fail-open).
    """
    system_prompt = _load_system_prompt(workspace_dir=workspace_dir, system_prompt_file=system_prompt_file)
    user_prompt = (
        "Analyze the following experimental results.\n"
        "Return ONLY a JSON object that conforms to the provided JSON Schema.\n\n"
        f"## Research Question\n{problem_md}\n\n"
        f"## Scoreboard\n```json\n{scoreboard_json}\n```\n\n"
        f"## Analysis Digest\n{analysis_digest}\n"
    )

    provider_norm = str(provider or "").strip().lower()
    try:
        if provider_norm == "anthropic":
            result = _call_anthropic(prompt=user_prompt, model=model, system_prompt=system_prompt)
        elif provider_norm == "codex_cli":
            result = _call_codex_cli(
                prompt=user_prompt,
                model=model,
                system_prompt=system_prompt,
                workspace_dir=workspace_dir,
            )
        else:
            result = _call_claude_code_cli(
                prompt=user_prompt,
                model=model,
                system_prompt=system_prompt,
                workspace_dir=workspace_dir,
            )

        checks_in = result.get("checks") or []
        checks: List[ChallengerCheck] = []
        for c in checks_in:
            if not isinstance(c, dict):
                continue
            item = str(c.get("item") or "")
            status = str(c.get("status") or "")
            reason = str(c.get("reason") or "")
            if not item or not status or not reason:
                continue
            checks.append(ChallengerCheck(item=item, status=status, reason=reason))

        flags = result.get("flags") or []
        if not isinstance(flags, list):
            flags = []
        flags2 = [str(x) for x in flags if str(x).strip()]
        overall = str(result.get("overall_concern_level") or "low").strip().lower()
        if overall not in {"low", "medium", "high"}:
            overall = "low"
        return ChallengerResult(checks=checks, flags=flags2, overall_concern_level=overall)
    except Exception:  # noqa: BLE001
        return ChallengerResult(checks=[], flags=[], overall_concern_level="low")


def _load_system_prompt(*, workspace_dir: Optional[Path], system_prompt_file: str) -> str:
    rel = str(system_prompt_file or "").strip()
    if not rel:
        return ""

    candidates: List[Path] = []
    p = Path(rel)
    if p.is_absolute():
        candidates.append(p)
    else:
        if workspace_dir is not None:
            candidates.append((workspace_dir / p).resolve())
        # Fallback: repo root next to the `resorch/` package directory.
        repo_root = Path(__file__).resolve().parents[1]
        candidates.append((repo_root / p).resolve())

    for c in candidates:
        try:
            if c.exists() and c.is_file():
                return c.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
    return ""


def _call_claude_code_cli(*, prompt: str, model: str, system_prompt: str, workspace_dir: Optional[Path]) -> Dict[str, Any]:
    if workspace_dir is None:
        raise ValueError("workspace_dir is required for provider=claude_code_cli")
    cfg = ClaudeCodeCliConfig(model=str(model) if model else None, timeout_sec=900, tools="", allowed_tools="")
    cli_json = run_claude_code_print_json(
        prompt=prompt,
        system_prompt=system_prompt or None,
        json_schema=CHALLENGER_SCHEMA,
        workspace_dir=workspace_dir,
        config=cfg,
    )
    out = extract_structured_output(cli_json)
    if not isinstance(out, dict):
        raise ValueError("Claude Code structured_output was not an object.")
    return out


def _call_codex_cli(*, prompt: str, model: str, system_prompt: str, workspace_dir: Optional[Path]) -> Dict[str, Any]:
    if workspace_dir is None:
        raise ValueError("workspace_dir is required for provider=codex_cli")
    codex_model = str(model or "").strip()
    if codex_model.lower() in {"haiku", "sonnet", "opus"}:
        codex_model = ""
    cfg = CodexCliConfig(
        model=codex_model or None,
        timeout_sec=900,
        sandbox="read-only",
    )
    prompt_full = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    cli_json = run_codex_exec_print_json(
        prompt=prompt_full,
        json_schema=CHALLENGER_SCHEMA,
        workspace_dir=workspace_dir,
        config=cfg,
    )
    out = codex_extract_structured_output(cli_json)
    if not isinstance(out, dict):
        raise ValueError("Codex structured_output was not an object.")
    return out


def _call_anthropic(*, prompt: str, model: str, system_prompt: str) -> Dict[str, Any]:
    m = str(model or "").strip()
    if m and "/" not in m and not m.startswith("claude-"):
        m = f"claude-{m}-4-5"
    client = AnthropicClient.from_env()
    resp = client.messages_create(
        model=m or "claude-sonnet-4-5",
        system=system_prompt or "You are a rigorous experiment reviewer. Output only JSON.",
        user=prompt,
        max_tokens=1024,
        temperature=0.0,
    )
    text = extract_text(resp)
    if not text:
        raise ValueError("Anthropic response had no text content.")
    return extract_json_object(text)
