from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class CodexRunResult:
    returncode: int
    jsonl_path: Path
    last_message_path: Path
    stderr_path: Path


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_overrides(overrides: List[str]) -> List[str]:
    # Work around older/invalid ~/.codex/config.toml values (e.g., model_reasoning_effort="xhigh").
    has_effort = any(o.strip().startswith("model_reasoning_effort=") for o in overrides)
    if not has_effort:
        overrides = ['model_reasoning_effort="high"', *overrides]
    return overrides


def parse_jsonl_line(line: str) -> Optional[Dict[str, Any]]:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return {"_parse_error": True, "raw": stripped}


def run_codex_exec_jsonl(
    *,
    prompt: str,
    cd: Path,
    sandbox: str,
    model: Optional[str],
    config_overrides: List[str],
    jsonl_path: Path,
    last_message_path: Path,
    stderr_path: Path,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> CodexRunResult:
    _ensure_parent(jsonl_path)
    _ensure_parent(last_message_path)
    _ensure_parent(stderr_path)

    overrides = _normalize_overrides(list(config_overrides))

    cmd: List[str] = ["codex", "exec", "--json", "--sandbox", sandbox, "--cd", str(cd), "--skip-git-repo-check", "--output-last-message", str(last_message_path)]
    if model:
        cmd.extend(["--model", model])
    for ov in overrides:
        cmd.extend(["-c", ov])

    _WAIT_TIMEOUT = 300  # seconds; avoid infinite block on stuck child

    with stderr_path.open("w", encoding="utf-8") as stderr_f, jsonl_path.open("w", encoding="utf-8") as jsonl_f:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_f,
            text=True,
            bufsize=1,
        )
        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("Failed to open stdin/stdout pipes for codex subprocess")

        try:
            proc.stdin.write(prompt)
            if not prompt.endswith("\n"):
                proc.stdin.write("\n")
            proc.stdin.close()

            for line in proc.stdout:
                jsonl_f.write(line)
                event = parse_jsonl_line(line)
                if event is None:
                    continue
                if on_event:
                    on_event(event)
        finally:
            # Guarantee child cleanup regardless of exceptions.
            try:
                proc.stdout.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                proc.wait(timeout=_WAIT_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    return CodexRunResult(
        returncode=int(proc.returncode or 0),
        jsonl_path=jsonl_path,
        last_message_path=last_message_path,
        stderr_path=stderr_path,
    )


_PASS_FAIL_RE = re.compile(r"^(PASS|FAIL)\b[:\s]*(.*)", re.IGNORECASE)


def _is_pre_exec_review_command(cmd: str) -> bool:
    lower = cmd.lower()
    if "claude --print" in lower:
        return True
    # Pre-exec review gate instructions include these anchor phrases.
    return "codex exec" in lower and "the script at" in lower and "pass or fail" in lower


def _extract_pass_fail(output: str) -> Optional[re.Match[str]]:
    m = _PASS_FAIL_RE.match(output.strip())
    if m:
        return m
    for line in output.splitlines():
        m2 = _PASS_FAIL_RE.match(line.strip())
        if m2:
            return m2
    return None


def extract_pre_exec_review_results(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Extract pre-exec review gate PASS/FAIL results from Codex JSONL output.

    Looks for command_execution items whose command matches the review gate
    (Claude or Codex invocation) and whose output contains PASS or FAIL.
    Returns a list of dicts:
      {"script": "<path>", "verdict": "PASS"|"FAIL", "reason": "...", "command": "..."}
    """
    results: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        return results

    try:
        lines = jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return results

    for line in lines:
        event = parse_jsonl_line(line)
        if event is None:
            continue

        item = event.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") != "command_execution":
            continue

        cmd = str(item.get("command") or "")
        if not _is_pre_exec_review_command(cmd):
            continue

        output = str(item.get("aggregated_output") or "").strip()
        if not output:
            continue

        m = _extract_pass_fail(output)
        if not m:
            continue

        verdict = m.group(1).upper()
        reason = m.group(2).strip()

        # Try to extract the script path from the command.
        script = ""
        script_match = re.search(r"the script at\s+(\S+)", cmd, re.IGNORECASE)
        if script_match:
            script = script_match.group(1).rstrip(".")

        results.append({
            "script": script,
            "verdict": verdict,
            "reason": reason,
            "command": cmd[:500],
        })

    return results
