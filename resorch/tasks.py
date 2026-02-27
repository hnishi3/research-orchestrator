from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from resorch.artifacts import register_artifact
from resorch.codex_runner import extract_pre_exec_review_results, run_codex_exec_jsonl
from resorch.ledger import Ledger
from resorch.utils import read_text, utc_now_iso

# Default timeout for shell_exec tasks (seconds).
# Can be overridden per-task via spec.timeout_sec.
_SHELL_EXEC_DEFAULT_TIMEOUT_SEC: int = 3600


def _parse_task_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["spec"] = json.loads(out.pop("spec_json") or "{}")
    out["deps"] = json.loads(out.pop("deps_json") or "[]")
    return out


def _parse_task_run_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["meta"] = json.loads(out.pop("meta_json") or "{}")
    return out


def create_task(*, ledger: Ledger, project_id: str, task_type: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    _ = ledger.get_project(project_id)  # validate exists
    task = ledger.insert_task(
        task_id=uuid4().hex,
        project_id=project_id,
        task_type=task_type,
        status="created",
        spec=spec,
        deps=[],
    )
    return _parse_task_row(task)


def list_tasks(ledger: Ledger, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    return [_parse_task_row(t) for t in ledger.list_tasks(project_id=project_id)]


def get_task(ledger: Ledger, task_id: str) -> Dict[str, Any]:
    return _parse_task_row(ledger.get_task(task_id))


def _derive_event_type(event: Dict[str, Any]) -> str:
    if event.get("_parse_error"):
        return "parse_error"
    msg = event.get("msg")
    if isinstance(msg, dict) and "type" in msg:
        return str(msg["type"])
    if "prompt" in event:
        return "prompt"
    if "error" in event:
        return "error"
    return "event"


def _load_prompt_text(*, spec: Dict[str, Any], repo_root: Path, workspace: Path) -> str:
    if "prompt_file" in spec and spec["prompt_file"]:
        p = Path(str(spec["prompt_file"]))
        if p.is_absolute():
            raise SystemExit(
                f"prompt_file must be a relative path (got absolute: {p}). "
                "Relative paths are resolved under workspace/ or repo_root/."
            )
        candidates = [workspace / p, repo_root / p]
        # Ensure resolved paths stay within allowed roots.
        safe_candidates = []
        for c in candidates:
            resolved = c.resolve()
            try:
                resolved.relative_to(workspace.resolve())
                safe_candidates.append(resolved)
                continue
            except ValueError:
                pass
            try:
                resolved.relative_to(repo_root.resolve())
                safe_candidates.append(resolved)
                continue
            except ValueError:
                pass
            # Path escapes both roots — skip it.
        for c in safe_candidates:
            if c.exists():
                return read_text(c)
        raise SystemExit(f"prompt_file not found. tried: {', '.join(str(c) for c in safe_candidates)}")

    if "prompt" in spec and spec["prompt"]:
        return str(spec["prompt"])

    raise SystemExit("Task spec must include either 'prompt' or 'prompt_file'.")


def _append_output_schema(*, prompt: str, schema_path: Path) -> str:
    if not schema_path.exists():
        return prompt
    schema_txt = read_text(schema_path).strip()
    if not schema_txt:
        return prompt

    return (
        prompt.rstrip()
        + "\n\n"
        + "Return ONLY a single JSON object that conforms to this JSON Schema. No markdown.\n"
        + schema_txt
        + "\n"
    )


def _status_from_returncode(returncode: int, stderr_text: str) -> str:
    if returncode == 0:
        return "success"
    lower = stderr_text.lower()
    rate_limit_markers = [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "http 429",
        "status 429",
        "error 429",
        "(429)",
        "too many requests",
        "quota exceeded",
    ]
    if any(m in lower for m in rate_limit_markers):
        return "rate_limited"
    blocked_markers = [
        "Failed to deserialize",
        "unknown variant",
        "login",
        "authentication",
        "Unauthorized",
        "401",
        "403",
    ]
    if any(m.lower() in lower for m in blocked_markers):
        return "blocked"
    return "failed"


def run_task(*, ledger: Ledger, project: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    task_type = task["type"]
    if task_type == "codex_exec":
        return _run_codex_exec_task(ledger=ledger, project=project, task=task)
    if task_type == "shell_exec":
        return _run_shell_exec_task(ledger=ledger, project=project, task=task)
    if task_type == "review_fix":
        return _run_review_fix_task(ledger=ledger, project=project, task=task)
    raise SystemExit(f"Unsupported task type: {task_type} (supported: codex_exec, shell_exec, review_fix)")


def _run_shell_exec_task(*, ledger: Ledger, project: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    spec = task["spec"]
    cmd = spec.get("command")
    if not cmd:
        raise SystemExit("shell_exec requires spec.command (string or list).")

    shell = bool(spec.get("shell", False))
    if isinstance(cmd, str):
        # Default to shell=False (safer). If you need shell features, pass
        # ["bash","-lc", "..."] or set spec.shell=true explicitly.
        if not shell:
            cmd = shlex.split(cmd)
            if not cmd:
                raise SystemExit("shell_exec requires spec.command to be non-empty.")
        else:
            cmd = str(cmd)
    elif isinstance(cmd, list):
        cmd = [str(x) for x in cmd]
        shell = False
    else:
        raise SystemExit("shell_exec requires spec.command (string or list).")

    workspace = Path(project["repo_path"]).resolve()
    run_dir = (workspace / str(spec.get("cd", "")).strip()).resolve()
    try:
        run_dir.relative_to(workspace)
    except ValueError:
        raise SystemExit(f"shell_exec cd must be within workspace: {run_dir}")

    log_dir = ledger.paths.logs_dir / "tasks" / task["id"]
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{utc_now_iso().replace(':', '')}_stdout.log"
    stderr_path = log_dir / f"{utc_now_iso().replace(':', '')}_stderr.log"

    run_id = uuid4().hex
    ledger.insert_task_run(
        run_id=run_id,
        task_id=task["id"],
        status="running",
        jsonl_path=None,
        last_message_path=None,
        meta={"runner": "shell_exec", "cd": str(run_dir)},
    )
    ledger.update_task_status(task["id"], "running")

    timeout_sec = spec.get("timeout_sec")
    if timeout_sec is not None:
        try:
            timeout_sec = int(timeout_sec)
        except (TypeError, ValueError):
            timeout_sec = None
    if timeout_sec is None:
        timeout_sec = _SHELL_EXEC_DEFAULT_TIMEOUT_SEC

    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
        try:
            proc = subprocess.run(
                cmd, cwd=str(run_dir), shell=shell, text=True,
                stdout=out_f, stderr=err_f, timeout=timeout_sec,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1

    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    if timed_out:
        status = "failed"
        meta_extra = {"timed_out": True, "timeout_sec": timeout_sec}
    else:
        status = _status_from_returncode(returncode, stderr_text)
        meta_extra = {}

    ledger.finish_task_run(
        run_id=run_id, status=status, exit_code=int(returncode),
        meta_updates={"stdout_path": str(stdout_path), "stderr_path": str(stderr_path), **meta_extra},
    )
    ledger.update_task_status(task["id"], status)

    return {
        "task": get_task(ledger, task["id"]),
        "run": _parse_task_run_row(ledger.get_task_run(run_id)),
    }


def _run_codex_exec_task(*, ledger: Ledger, project: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    spec = task["spec"]
    workspace = Path(project["repo_path"]).resolve()

    cd_rel = str(spec.get("cd", "")).strip()
    cd_path = (workspace / cd_rel).resolve()
    try:
        cd_path.relative_to(workspace)
    except ValueError:
        raise SystemExit(f"codex_exec cd must be within workspace: {cd_path}")
    if not cd_path.exists():
        raise SystemExit(f"Task cd path does not exist: {cd_path}")

    sandbox = str(spec.get("sandbox", "danger-full-access"))
    model = spec.get("model")
    config_overrides = spec.get("config_overrides", [])
    if not isinstance(config_overrides, list):
        raise SystemExit("spec.config_overrides must be a list of strings.")

    prompt = _load_prompt_text(spec=spec, repo_root=ledger.paths.root, workspace=workspace)

    append_schema = spec.get("append_schema", True)
    schema_path = Path(spec.get("schema_path") or (ledger.paths.root / "schemas" / "task_result.schema.json"))
    if append_schema:
        prompt = _append_output_schema(prompt=prompt, schema_path=schema_path)

    log_dir = ledger.paths.logs_dir / "tasks" / task["id"]
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = log_dir / f"{utc_now_iso().replace(':', '')}.jsonl"
    last_message_path = log_dir / f"{utc_now_iso().replace(':', '')}_last_message.txt"
    stderr_path = log_dir / f"{utc_now_iso().replace(':', '')}_stderr.log"

    run_id = uuid4().hex
    with ledger.transaction():
        ledger.insert_task_run(
            run_id=run_id,
            task_id=task["id"],
            status="running",
            jsonl_path=str(jsonl_path),
            last_message_path=str(last_message_path),
            meta={"runner": "codex_exec", "sandbox": sandbox, "model": model, "cd": str(cd_path), "schema_path": str(schema_path)},
        )
        ledger.update_task_status(task["id"], "running")

    def _on_event(ev: Dict[str, Any]) -> None:
        ledger.insert_task_event(task_run_id=run_id, event_type=_derive_event_type(ev), data=ev)

    try:
        result = run_codex_exec_jsonl(
            prompt=prompt,
            cd=cd_path,
            sandbox=sandbox,
            model=model,
            config_overrides=[str(x) for x in config_overrides],
            jsonl_path=jsonl_path,
            last_message_path=last_message_path,
            stderr_path=stderr_path,
            on_event=_on_event,
        )
    except BaseException:
        # Ensure task is not left in 'running' state on unexpected errors.
        with ledger.transaction():
            ledger.finish_task_run(run_id=run_id, status="failed", exit_code=-1)
            ledger.update_task_status(task["id"], "failed")
        raise
    ledger.conn().commit()

    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    status = _status_from_returncode(result.returncode, stderr_text)

    # Best-effort: extract pre-exec review gate results from JSONL.
    pre_exec_reviews: List[Dict[str, Any]] = []
    try:
        pre_exec_reviews = extract_pre_exec_review_results(jsonl_path)
    except Exception:  # noqa: BLE001
        pass

    meta_updates: Dict[str, Any] = {}
    if pre_exec_reviews:
        meta_updates["pre_exec_review_results"] = pre_exec_reviews

    with ledger.transaction():
        ledger.finish_task_run(run_id=run_id, status=status, exit_code=int(result.returncode), meta_updates=meta_updates)
        ledger.update_task_status(task["id"], status)

    # Best-effort: parse task_result schema and register artifacts.
    if last_message_path.exists():
        try:
            task_result = json.loads(last_message_path.read_text(encoding="utf-8"))
            if not isinstance(task_result, dict):
                task_result = {}
            for a in (task_result.get("artifacts_created") or []):
                rel = a.get("path")
                if rel:
                    register_artifact(
                        ledger=ledger,
                        project=project,
                        kind=str(a.get("kind") or "artifact"),
                        relative_path=str(rel),
                        meta={"description": a.get("description")},
                    )
        except json.JSONDecodeError:
            pass

    task_dict = get_task(ledger, task["id"])
    if pre_exec_reviews:
        task_dict["pre_exec_review_results"] = pre_exec_reviews

    return {
        "task": task_dict,
        "run": _parse_task_run_row(ledger.get_task_run(run_id)),
    }


def _read_targets_for_prompt(*, workspace: Path, target_paths: List[str], max_chars: int) -> List[str]:
    blobs: List[str] = []
    for t in target_paths:
        p = (workspace / str(t)).resolve()
        try:
            p.relative_to(workspace)
        except ValueError:
            continue
        if not p.exists() or not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "\n…(truncated)…\n"
        blobs.append(f"\n--- FILE: {t} ---\n{txt}\n")
    return blobs


def _build_review_fix_prompt(*, task: Dict[str, Any], workspace: Path) -> str:
    spec = task.get("spec") or {}
    if not isinstance(spec, dict):
        spec = {}

    stage = str(spec.get("stage") or "")
    severity = str(spec.get("severity") or "")
    category = str(spec.get("category") or "")
    message = str(spec.get("message") or "")
    suggested_fix = spec.get("suggested_fix")
    target_paths = spec.get("target_paths") or []
    if not isinstance(target_paths, list):
        target_paths = []
    target_paths = [str(x) for x in target_paths if x]

    max_chars = int(spec.get("max_target_chars", 40_000))
    target_blobs = _read_targets_for_prompt(workspace=workspace, target_paths=target_paths, max_chars=max_chars)

    prompt = (
        "You are Codex. Apply the requested review fix to the workspace.\n"
        "Follow these rules:\n"
        "- Make the smallest safe change that satisfies the finding.\n"
        "- Update or add tests if needed.\n"
        "- Do not change unrelated files.\n\n"
        "Review finding:\n"
        + json.dumps(
            {
                "stage": stage,
                "severity": severity,
                "category": category,
                "message": message,
                "suggested_fix": suggested_fix,
                "target_paths": target_paths,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nTarget contents (may be truncated):\n"
        + "\n".join(target_blobs)
        + "\n"
    )
    return prompt


def _run_review_fix_task(*, ledger: Ledger, project: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    spec = task.get("spec") or {}
    if not isinstance(spec, dict):
        spec = {}

    workspace = Path(project["repo_path"]).resolve()
    cd_rel = str(spec.get("cd", "")).strip()
    cd_path = (workspace / cd_rel).resolve() if cd_rel else workspace
    try:
        cd_path.relative_to(workspace)
    except ValueError:
        raise SystemExit(f"review_fix cd must be within workspace: {cd_path}")
    if not cd_path.exists():
        raise SystemExit(f"Task cd path does not exist: {cd_path}")

    sandbox = str(spec.get("sandbox") or "danger-full-access")
    model = spec.get("model")
    config_overrides = spec.get("config_overrides", [])
    if not isinstance(config_overrides, list):
        raise SystemExit("spec.config_overrides must be a list of strings.")

    prompt = _build_review_fix_prompt(task=task, workspace=workspace.resolve())

    append_schema = spec.get("append_schema", True)
    schema_path = Path(spec.get("schema_path") or (ledger.paths.root / "schemas" / "task_result.schema.json"))
    if append_schema:
        prompt = _append_output_schema(prompt=prompt, schema_path=schema_path)

    # Store the prompt under the workspace for reproducibility.
    prompt_rel = f"runs/review_fix/{task['id']}/prompt.md"
    prompt_abs = (workspace / prompt_rel).resolve()
    prompt_abs.parent.mkdir(parents=True, exist_ok=True)
    prompt_abs.write_text(prompt, encoding="utf-8")
    try:
        register_artifact(
            ledger=ledger,
            project=project,
            kind="review_fix_prompt_md",
            relative_path=prompt_rel,
            meta={"task_id": task["id"]},
        )
    except SystemExit:
        # Best-effort: keep running even if artifact registration fails.
        pass

    log_dir = ledger.paths.logs_dir / "tasks" / task["id"]
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = log_dir / f"{utc_now_iso().replace(':', '')}.jsonl"
    last_message_path = log_dir / f"{utc_now_iso().replace(':', '')}_last_message.txt"
    stderr_path = log_dir / f"{utc_now_iso().replace(':', '')}_stderr.log"

    run_id = uuid4().hex
    with ledger.transaction():
        ledger.insert_task_run(
            run_id=run_id,
            task_id=task["id"],
            status="running",
            jsonl_path=str(jsonl_path),
            last_message_path=str(last_message_path),
            meta={
                "runner": "review_fix",
                "sandbox": sandbox,
                "model": model,
                "cd": str(cd_path),
                "schema_path": str(schema_path),
                "prompt_path": prompt_rel,
            },
        )
        ledger.update_task_status(task["id"], "running")

    def _on_event(ev: Dict[str, Any]) -> None:
        ledger.insert_task_event(task_run_id=run_id, event_type=_derive_event_type(ev), data=ev)

    try:
        result = run_codex_exec_jsonl(
            prompt=prompt,
            cd=cd_path,
            sandbox=sandbox,
            model=model,
            config_overrides=[str(x) for x in config_overrides],
            jsonl_path=jsonl_path,
            last_message_path=last_message_path,
            stderr_path=stderr_path,
            on_event=_on_event,
        )
    except BaseException:
        # Ensure task is not left in 'running' state on unexpected errors.
        with ledger.transaction():
            ledger.finish_task_run(run_id=run_id, status="failed", exit_code=-1)
            ledger.update_task_status(task["id"], "failed")
        raise
    ledger.conn().commit()

    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    status = _status_from_returncode(result.returncode, stderr_text)
    with ledger.transaction():
        ledger.finish_task_run(run_id=run_id, status=status, exit_code=int(result.returncode))
        ledger.update_task_status(task["id"], status)

    # Best-effort: parse task_result schema and register artifacts.
    if last_message_path.exists():
        try:
            task_result = json.loads(last_message_path.read_text(encoding="utf-8"))
            if not isinstance(task_result, dict):
                task_result = {}
            for a in (task_result.get("artifacts_created") or []):
                rel = a.get("path")
                if rel:
                    register_artifact(
                        ledger=ledger,
                        project=project,
                        kind=str(a.get("kind") or "artifact"),
                        relative_path=str(rel),
                        meta={"description": a.get("description")},
                    )
        except json.JSONDecodeError:
            pass

    return {
        "task": get_task(ledger, task["id"]),
        "run": _parse_task_run_row(ledger.get_task_run(run_id)),
    }
