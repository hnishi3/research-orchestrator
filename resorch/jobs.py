from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from jsonschema import Draft202012Validator

from resorch.constraints import get_compute_config, load_constraints
from resorch.ledger import Ledger
from resorch.projects import get_project
from resorch.artifacts import put_artifact
from resorch.providers.http_json import HttpJsonError
from resorch.providers.openai import OpenAIClient, is_response_done
from resorch.providers.anthropic import AnthropicClient, extract_text as anthropic_extract_text
from resorch.providers.claude_code_cli import (
    ClaudeCodeCliConfig,
    ClaudeCodeCliError,
    extract_structured_output as claude_extract_structured_output,
    run_claude_code_print_json,
)
from resorch.providers.codex_cli import (
    CodexCliConfig,
    CodexCliError,
    extract_structured_output as codex_extract_structured_output,
    run_codex_exec_print_json,
)
from resorch.openai_tools import extract_json_object_from_response, run_response_to_completion_with_fallback
from resorch.reviews import ingest_review_result, write_review_request

log = logging.getLogger(__name__)
from resorch.utils import extract_json_object

# Providers that complete inline (no background polling needed).
_SYNC_PROVIDERS = frozenset({"anthropic", "claude_code_cli", "codex_cli"})


def _parse_job_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["spec"] = json.loads(out.pop("spec_json") or "{}")
    out["result"] = json.loads(out.pop("result_json") or "null")
    return out


def create_job(
    *, ledger: Ledger, project_id: Optional[str], provider: str, kind: str, spec: Dict[str, Any]
) -> Dict[str, Any]:
    if project_id:
        _ = ledger.get_project(project_id)  # validate
    job = ledger.insert_job(
        job_id=uuid4().hex,
        project_id=project_id,
        provider=provider,
        kind=kind,
        status="created",
        spec=spec,
    )
    return _parse_job_row(job)


def list_jobs(ledger: Ledger, *, project_id: Optional[str] = None, status: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    return [_parse_job_row(j) for j in ledger.list_jobs(project_id=project_id, status=status, limit=limit)]


def get_job(ledger: Ledger, job_id: str) -> Dict[str, Any]:
    return _parse_job_row(ledger.get_job(job_id))


def run_job(*, ledger: Ledger, job_id: str) -> Dict[str, Any]:
    job = get_job(ledger, job_id)
    provider = job["provider"]
    kind = job["kind"]

    if job["status"] in {"running", "submitted", "submitted_external", "running_external"}:
        return job
    if job["status"] in {"succeeded", "failed", "canceled", "completed_external", "failed_external"}:
        return job

    with ledger.transaction():
        ledger.update_job(job_id=job_id, status="running", started=True)
        ledger.insert_job_event(job_id=job_id, event_type="started", data={"provider": provider, "kind": kind})

    try:
        if provider == "openai":
            return _run_openai_job(ledger=ledger, job_id=job_id)
        if provider == "anthropic":
            return _run_anthropic_job(ledger=ledger, job_id=job_id)
        if provider in {"claude_code_cli", "codex_cli"}:
            return _run_claude_code_cli_job(ledger=ledger, job_id=job_id)
        if provider == "compute":
            return _run_compute_job(ledger=ledger, job_id=job_id)
    except HttpJsonError as e:
        with ledger.transaction():
            ledger.update_job(job_id=job_id, status="failed", error=f"HTTP {e.status}: {e.message}", result={"body": e.body_text}, finished=True)
            ledger.insert_job_event(job_id=job_id, event_type="error", data={"status": e.status, "message": e.message, "body": e.body_text})
        raise
    except Exception as e:  # noqa: BLE001
        with ledger.transaction():
            ledger.update_job(job_id=job_id, status="failed", error=str(e), finished=True)
            ledger.insert_job_event(job_id=job_id, event_type="error", data={"message": str(e)})
        raise

    raise SystemExit(f"job.run is not implemented for provider/kind: {provider}/{kind}")


def poll_job(*, ledger: Ledger, job_id: str) -> Dict[str, Any]:
    job = get_job(ledger, job_id)
    if job["status"] not in {"submitted", "running", "submitted_external", "running_external"}:
        return job

    provider = job["provider"]
    try:
        if provider == "openai":
            return _poll_openai_job(ledger=ledger, job_id=job_id)
        if provider == "compute":
            return _poll_compute_job(ledger=ledger, job_id=job_id)
    except HttpJsonError as e:
        with ledger.transaction():
            ledger.update_job(job_id=job_id, status="failed", error=f"HTTP {e.status}: {e.message}", result={"body": e.body_text}, finished=True)
            ledger.insert_job_event(job_id=job_id, event_type="error", data={"status": e.status, "message": e.message, "body": e.body_text})
        raise

    if provider in _SYNC_PROVIDERS:
        # Re-fetch so caller gets the latest status from the ledger.
        refreshed = get_job(ledger, job_id)
        if refreshed["status"] in {"submitted", "running", "submitted_external", "running_external"}:
            log.warning("Sync provider %s job %s still in non-terminal status %s", provider, job_id, refreshed["status"])
        return refreshed

    raise SystemExit(f"job.poll is not implemented for provider: {provider}")


def _compute_job_dir(*, workspace: Path, job_id: str) -> Path:
    return (workspace / "jobs" / job_id).resolve()


def _compute_command_string(spec: Dict[str, Any]) -> str:
    cmd = spec.get("command")
    if not cmd:
        raise SystemExit("compute job requires spec.command (string or list).")
    if isinstance(cmd, str):
        return str(cmd)
    if isinstance(cmd, list):
        parts = [str(x) for x in cmd if x is not None]
        if not parts:
            raise SystemExit("compute job requires spec.command to be non-empty.")
        return " ".join(shlex.quote(p) for p in parts)
    raise SystemExit("compute job requires spec.command (string or list).")


def _compute_run_dir(*, workspace: Path, spec: Dict[str, Any]) -> Path:
    cd_rel = str(spec.get("cd") or "").strip()
    run_dir = (workspace / cd_rel).resolve() if cd_rel else workspace.resolve()
    try:
        run_dir.relative_to(workspace)
    except ValueError:
        raise SystemExit(f"compute job cd must be within the workspace: cd={cd_rel!r}")
    if not run_dir.exists():
        raise SystemExit(f"compute job cd path does not exist: {run_dir}")
    return run_dir


def _compute_env(spec: Dict[str, Any]) -> Dict[str, str]:
    env = dict(os.environ)
    extra = spec.get("env") or {}
    if not isinstance(extra, dict):
        return env
    for k, v in extra.items():
        if k is None:
            continue
        env[str(k)] = str(v)
    return env


def _count_running_compute_jobs(*, ledger: Ledger, project_id: str, workspace: Path) -> int:
    n = 0
    for j in ledger.list_jobs(project_id=project_id, status=None, limit=500):
        if str(j.get("provider") or "") != "compute":
            continue
        if str(j.get("status") or "") != "running_external":
            continue
        jid = str(j.get("id") or "")
        if jid:
            if (_compute_job_dir(workspace=workspace, job_id=jid) / "exit_code.txt").exists():
                continue
        n += 1
    return n


def _run_compute_job(*, ledger: Ledger, job_id: str) -> Dict[str, Any]:
    job = get_job(ledger, job_id)
    if str(job.get("provider") or "") != "compute":
        raise SystemExit("Internal error: _run_compute_job called for non-compute job")

    project_id = job.get("project_id")
    if not project_id:
        raise SystemExit("compute jobs require job.project_id")
    project = get_project(ledger, str(project_id))
    workspace = Path(project["repo_path"]).resolve()
    spec = job.get("spec") or {}
    if not isinstance(spec, dict):
        spec = {}

    constraints = load_constraints(ledger=ledger, project_id=str(project_id))
    compute_cfg = get_compute_config(constraints)
    backend = str(spec.get("backend") or compute_cfg["backend"]).strip().lower()
    if backend not in {"local", "slurm"}:
        raise SystemExit("compute job spec.backend must be one of: local, slurm")

    job_dir = _compute_job_dir(workspace=workspace, job_id=job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    cmd_str = _compute_command_string(spec)
    run_dir = _compute_run_dir(workspace=workspace, spec=spec)

    # Write a stable job record artifact for debugging/repro.
    put_artifact(
        ledger=ledger,
        project=project,
        relative_path=f"jobs/{job_id}/job.json",
        content=json.dumps(
            {
                "job_id": job_id,
                "provider": "compute",
                "backend": backend,
                "cd": str(spec.get("cd") or ""),
                "command": cmd_str,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        mode="overwrite",
        kind="compute_job_json",
    )

    if backend == "slurm":
        return _submit_compute_slurm_job(ledger=ledger, job=job, project=project, workspace=workspace, cmd_str=cmd_str, run_dir=run_dir, compute_cfg=compute_cfg)

    return _submit_compute_local_job(ledger=ledger, job=job, project=project, workspace=workspace, cmd_str=cmd_str, run_dir=run_dir, compute_cfg=compute_cfg)


def _submit_compute_local_job(
    *,
    ledger: Ledger,
    job: Dict[str, Any],
    project: Dict[str, Any],
    workspace: Path,
    cmd_str: str,
    run_dir: Path,
    compute_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    job_id = str(job["id"])
    project_id = str(job["project_id"])
    job_dir = _compute_job_dir(workspace=workspace, job_id=job_id)

    script_rel = f"jobs/{job_id}/run_local.sh"
    script_path = (workspace / script_rel).resolve()
    cmd_quoted = shlex.quote(cmd_str)
    run_dir_quoted = shlex.quote(str(run_dir))
    started_at_path = shlex.quote(str((job_dir / "started_at.txt").resolve()))
    finished_at_path = shlex.quote(str((job_dir / "finished_at.txt").resolve()))
    stdout_path = shlex.quote(str((job_dir / "stdout.log").resolve()))
    stderr_path = shlex.quote(str((job_dir / "stderr.log").resolve()))
    exit_code_path = shlex.quote(str((job_dir / "exit_code.txt").resolve()))
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -u",
            "set -o pipefail",
            "",
            f"cd {run_dir_quoted}",
            f"date -Iseconds -u > {started_at_path}",
            f"bash -lc {cmd_quoted} > {stdout_path} 2> {stderr_path}",
            "code=$?",
            f"echo \"$code\" > {exit_code_path}",
            f"date -Iseconds -u > {finished_at_path}",
            "exit \"$code\"",
            "",
        ]
    )
    put_artifact(
        ledger=ledger,
        project=project,
        relative_path=script_rel,
        content=script,
        mode="overwrite",
        kind="compute_job_script_sh",
    )
    try:
        script_path.chmod(0o755)
    except OSError:
        pass

    max_parallel = int((compute_cfg.get("local") or {}).get("max_parallel") or 1)
    running = _count_running_compute_jobs(ledger=ledger, project_id=project_id, workspace=workspace)

    result_prev = job.get("result") if isinstance(job.get("result"), dict) else {}
    if running >= max_parallel:
        with ledger.transaction():
            ledger.update_job(
                job_id=job_id,
                status="submitted_external",
                result={**result_prev, "backend": "local", "job_dir": f"jobs/{job_id}", "queued": True, "max_parallel": max_parallel},
            )
            ledger.insert_job_event(
                job_id=job_id,
                event_type="compute.local.queued",
                data={"running_jobs": running, "max_parallel": max_parallel},
            )
        return get_job(ledger, job_id)

    env = _compute_env(job.get("spec") or {})
    try:
        proc = subprocess.Popen(
            ["bash", str(script_path)],
            cwd=str(job_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:  # noqa: BLE001
        with ledger.transaction():
            ledger.update_job(job_id=job_id, status="failed_external", error=str(e), result={**result_prev, "backend": "local"}, finished=True)
            ledger.insert_job_event(job_id=job_id, event_type="compute.local.submit_error", data={"message": str(e)})
        return get_job(ledger, job_id)

    with ledger.transaction():
        ledger.update_job(
            job_id=job_id,
            status="running_external",
            remote_id=str(proc.pid),
            result={**result_prev, "backend": "local", "job_dir": f"jobs/{job_id}", "pid": int(proc.pid), "queued": False},
        )
        ledger.insert_job_event(job_id=job_id, event_type="compute.local.submitted", data={"pid": int(proc.pid)})
    return get_job(ledger, job_id)


def _submit_compute_slurm_job(
    *,
    ledger: Ledger,
    job: Dict[str, Any],
    project: Dict[str, Any],
    workspace: Path,
    cmd_str: str,
    run_dir: Path,
    compute_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    job_id = str(job["id"])
    job_dir = _compute_job_dir(workspace=workspace, job_id=job_id)
    slurm_cfg = compute_cfg.get("slurm") or {}
    sbatch_cmd = str(slurm_cfg.get("sbatch") or "sbatch")
    extra_args = slurm_cfg.get("extra_sbatch_args") or []
    if not isinstance(extra_args, list):
        extra_args = []

    job_name = str((job.get("spec") or {}).get("job_name") or f"resorch-{job_id[:8]}")

    script_rel = f"jobs/{job_id}/slurm_job.sh"
    script_path = (workspace / script_rel).resolve()
    cmd_quoted = shlex.quote(cmd_str)
    run_dir_quoted = shlex.quote(str(run_dir))
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -u",
            "set -o pipefail",
            "",
            f"cd {run_dir_quoted}",
            f"bash -lc {cmd_quoted}",
            "",
        ]
    )
    put_artifact(
        ledger=ledger,
        project=project,
        relative_path=script_rel,
        content=script,
        mode="overwrite",
        kind="compute_slurm_script_sh",
    )
    try:
        script_path.chmod(0o755)
    except OSError:
        pass

    stdout_path = (job_dir / "stdout.log").resolve()
    stderr_path = (job_dir / "stderr.log").resolve()

    cmd = [
        sbatch_cmd,
        "--parsable",
        "--job-name",
        job_name,
        "--output",
        str(stdout_path),
        "--error",
        str(stderr_path),
        *[str(x) for x in extra_args if x],
        str(script_path),
    ]
    result_prev = job.get("result") if isinstance(job.get("result"), dict) else {}

    try:
        proc = subprocess.run(cmd, cwd=str(job_dir), check=False, text=True, capture_output=True)
    except FileNotFoundError as e:
        with ledger.transaction():
            ledger.update_job(job_id=job_id, status="failed_external", error=str(e), result={**result_prev, "backend": "slurm"}, finished=True)
            ledger.insert_job_event(job_id=job_id, event_type="compute.slurm.sbatch_missing", data={"message": str(e)})
        return get_job(ledger, job_id)

    if proc.returncode != 0:
        with ledger.transaction():
            ledger.update_job(
                job_id=job_id,
                status="failed_external",
                error=(proc.stderr or proc.stdout or f"sbatch exit {proc.returncode}").strip(),
                result={**result_prev, "backend": "slurm", "sbatch_cmd": cmd},
                finished=True,
            )
            ledger.insert_job_event(
                job_id=job_id,
                event_type="compute.slurm.sbatch_failed",
                data={"returncode": int(proc.returncode), "stdout": proc.stdout, "stderr": proc.stderr},
            )
        return get_job(ledger, job_id)

    raw = (proc.stdout or "").strip()
    job_remote = raw.split(";", 1)[0].strip() if raw else ""
    if not job_remote:
        job_remote = raw
    if not job_remote:
        with ledger.transaction():
            ledger.update_job(job_id=job_id, status="failed_external", error="Failed to parse slurm job id from sbatch output", result={**result_prev, "backend": "slurm", "sbatch_stdout": proc.stdout}, finished=True)
            ledger.insert_job_event(job_id=job_id, event_type="compute.slurm.sbatch_parse_error", data={"stdout": proc.stdout, "stderr": proc.stderr})
        return get_job(ledger, job_id)

    with ledger.transaction():
        ledger.update_job(
            job_id=job_id,
            status="submitted_external",
            remote_id=str(job_remote),
            result={**result_prev, "backend": "slurm", "job_dir": f"jobs/{job_id}", "slurm_job_id": str(job_remote), "sbatch_cmd": cmd},
        )
        ledger.insert_job_event(job_id=job_id, event_type="compute.slurm.submitted", data={"slurm_job_id": str(job_remote)})
    return get_job(ledger, job_id)


def _poll_compute_job(*, ledger: Ledger, job_id: str) -> Dict[str, Any]:
    job = get_job(ledger, job_id)
    if str(job.get("provider") or "") != "compute":
        raise SystemExit("Internal error: _poll_compute_job called for non-compute job")

    project_id = job.get("project_id")
    if not project_id:
        raise SystemExit("compute jobs require job.project_id")
    project = get_project(ledger, str(project_id))
    workspace = Path(project["repo_path"]).resolve()

    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    backend = str(result.get("backend") or (job.get("spec") or {}).get("backend") or "").strip().lower()
    if not backend:
        constraints = load_constraints(ledger=ledger, project_id=str(project_id))
        backend = str(get_compute_config(constraints)["backend"]).strip().lower()
    if backend not in {"local", "slurm"}:
        raise SystemExit("compute job backend must be one of: local, slurm")

    if backend == "slurm":
        out = _poll_compute_slurm_job(ledger=ledger, job=job, project=project, workspace=workspace)
    else:
        out = _poll_compute_local_job(ledger=ledger, job=job, project=project, workspace=workspace)

    # If the job just reached a terminal state, create an ingest task (once).
    out_job = get_job(ledger, job_id)
    if str(out_job.get("status") or "") in {"completed_external", "failed_external"}:
        _ensure_compute_ingest_task(ledger=ledger, job=out_job, project=project)
    return get_job(ledger, job_id)


def _poll_compute_local_job(*, ledger: Ledger, job: Dict[str, Any], project: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
    job_id = str(job["id"])
    project_id = str(job["project_id"])
    job_dir = _compute_job_dir(workspace=workspace, job_id=job_id)
    exit_code_path = (job_dir / "exit_code.txt").resolve()

    # If queued, try to start when a slot is available.
    if str(job.get("status") or "") == "submitted_external" and not job.get("remote_id"):
        constraints = load_constraints(ledger=ledger, project_id=project_id)
        compute_cfg = get_compute_config(constraints)
        max_parallel = int((compute_cfg.get("local") or {}).get("max_parallel") or 1)
        running = _count_running_compute_jobs(ledger=ledger, project_id=project_id, workspace=workspace)
        if running < max_parallel:
            cmd_str = _compute_command_string(job.get("spec") or {})
            run_dir = _compute_run_dir(workspace=workspace, spec=job.get("spec") or {})
            return _submit_compute_local_job(ledger=ledger, job=job, project=project, workspace=workspace, cmd_str=cmd_str, run_dir=run_dir, compute_cfg=compute_cfg)

    if not exit_code_path.exists():
        return get_job(ledger, job_id)

    try:
        code_raw = exit_code_path.read_text(encoding="utf-8", errors="replace").strip()
        exit_code = int(code_raw)
    except (OSError, ValueError):
        exit_code = 1

    status = "completed_external" if exit_code == 0 else "failed_external"
    result_prev = job.get("result") if isinstance(job.get("result"), dict) else {}
    with ledger.transaction():
        ledger.update_job(
            job_id=job_id,
            status=status,
            result={**result_prev, "backend": "local", "job_dir": f"jobs/{job_id}", "exit_code": exit_code},
            finished=True,
        )
        ledger.insert_job_event(job_id=job_id, event_type="compute.local.finished", data={"exit_code": exit_code, "status": status})
    return get_job(ledger, job_id)


def _slurm_state_to_status(state: str) -> str:
    s = (state or "").strip().upper()
    if not s:
        return ""
    s = s.split()[0]
    s = s.split("+")[0]
    if s in {"PENDING", "CONFIGURING"}:
        return "submitted_external"
    if s in {"RUNNING", "COMPLETING"}:
        return "running_external"
    if s in {"COMPLETED"}:
        return "completed_external"
    if s in {"FAILED", "CANCELLED", "CANCELLED_BY_SIGNAL", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}:
        return "failed_external"
    return ""


def _poll_compute_slurm_job(*, ledger: Ledger, job: Dict[str, Any], project: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
    job_id = str(job["id"])
    project_id = str(job["project_id"])
    result_prev = job.get("result") if isinstance(job.get("result"), dict) else {}

    constraints = load_constraints(ledger=ledger, project_id=project_id)
    slurm_cfg = get_compute_config(constraints).get("slurm") or {}
    sacct_cmd = str(slurm_cfg.get("sacct") or "sacct")
    squeue_cmd = str(slurm_cfg.get("squeue") or "squeue")

    slurm_job_id = str(job.get("remote_id") or result_prev.get("slurm_job_id") or "").strip()
    if not slurm_job_id:
        raise SystemExit("slurm compute job is missing remote_id (slurm_job_id)")

    # Prefer sacct for terminal state + exit code.
    state = ""
    exit_code: Optional[int] = None
    try:
        proc = subprocess.run(
            [sacct_cmd, "-j", slurm_job_id, "-n", "-P", "-o", "JobIDRaw,State,ExitCode"],
            cwd=str(_compute_job_dir(workspace=workspace, job_id=job_id)),
            check=False,
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0:
            for line in (proc.stdout or "").splitlines():
                parts = line.strip().split("|")
                if len(parts) < 3:
                    continue
                jid_raw, st, ec = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if jid_raw != slurm_job_id:
                    continue
                state = st
                ec0 = ec.split(":", 1)[0].strip()
                try:
                    exit_code = int(ec0)
                except ValueError:
                    exit_code = None
                break
    except FileNotFoundError:
        proc = None  # type: ignore[assignment]

    status = _slurm_state_to_status(state)
    if not status:
        # Fallback: squeue for non-terminal state.
        try:
            proc2 = subprocess.run(
                [squeue_cmd, "-j", slurm_job_id, "-h", "-o", "%T"],
                cwd=str(_compute_job_dir(workspace=workspace, job_id=job_id)),
                check=False,
                text=True,
                capture_output=True,
            )
            if proc2.returncode == 0:
                state2 = (proc2.stdout or "").strip()
                status = _slurm_state_to_status(state2)
                if not state:
                    state = state2
        except FileNotFoundError:
            status = ""

    if not status:
        # Unknown: keep current state.
        return get_job(ledger, job_id)

    updates: Dict[str, Any] = {"backend": "slurm", "job_dir": f"jobs/{job_id}", "slurm_job_id": slurm_job_id, "slurm_state": state}
    if exit_code is not None:
        updates["exit_code"] = exit_code

    finished = status in {"completed_external", "failed_external"}
    if finished and status == "completed_external" and exit_code not in {None, 0}:
        status = "failed_external"

    with ledger.transaction():
        ledger.update_job(job_id=job_id, status=status, result={**result_prev, **updates}, finished=finished)
        ledger.insert_job_event(job_id=job_id, event_type="compute.slurm.poll", data={"slurm_job_id": slurm_job_id, "state": state, "status": status, "exit_code": exit_code})

    return get_job(ledger, job_id)


def _ensure_compute_ingest_task(*, ledger: Ledger, job: Dict[str, Any], project: Dict[str, Any]) -> None:
    from resorch.tasks import create_task  # local import to avoid heavy imports on hot paths

    job_id = str(job["id"])
    project_id = str(job["project_id"])
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    existing = result.get("ingest_task_id")
    if isinstance(existing, str) and existing.strip():
        return

    prompt_rel = f"jobs/{job_id}/ingest_prompt.md"
    prompt = "\n".join(
        [
            "# Ingest external compute job results",
            "",
            f"- Job id: `{job_id}`",
            f"- Status: `{job.get('status')}`",
            f"- Backend: `{(result.get('backend') or '')}`",
            f"- Job dir: `{result.get('job_dir') or ('jobs/' + job_id)}`",
            "",
            "## Instructions",
            "1. Inspect the job directory logs (`stdout.log`, `stderr.log`, `exit_code.txt`) and any output artifacts produced under the workspace.",
            "2. Summarize the results as evidence and update:",
            "   - `results/scoreboard.json` (update `primary_metric.current.mean` and add a run entry under `runs[]`)",
            "   - `notes/analysis_digest.md` (fill in `## Latest` and update next actions)",
            "3. If the job failed, capture the failure mode and propose the next cheapest debugging experiment.",
            "",
            "Return JSON per the task_result schema and list any artifacts created/updated.",
            "",
        ]
    )
    put_artifact(
        ledger=ledger,
        project=project,
        relative_path=prompt_rel,
        content=prompt,
        mode="overwrite",
        kind="compute_ingest_prompt_md",
    )

    task = create_task(
        ledger=ledger,
        project_id=project_id,
        task_type="codex_exec",
        spec={
            "cd": ".",
            "sandbox": "danger-full-access",
            "prompt_file": prompt_rel,
            "append_schema": True,
        },
    )

    with ledger.transaction():
        ledger.update_job(job_id=job_id, result={**result, "ingest_task_id": str(task["id"])})
        ledger.insert_job_event(job_id=job_id, event_type="compute.ingest_task_created", data={"task_id": str(task["id"])})


def _run_openai_job(*, ledger: Ledger, job_id: str) -> Dict[str, Any]:
    job = get_job(ledger, job_id)
    kind = job["kind"]
    spec = job["spec"]
    if kind in {"review", "code_review"}:
        return _run_openai_review_job(ledger=ledger, job_id=job_id)
    if kind not in {"response", "responses.create", "deep_research"}:
        raise SystemExit(f"Unsupported openai job kind: {kind}")

    payload: Dict[str, Any]
    if kind == "deep_research":
        query = spec.get("query")
        if not query:
            raise SystemExit("openai/deep_research job requires spec.query (string).")
        model = str(spec.get("model") or "o3-deep-research")
        tools = spec.get("tools")
        if tools is None:
            tools = [{"type": "web_search_preview"}]
        background = spec.get("background")
        if background is None:
            background = True

        payload = {"model": model, "input": query, "tools": tools, "background": bool(background)}
        overrides = spec.get("payload_overrides")
        if overrides is not None:
            if not isinstance(overrides, dict):
                raise SystemExit("spec.payload_overrides must be an object.")
            payload.update(overrides)
    else:
        raw_payload = spec.get("payload")
        if not isinstance(raw_payload, dict):
            raise SystemExit("OpenAI job spec requires `payload` (object).")
        payload = raw_payload

    client = OpenAIClient.from_env()
    resp = client.responses_create(payload)

    remote_id = resp.get("id")
    done = is_response_done(resp)
    if done is True:
        status = "succeeded"
        finished = True
    elif done is False:
        status = "failed"
        finished = True
    else:
        # Even if `background` is omitted/false, treat non-terminal statuses as submitted
        # and let the user poll (or use webhooks) to reach a terminal state.
        status = "submitted"
        finished = False

    with ledger.transaction():
        ledger.update_job(job_id=job_id, status=status, remote_id=str(remote_id) if remote_id else None, result=resp, finished=finished)
        ledger.insert_job_event(job_id=job_id, event_type="openai.responses_create", data={"remote_id": remote_id, "status": resp.get("status")})

    _maybe_write_job_artifact(ledger=ledger, job=job, payload=payload, result=resp)

    return get_job(ledger, job_id)


def _run_openai_review_job(*, ledger: Ledger, job_id: str) -> Dict[str, Any]:
    job = get_job(ledger, job_id)
    kind = job["kind"]
    spec = job["spec"]
    if kind not in {"review", "code_review"}:
        raise SystemExit(f"Unsupported openai job kind: {kind}")
    is_code_review = kind == "code_review"

    project_id = job.get("project_id") or spec.get("project_id")
    if not project_id:
        raise SystemExit("openai/review job requires job.project_id (or spec.project_id).")

    stage = str(spec.get("stage") or "analysis")
    mode = str(spec.get("mode") or "balanced")
    targets = spec.get("targets") or []
    questions = spec.get("questions") or []
    if not isinstance(targets, list) or not targets:
        raise SystemExit("openai/review job requires spec.targets (array of paths).")
    if not isinstance(questions, list) or not questions:
        questions = ["Any major issues? Any missing baselines/related work?"]

    project = get_project(ledger, str(project_id))
    req = write_review_request(
        ledger=ledger,
        project=project,
        stage=stage,
        mode=mode,
        targets=[str(x) for x in targets],
        questions=[str(x) for x in questions],
        rubric=spec.get("rubric"),
        time_budget_minutes=spec.get("time_budget_minutes"),
    )

    ws = Path(project["repo_path"]).resolve()
    include_contents = bool(spec.get("include_target_contents", True))
    max_chars = int(spec.get("max_target_chars", 60_000))

    # Build (path, blob) pairs so we can filter by extension later.
    target_pairs: List[Tuple[str, str]] = []
    if include_contents:
        for t in targets:
            p = (ws / str(t)).resolve()
            try:
                p.relative_to(ws)
            except ValueError:
                continue
            content = _safe_read_text(p, max_chars=max_chars)
            if not content:
                continue
            target_pairs.append((str(t), f"\n--- FILE: {t} ---\n{content}\n"))

    schema_path = ledger.paths.root / "review" / "review_result.schema.json"
    schema_raw = _safe_read_text(schema_path, max_chars=200_000)
    if schema_raw.strip():
        schema_obj = json.loads(schema_raw)
    else:
        # Keep the job usable in minimal test repos; production repos should include this file.
        schema_obj = {"type": "object"}
    schema_txt = json.dumps(schema_obj, ensure_ascii=False, indent=2)

    request_json = _safe_read_text(Path(req["request_json_path"]), max_chars=60_000)
    default_system_prompt_path = "prompts/reviewer_code.md" if is_code_review else None
    system_prompt_path = spec.get("system_prompt_file") or default_system_prompt_path
    system_prompt = ""
    if system_prompt_path:
        system_prompt = _safe_read_text((ledger.paths.root / str(system_prompt_path)).resolve(), max_chars=40_000).strip()
    if not system_prompt:
        system_prompt = (
            "You are a strict, detail-oriented pre-execution code reviewer."
            if is_code_review
            else "You are a strict, detail-oriented research reviewer."
        )

    _MAX_TOTAL_TARGET_CHARS = int(spec.get("max_total_target_chars", 300_000))
    total_target_chars = sum(len(b) for _, b in target_pairs)
    if total_target_chars > _MAX_TOTAL_TARGET_CHARS:
        log.warning(
            "OpenAI review target contents too large (%d chars > %d); prioritising code files.",
            total_target_chars, _MAX_TOTAL_TARGET_CHARS,
        )
        # Keep code/config files, drop data/binary blobs first.
        _CODE_EXTS = {".py", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".json", ".md", ".txt", ".ini", ".conf"}
        code_blobs: List[str] = []
        other_names: List[str] = []
        for tpath, blob in target_pairs:
            if Path(tpath).suffix.lower() in _CODE_EXTS:
                code_blobs.append(blob)
            else:
                other_names.append(tpath)
        # If code blobs alone still exceed the limit, truncate the list.
        code_total = sum(len(b) for b in code_blobs)
        if code_total > _MAX_TOTAL_TARGET_CHARS:
            kept: List[str] = []
            running = 0
            for b in code_blobs:
                if running + len(b) > _MAX_TOTAL_TARGET_CHARS:
                    break
                kept.append(b)
                running += len(b)
            code_blobs = kept
        target_section = (
            "Target contents (code/config files; may be truncated):\n"
            + "\n".join(code_blobs)
        )
        if other_names:
            target_section += (
                "\n\nOther changed files (data/binary, contents omitted):\n"
                + "\n".join(f"  - {n}" for n in other_names)
            )
    else:
        target_section = (
            "Target contents (may be truncated):\n"
            + "\n".join(b for _, b in target_pairs)
        )

    research_ctx = _build_research_context_section(ws) if is_code_review else ""
    challenger_section = _build_challenger_section(spec)

    base_prompt = (
        system_prompt.rstrip()
        + "\n"
        "Call the function `submit_review` with arguments that conform to the following JSON Schema.\n"
        "Do not wrap in markdown or code fences.\n\n"
        + schema_txt
        + "\n\nReview request JSON:\n"
        + request_json
        + "\n\n"
        + target_section
        + research_ctx
        + challenger_section
    )

    reviewer = str(spec.get("reviewer") or "openai")
    model = str(spec.get("model") or os.environ.get("OPENAI_REVIEW_MODEL") or "gpt-5.2-pro")
    background = spec.get("background")
    if background is None:
        background = True

    tool_name = "submit_review"
    tool_desc = "Submit a single ReviewResult JSON object."
    tool_variant = {
        "tools": [{"type": "function", "name": tool_name, "description": tool_desc, "parameters": schema_obj}],
        "tool_choice": {"type": "function", "name": tool_name},
    }

    max_retries = int(spec.get("max_retries", 2))
    last_err: Optional[str] = None

    client = OpenAIClient.from_env()
    validator = Draft202012Validator(schema_obj)

    for attempt in range(max_retries + 1):
        prompt = base_prompt
        if last_err:
            prompt = (
                base_prompt
                + "\n\nYour previous output did not match the schema.\n"
                + "Fix and re-submit by calling submit_review again.\n\nSchema error:\n"
                + last_err
            )

        base_payload: Dict[str, Any] = {"model": model, "input": prompt, "background": bool(background)}
        temperature = spec.get("temperature")
        if temperature is not None:
            base_payload["temperature"] = float(temperature)

        payload_variants = [{**base_payload, **tool_variant}]
        resp = run_response_to_completion_with_fallback(client=client, payload_variants=payload_variants)
        remote_id = resp.get("id")

        out = extract_json_object_from_response(resp, function_name=tool_name)
        out["project_id"] = str(project_id)
        out["stage"] = stage
        out["reviewer"] = reviewer

        errors = sorted(validator.iter_errors(out), key=lambda e: e.json_path)
        if errors:
            if attempt >= max_retries:
                last_err = errors[0].message
                raise ValueError(f"openai/review output schema validation failed: {last_err}")
            last_err = errors[0].message
            continue

        reviews_dir = ws / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        uid = uuid4().hex[:6]
        resp_path = reviews_dir / f"RESP-{stage}-{_today_ymd()}-{uid}-{reviewer}.json"
        resp_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        ingested = ingest_review_result(
            ledger=ledger,
            result_path=resp_path,
            create_fix_tasks="major_blocker" if is_code_review else "none",
        )

        result_obj = {
            "provider_response": resp,
            "review_result": out,
            "ingested": {
                "stored_path": ingested.get("stored_path"),
                "review_id": ingested.get("review", {}).get("id"),
                "tasks_created": len(ingested.get("tasks_created") or []),
            },
        }

        with ledger.transaction():
            ledger.update_job(
                job_id=job_id,
                status="succeeded",
                remote_id=str(remote_id) if remote_id else None,
                result=result_obj,
                finished=True,
            )
            ledger.insert_job_event(
                job_id=job_id,
                event_type="openai.review.completed",
                data={"reviewer": reviewer, "stage": stage, "model": model, "remote_id": remote_id},
            )
        return get_job(ledger, job_id)

    raise SystemExit("openai/review job failed unexpectedly (exhausted retries).")


def _poll_openai_job(*, ledger: Ledger, job_id: str) -> Dict[str, Any]:
    job = get_job(ledger, job_id)
    remote_id = job.get("remote_id")
    if not remote_id:
        raise SystemExit("Cannot poll OpenAI job without remote_id.")

    client = OpenAIClient.from_env()
    resp = client.responses_get(str(remote_id))
    done = is_response_done(resp)

    if done is True:
        status = "succeeded"
        finished = True
    elif done is False:
        status = "failed"
        finished = True
    else:
        status = "running"
        finished = False

    with ledger.transaction():
        ledger.update_job(job_id=job_id, status=status, result=resp, finished=finished)
        ledger.insert_job_event(job_id=job_id, event_type="openai.responses_get", data={"remote_id": remote_id, "status": resp.get("status")})

    _maybe_write_job_artifact(ledger=ledger, job=job, payload=None, result=resp)

    return get_job(ledger, job_id)


def _maybe_write_job_artifact(
    *,
    ledger: Ledger,
    job: Dict[str, Any],
    payload: Optional[Dict[str, Any]],
    result: Dict[str, Any],
) -> None:
    spec = job.get("spec") or {}
    project_id = job.get("project_id")
    artifact_path = spec.get("artifact_path")
    if not project_id or not artifact_path:
        return
    project = get_project(ledger, str(project_id))

    kind = str(spec.get("artifact_kind") or f"{job['provider']}_{job['kind']}")
    fmt = str(spec.get("artifact_format") or "json").lower()
    if fmt == "text":
        txt = result.get("output_text")
        if not isinstance(txt, str):
            txt = json.dumps(result, ensure_ascii=False, indent=2)
        content = txt
        if not content.endswith("\n"):
            content += "\n"
    else:
        content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    put_artifact(ledger=ledger, project=project, relative_path=str(artifact_path), content=content, mode="overwrite", kind=kind)


def _safe_read_text(path: Path, max_chars: int = 60_000) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    # Strip null bytes that break subprocess argument passing.
    txt = txt.replace("\x00", "")
    if len(txt) > max_chars:
        return txt[:max_chars] + "\n…(truncated)…\n"
    return txt


def _build_research_context_section(ws: Path, max_chars: int = 10_000) -> str:
    """Read notes/problem.md and notes/method.md and return a context block for code review."""
    parts: List[str] = []
    for rel in ("notes/problem.md", "notes/method.md"):
        p = (ws / rel).resolve()
        txt = _safe_read_text(p, max_chars=max_chars)
        if txt.strip():
            parts.append(f"\n--- RESEARCH CONTEXT: {rel} ---\n{txt}\n")
    return "".join(parts)


def _build_challenger_section(spec: Dict[str, Any]) -> str:
    """Build a challenger-flags section for the review prompt."""
    flags = spec.get("challenger_flags")
    if not flags or not isinstance(flags, list):
        return ""
    flags_text = "\n".join(f"  - {f}" for f in flags if f)
    if not flags_text.strip():
        return ""
    return (
        "\n\n--- INTERPRETATION CHALLENGER FLAGS ---\n"
        "The automated Interpretation Challenger raised the following scientific concerns.\n"
        "You MUST address each flag in your review findings (confirm, refute, or note as inconclusive):\n"
        + flags_text
        + "\n--- END CHALLENGER FLAGS ---\n"
    )


def _today_ymd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _run_anthropic_job(*, ledger: Ledger, job_id: str) -> Dict[str, Any]:
    job = get_job(ledger, job_id)
    kind = job["kind"]
    spec = job["spec"]

    if kind not in {"review", "code_review"}:
        raise SystemExit(f"Unsupported anthropic job kind: {kind}")
    is_code_review = kind == "code_review"

    project_id = job.get("project_id") or spec.get("project_id")
    if not project_id:
        raise SystemExit("anthropic/review job requires job.project_id (or spec.project_id).")

    stage = str(spec.get("stage") or "analysis")
    mode = str(spec.get("mode") or "balanced")
    targets = spec.get("targets") or []
    questions = spec.get("questions") or []
    if not isinstance(targets, list) or not targets:
        raise SystemExit("anthropic/review job requires spec.targets (array of paths).")
    if not isinstance(questions, list) or not questions:
        questions = ["Any major issues? Any missing baselines/related work?"]

    project = get_project(ledger, str(project_id))
    req = write_review_request(
        ledger=ledger,
        project=project,
        stage=stage,
        mode=mode,
        targets=[str(x) for x in targets],
        questions=[str(x) for x in questions],
        rubric=spec.get("rubric"),
        time_budget_minutes=spec.get("time_budget_minutes"),
    )

    ws = Path(project["repo_path"]).resolve()
    include_contents = bool(spec.get("include_target_contents", True))
    max_chars = int(spec.get("max_target_chars", 60_000))

    target_blobs: List[str] = []
    if include_contents:
        for t in targets:
            p = (ws / str(t)).resolve()
            try:
                p.relative_to(ws)
            except ValueError:
                continue
            content = _safe_read_text(p, max_chars=max_chars)
            if not content:
                continue
            target_blobs.append(f"\n--- FILE: {t} ---\n{content}\n")

    schema_path = ledger.paths.root / "review" / "review_result.schema.json"
    schema_txt = _safe_read_text(schema_path, max_chars=60_000)

    request_json = _safe_read_text(Path(req["request_json_path"]), max_chars=60_000)
    research_ctx = _build_research_context_section(ws) if is_code_review else ""
    user_msg = (
        "Return ONLY a single JSON object that conforms to the following JSON Schema.\n"
        "Do not wrap in markdown or code fences.\n\n"
        + schema_txt
        + "\n\nReview request JSON:\n"
        + request_json
        + "\n\nTarget contents (may be truncated):\n"
        + "\n".join(target_blobs)
        + research_ctx
    )

    # Default to a current Sonnet model alias; override via spec["model"] or $ANTHROPIC_MODEL.
    model = str(spec.get("model") or os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-5")
    max_tokens = int(spec.get("max_tokens", 2048))
    temperature = spec.get("temperature")

    default_system_prompt_path = "prompts/reviewer_code.md" if is_code_review else None
    system_prompt_path = spec.get("system_prompt_file") or default_system_prompt_path
    system_prompt = ""
    if system_prompt_path:
        system_prompt = _safe_read_text((ledger.paths.root / str(system_prompt_path)).resolve(), max_chars=40_000).strip()
    if not system_prompt:
        system_prompt = (
            "You are a strict, detail-oriented pre-execution code reviewer."
            if is_code_review
            else "You are a strict, detail-oriented research reviewer."
        )

    client = AnthropicClient.from_env()
    resp = client.messages_create(
        model=model,
        system=system_prompt,
        user=user_msg,
        max_tokens=max_tokens,
        temperature=float(temperature) if temperature is not None else None,
    )
    text = anthropic_extract_text(resp)
    out = extract_json_object(text)

    reviewer = str(out.get("reviewer") or spec.get("reviewer") or "claude")
    out["project_id"] = str(project_id)
    out["stage"] = stage
    out["reviewer"] = reviewer

    reviews_dir = ws / "reviews"
    reviews_rel = "reviews"
    if is_code_review:
        reviews_dir = reviews_dir / "code"
        reviews_rel = "reviews/code"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    uid = uuid4().hex[:6]
    resp_path = reviews_dir / f"RESP-{stage}-{_today_ymd()}-{uid}-{reviewer}.json"
    resp_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ingested = ingest_review_result(
        ledger=ledger,
        result_path=resp_path,
        reviews_rel=reviews_rel,
        create_fix_tasks="major_blocker" if is_code_review else "none",
    )

    result_obj = {
        "provider_response": resp,
        "review_result": out,
        "ingested": {
            "stored_path": ingested.get("stored_path"),
            "review_id": ingested.get("review", {}).get("id"),
            "tasks_created": len(ingested.get("tasks_created") or []),
        },
    }
    with ledger.transaction():
        ledger.update_job(job_id=job_id, status="succeeded", result=result_obj, finished=True)
        ledger.insert_job_event(job_id=job_id, event_type="anthropic.review.completed", data={"reviewer": reviewer, "stage": stage})

    return get_job(ledger, job_id)


def _run_claude_code_cli_job(*, ledger: Ledger, job_id: str) -> Dict[str, Any]:
    job = get_job(ledger, job_id)
    kind = job["kind"]
    spec = job["spec"]
    provider_name = str(job.get("provider") or "claude_code_cli").strip() or "claude_code_cli"
    if provider_name not in {"claude_code_cli", "codex_cli"}:
        provider_name = "claude_code_cli"

    if kind not in {"review", "code_review"}:
        raise SystemExit(f"Unsupported {provider_name} job kind: {kind}")
    is_code_review = kind == "code_review"

    project_id = job.get("project_id") or spec.get("project_id")
    if not project_id:
        raise SystemExit(f"{provider_name}/{kind} job requires job.project_id (or spec.project_id).")

    stage = str(spec.get("stage") or "analysis")
    mode = str(spec.get("mode") or "balanced")
    targets = spec.get("targets") or []
    questions = spec.get("questions") or []
    if not isinstance(targets, list) or not targets:
        raise SystemExit(f"{provider_name}/{kind} job requires spec.targets (array of paths).")
    if not isinstance(questions, list) or not questions:
        questions = ["Any major issues? Any missing baselines/related work?"]

    project = get_project(ledger, str(project_id))
    req = write_review_request(
        ledger=ledger,
        project=project,
        stage=stage,
        mode=mode,
        targets=[str(x) for x in targets],
        questions=[str(x) for x in questions],
        rubric=spec.get("rubric"),
        time_budget_minutes=spec.get("time_budget_minutes"),
    )

    ws = Path(project["repo_path"]).resolve()
    include_contents = bool(spec.get("include_target_contents", True))
    max_chars = int(spec.get("max_target_chars", 60_000))

    target_pairs: List[Tuple[str, str]] = []
    if include_contents:
        for t in targets:
            p = (ws / str(t)).resolve()
            try:
                p.relative_to(ws)
            except ValueError:
                continue
            content = _safe_read_text(p, max_chars=max_chars)
            if not content:
                continue
            target_pairs.append((str(t), f"\n--- FILE: {t} ---\n{content}\n"))

    schema_path = ledger.paths.root / "review" / "review_result.schema.json"
    schema_raw = _safe_read_text(schema_path, max_chars=200_000)
    schema_obj = json.loads(schema_raw) if schema_raw.strip() else {"type": "object"}

    request_json = _safe_read_text(Path(req["request_json_path"]), max_chars=60_000)

    # If embedding all target contents would exceed a safe limit, prioritise
    # code/config files and let Claude read other files on demand via its Read tool.
    _MAX_TOTAL_TARGET_CHARS = int(spec.get("max_total_target_chars", 300_000))
    total_chars = sum(len(b) for _, b in target_pairs)
    if total_chars > _MAX_TOTAL_TARGET_CHARS:
        _CODE_EXTS = {".py", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".json", ".md", ".txt", ".ini", ".conf"}
        code_blobs: List[str] = []
        other_names: List[str] = []
        for tpath, blob in target_pairs:
            if Path(tpath).suffix.lower() in _CODE_EXTS:
                code_blobs.append(blob)
            else:
                other_names.append(tpath)
        code_total = sum(len(b) for b in code_blobs)
        if code_total > _MAX_TOTAL_TARGET_CHARS:
            kept: List[str] = []
            running = 0
            for b in code_blobs:
                if running + len(b) > _MAX_TOTAL_TARGET_CHARS:
                    break
                kept.append(b)
                running += len(b)
            code_blobs = kept
        target_section = (
            "Target contents (code/config files; may be truncated):\n"
            + "\n".join(code_blobs)
        )
        if other_names:
            target_section += (
                "\n\nOther changed files (data/binary — use the Read tool to inspect if needed):\n"
                + "\n".join(f"  - {n}" for n in other_names)
            )
    else:
        target_section = (
            "Target contents (may be truncated):\n"
            + "\n".join(b for _, b in target_pairs)
        )

    research_ctx = _build_research_context_section(ws) if is_code_review else ""
    challenger_section = _build_challenger_section(spec)

    user_msg = (
        "Return ONLY a single JSON object that conforms to the provided JSON Schema.\n"
        "Do not wrap in markdown or code fences.\n\n"
        "Review request JSON:\n"
        + request_json
        + "\n\n"
        + target_section
        + research_ctx
        + challenger_section
    )

    reviewer = str(spec.get("reviewer") or ("codex" if provider_name == "codex_cli" else "claude_code"))
    default_system_prompt_path = "prompts/reviewer_code.md" if is_code_review else "prompts/reviewer_research.md"
    system_prompt_path = spec.get("system_prompt_file") or default_system_prompt_path
    system_prompt = _safe_read_text((ledger.paths.root / str(system_prompt_path)).resolve(), max_chars=40_000).strip()
    if not system_prompt:
        system_prompt = (
            "You are a strict, detail-oriented pre-execution code reviewer."
            if is_code_review
            else "You are a strict, detail-oriented research reviewer."
        )

    validator = Draft202012Validator(schema_obj)
    model_for_event: Optional[str] = str(spec.get("model")) if spec.get("model") else None

    try:
        if provider_name == "codex_cli":
            config_overrides = spec.get("config_overrides", [])
            if not isinstance(config_overrides, list):
                raise SystemExit("codex_cli/review spec.config_overrides must be a list of strings.")
            cfg = CodexCliConfig(
                model=model_for_event,
                timeout_sec=int(spec.get("timeout_sec", 900)),
                sandbox=str(spec.get("sandbox") or "read-only"),
                config_overrides=[str(x) for x in config_overrides],
            )
            cli_json = run_codex_exec_print_json(
                prompt=f"{system_prompt}\n\n{user_msg}",
                json_schema=schema_obj,
                workspace_dir=ws,
                config=cfg,
            )
            out = codex_extract_structured_output(cli_json)
            provider_error_cls = CodexCliError
        else:
            cfg = ClaudeCodeCliConfig(
                model=model_for_event,
                timeout_sec=int(spec.get("timeout_sec", 900)),
                tools=str(spec.get("tools") or "Read"),
                allowed_tools=str(spec.get("allowed_tools") or "Read"),
                no_session_persistence=bool(spec.get("no_session_persistence", True)),
                permission_mode=str(spec.get("permission_mode")) if spec.get("permission_mode") else None,
            )
            cli_json = run_claude_code_print_json(
                prompt=user_msg,
                system_prompt=system_prompt,
                json_schema=schema_obj,
                workspace_dir=ws,
                config=cfg,
            )
            out = claude_extract_structured_output(cli_json)
            provider_error_cls = ClaudeCodeCliError
        if not isinstance(out, dict):
            raise provider_error_cls(f"{provider_name} structured_output was not a JSON object.")
        out["project_id"] = str(project_id)
        out["stage"] = stage
        out["reviewer"] = reviewer

        # Repair missing recommendation: infer from findings severity counts.
        _VALID_RECS = {"accept", "minor", "major", "reject"}
        if "recommendation" not in out or out.get("recommendation") not in _VALID_RECS:
            findings = out.get("findings") or []
            severities = [str(f.get("severity", "")).lower() for f in findings if isinstance(f, dict)]
            if any(s == "blocker" for s in severities):
                out["recommendation"] = "reject"
            elif any(s == "major" for s in severities):
                out["recommendation"] = "major"
            elif any(s == "minor" for s in severities):
                out["recommendation"] = "minor"
            else:
                out["recommendation"] = "accept"

        errors = sorted(validator.iter_errors(out), key=lambda e: e.json_path)
        if errors:
            raise provider_error_cls(f"{provider_name}/review output schema validation failed: {errors[0].message}")

        reviews_dir = ws / "reviews"
        reviews_rel = "reviews"
        if is_code_review:
            reviews_dir = reviews_dir / "code"
            reviews_rel = "reviews/code"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        uid = uuid4().hex[:6]
        resp_path = reviews_dir / f"RESP-{stage}-{_today_ymd()}-{uid}-{reviewer}.json"
        resp_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        ingested = ingest_review_result(
            ledger=ledger,
            result_path=resp_path,
            reviews_rel=reviews_rel,
            create_fix_tasks="major_blocker" if is_code_review else "none",
        )

        result_obj = {
            "provider_response": cli_json,
            "review_result": out,
            "ingested": {
                "stored_path": ingested.get("stored_path"),
                "review_id": ingested.get("review", {}).get("id"),
                "tasks_created": len(ingested.get("tasks_created") or []),
            },
        }
        with ledger.transaction():
            ledger.update_job(job_id=job_id, status="succeeded", result=result_obj, finished=True)
            ledger.insert_job_event(
                job_id=job_id,
                event_type=f"{provider_name}.review.completed",
                data={"reviewer": reviewer, "stage": stage, "model": model_for_event},
            )
        return get_job(ledger, job_id)
    except (ClaudeCodeCliError, CodexCliError) as e:
        ledger.insert_job_event(job_id=job_id, event_type=f"{provider_name}.review.error", data={"message": str(e)})

        fallback_provider = str(spec.get("fallback_provider") or "openai").strip()
        if fallback_provider.lower() in {"", "none", "false", "0"}:
            fallback_provider = ""
        if fallback_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            fallback_provider = ""
        if fallback_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            fallback_provider = ""

        if not fallback_provider:
            ledger.update_job(job_id=job_id, status="failed", error=str(e), finished=True)
            raise

        # Provider-specific keys that should not leak into fallback job specs.
        scrub_keys = {
            "model",
            "timeout_sec",
            "system_prompt_file",
            "tools",
            "allowed_tools",
            "no_session_persistence",
            "permission_mode",
            "sandbox",
            "config_overrides",
            "fallback_provider",
        }
        fb_spec = {k: v for k, v in spec.items() if k not in scrub_keys}
        fb_spec["reviewer"] = fallback_provider

        ledger.insert_job_event(
            job_id=job_id,
            event_type=f"{provider_name}.review.fallback",
            data={"fallback_provider": fallback_provider, "error": str(e)},
        )
        fb = create_job(ledger=ledger, project_id=str(project_id), provider=fallback_provider, kind=kind, spec=fb_spec)
        fb = run_job(ledger=ledger, job_id=fb["id"])

        result_obj = {
            "fallback_provider": fallback_provider,
            "fallback_job_id": fb["id"],
            "fallback_job_status": fb["status"],
            "fallback_job_result": fb.get("result"),
            "original_error": str(e),
        }
        status = "succeeded" if fb["status"] == "succeeded" else "failed"
        ledger.update_job(job_id=job_id, status=status, result=result_obj, error=str(e) if status == "failed" else None, finished=True)
        return get_job(ledger, job_id)
