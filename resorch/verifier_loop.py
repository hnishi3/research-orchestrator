from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from resorch.utils import utc_now_iso

log = logging.getLogger(__name__)


def _append_check(checks: List[Dict[str, str]], *, name: str, status: str, detail: str) -> None:
    checks.append({"name": str(name), "status": str(status), "detail": str(detail)})


def _dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _safe_read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, str(exc)
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON ({exc})"
    if not isinstance(parsed, dict):
        return None, "scoreboard.json must be a JSON object"
    return parsed, None


def _call_generate_verification_checklist(
    *,
    workspace: Path,
    ledger: Any,
    project_id: str,
):
    from resorch.verification_checklist import generate_verification_checklist

    sig = inspect.signature(generate_verification_checklist)
    params = sig.parameters
    lightweight_enabled = "lightweight" in params

    kwargs: Dict[str, Any] = {"include_manuscript_checks": True}
    if lightweight_enabled:
        kwargs["lightweight"] = True
    if "project_id" in params:
        kwargs["project_id"] = str(project_id)

    attempts = []
    if "workspace_dir" in params:
        attempts.append(lambda: generate_verification_checklist(workspace_dir=workspace, **kwargs))
    elif "workspace" in params:
        attempts.append(lambda: generate_verification_checklist(workspace=workspace, **kwargs))
    else:
        attempts.append(lambda: generate_verification_checklist(workspace, **kwargs))

    if ledger is not None:
        attempts.append(
            lambda: generate_verification_checklist(
                ledger,
                str(project_id),
                include_manuscript_checks=True,
                **({"lightweight": True} if lightweight_enabled else {}),
            )
        )

    last_type_error: Optional[TypeError] = None
    for attempt in attempts:
        try:
            return attempt(), lightweight_enabled
        except TypeError as exc:
            last_type_error = exc
            continue
    if last_type_error is not None:
        raise last_type_error
    raise RuntimeError("Failed to call generate_verification_checklist()")


def _render_markdown(result: Dict[str, Any]) -> str:
    checks = result.get("checks") if isinstance(result.get("checks"), list) else []
    failed_checks = [c for c in checks if isinstance(c, dict) and c.get("status") == "fail"]
    needs_human_items = result.get("needs_human_items") if isinstance(result.get("needs_human_items"), list) else []

    lines: List[str] = [
        "# Post-Step Verification",
        f"- Verdict: {result.get('verdict')}",
        f"- Timestamp: {result.get('timestamp')}",
        "## Failed Checks",
    ]
    if failed_checks:
        for chk in failed_checks:
            lines.append(f"- {chk.get('name')}: {chk.get('detail')}")
    else:
        lines.append("- (none)")
    lines.append("## Needs Human")
    if needs_human_items:
        for item in needs_human_items:
            lines.append(f"- {item}")
    else:
        lines.append("- (none)")
    return "\n".join(lines) + "\n"


def _write_outputs(workspace: Path, result: Dict[str, Any]) -> None:
    out_dir = workspace / "notes" / "autopilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verifier_last.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "verifier_last.md").write_text(_render_markdown(result), encoding="utf-8")


def run_post_step_verification(
    workspace: Path,
    ledger: Any = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    workspace_path = Path(workspace).resolve()
    checks: List[Dict[str, str]] = []
    fail_items: List[str] = []
    needs_human_items: List[str] = []

    # Required workspace files.
    required_files = ["notes/problem.md", "results/scoreboard.json"]
    for rel in required_files:
        p = workspace_path / rel
        if p.exists():
            _append_check(checks, name=f"required_file:{rel}", status="pass", detail="exists")
        else:
            _append_check(checks, name=f"required_file:{rel}", status="fail", detail="missing required file")
            fail_items.append(f"Create required file: {rel}")

    # Scoreboard primary_metric schema check.
    scoreboard_path = workspace_path / "results" / "scoreboard.json"
    if not scoreboard_path.exists():
        _append_check(
            checks,
            name="scoreboard_primary_metric",
            status="fail",
            detail="results/scoreboard.json not found",
        )
        fail_items.append("Create results/scoreboard.json with primary_metric.name/direction/current.")
    else:
        scoreboard_obj, scoreboard_err = _safe_read_json(scoreboard_path)
        if scoreboard_obj is None:
            _append_check(
                checks,
                name="scoreboard_primary_metric",
                status="fail",
                detail=f"results/scoreboard.json {scoreboard_err}",
            )
            fail_items.append("Fix results/scoreboard.json so it is valid JSON object.")
        else:
            pm = scoreboard_obj.get("primary_metric")
            if not isinstance(pm, dict):
                _append_check(
                    checks,
                    name="scoreboard_primary_metric",
                    status="fail",
                    detail="missing primary_metric object",
                )
                fail_items.append("Add primary_metric object to results/scoreboard.json.")
            else:
                missing = [k for k in ("name", "direction", "current") if k not in pm]
                if missing:
                    _append_check(
                        checks,
                        name="scoreboard_primary_metric",
                        status="fail",
                        detail=f"missing key(s): {', '.join(missing)}",
                    )
                    fail_items.append(
                        "Add primary_metric keys in results/scoreboard.json: name, direction, current."
                    )
                else:
                    _append_check(
                        checks,
                        name="scoreboard_primary_metric",
                        status="pass",
                        detail="primary_metric has required keys",
                    )

    # Manuscript consistency (optional dependency + optional file).
    manuscript_path = workspace_path / "paper" / "manuscript.md"
    if manuscript_path.exists():
        try:
            from resorch.manuscript_checker import check_manuscript_consistency
        except ImportError as exc:
            _append_check(
                checks,
                name="manuscript_consistency",
                status="needs_human",
                detail=f"manuscript_checker unavailable ({exc})",
            )
            needs_human_items.append("manuscript_checker unavailable; manuscript consistency checks were skipped.")
        else:
            try:
                report = check_manuscript_consistency(workspace_path)
                failed = [
                    c for c in getattr(report, "checks", [])
                    if (not bool(getattr(c, "passed", False))) and bool(getattr(c, "applicable", True))
                ]
                if failed:
                    _append_check(
                        checks,
                        name="manuscript_consistency",
                        status="fail",
                        detail=f"{len(failed)} consistency check(s) failed",
                    )
                    for chk in failed:
                        cid = str(getattr(chk, "check_id", "check"))
                        msg = str(getattr(chk, "message", "")).strip()
                        fail_items.append(f"manuscript_consistency/{cid}: {msg or 'failed'}")
                else:
                    _append_check(
                        checks,
                        name="manuscript_consistency",
                        status="pass",
                        detail="all applicable manuscript checks passed",
                    )
            except Exception as exc:  # noqa: BLE001
                _append_check(
                    checks,
                    name="manuscript_consistency",
                    status="needs_human",
                    detail=f"manuscript consistency failed to execute ({exc})",
                )
                needs_human_items.append(f"manuscript consistency execution error: {exc}")
    else:
        _append_check(
            checks,
            name="manuscript_consistency",
            status="pass",
            detail="paper/manuscript.md not found; check skipped",
        )

    # Verification checklist (optional dependency, only when context provided).
    if ledger is not None and project_id is not None:
        try:
            checklist, lightweight_enabled = _call_generate_verification_checklist(
                workspace=workspace_path,
                ledger=ledger,
                project_id=str(project_id),
            )
            items = getattr(checklist, "items", [])
            if not isinstance(items, list):
                items = []
            fail_count = 0
            needs_human_count = 0
            for item in items:
                if isinstance(item, dict):
                    item_id = str(item.get("id") or "item")
                    status = str(item.get("auto_status") or "").strip().lower()
                    evidence = str(item.get("auto_evidence") or item.get("question") or "").strip()
                else:
                    item_id = str(getattr(item, "id", "item"))
                    status = str(getattr(item, "auto_status", "")).strip().lower()
                    evidence = str(getattr(item, "auto_evidence", "") or getattr(item, "question", "")).strip()
                if status == "fail":
                    fail_count += 1
                    fail_items.append(f"verification_checklist/{item_id}: {evidence or 'failed'}")
                elif status == "needs_human":
                    needs_human_count += 1
                    needs_human_items.append(f"verification_checklist/{item_id}: {evidence or 'needs human review'}")
            if fail_count > 0:
                status = "fail"
            elif needs_human_count > 0:
                status = "needs_human"
            else:
                status = "pass"
            mode_txt = "enabled" if lightweight_enabled else "fallback"
            _append_check(
                checks,
                name="verification_checklist",
                status=status,
                detail=f"lightweight={mode_txt}, fail={fail_count}, needs_human={needs_human_count}",
            )
        except ImportError as exc:
            _append_check(
                checks,
                name="verification_checklist",
                status="needs_human",
                detail=f"verification_checklist unavailable ({exc})",
            )
            needs_human_items.append("verification_checklist unavailable; checklist verification skipped.")
        except Exception as exc:  # noqa: BLE001
            _append_check(
                checks,
                name="verification_checklist",
                status="needs_human",
                detail=f"verification checklist failed to execute ({exc})",
            )
            needs_human_items.append(f"verification checklist execution error: {exc}")
    else:
        _append_check(
            checks,
            name="verification_checklist",
            status="pass",
            detail="ledger/project_id not provided; check skipped",
        )

    fail_items = _dedupe(fail_items)
    needs_human_items = _dedupe(needs_human_items)

    verdict = "pass"
    if fail_items:
        verdict = "fail"
    elif needs_human_items:
        verdict = "needs_human"

    result: Dict[str, Any] = {
        "verdict": verdict,
        "checks": checks,
        "fail_items": fail_items,
        "needs_human_items": needs_human_items,
        "timestamp": utc_now_iso(),
    }

    try:
        _write_outputs(workspace_path, result)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to write verifier outputs: %s", exc)
        _append_check(
            checks,
            name="verifier_output_files",
            status="needs_human",
            detail=f"failed to write verifier_last outputs ({exc})",
        )
        needs_human_items = _dedupe(needs_human_items + [f"failed to write verifier outputs: {exc}"])
        if not fail_items:
            verdict = "needs_human"
        result = {
            "verdict": verdict,
            "checks": checks,
            "fail_items": fail_items,
            "needs_human_items": needs_human_items,
            "timestamp": result["timestamp"],
        }

    return result


__all__ = ["run_post_step_verification"]
