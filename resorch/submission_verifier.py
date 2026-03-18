from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from resorch.manuscript_checker import check_manuscript_consistency, write_consistency_report
from resorch.utils import utc_now_iso
from resorch.verification_checklist import generate_verification_checklist, write_checklist


_PLACEHOLDER_ANY_RE = re.compile(r"{{[^{}]+}}")


def _safe_read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, str(exc)
    if not raw.strip():
        return None, "empty file"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "JSON root must be an object"
    return parsed, None


def _extract_claim_evidence_ids(text: str) -> List[str]:
    lines = text.splitlines()
    evidence_ids: List[str] = []
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("- evidence_ids:"):
            j = idx + 1
            while j < len(lines):
                cur = lines[j]
                if not cur.strip():
                    j += 1
                    continue
                if re.match(r"^\s{2,}-\s+", cur):
                    item = re.sub(r"^\s*-\s+", "", cur).strip()
                    if item and item != "(none)":
                        evidence_ids.append(item)
                    j += 1
                    continue
                break
    if evidence_ids:
        return evidence_ids

    singular = re.findall(r"^\s*-?\s*evidence_id\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return [s.strip() for s in singular if s.strip() and s.strip() != "(none)"]


def _parse_meta_json(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _add_check(checks: List[Dict[str, str]], name: str, status: str, detail: str) -> None:
    checks.append({"name": name, "status": status, "detail": detail})


def _add_attention(
    attention_map: List[Dict[str, str]],
    *,
    file_path: str,
    reason: str,
    location: str = "",
) -> None:
    attention_map.append({"file": file_path, "location": location, "reason": reason})


def _load_render_function(repo_root: Path):
    script_path = repo_root / "scripts" / "render_manuscript.py"
    if not script_path.exists():
        raise FileNotFoundError(f"render script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("render_manuscript_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load render script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, "render_manuscript", None)
    if not callable(func):
        raise RuntimeError("render_manuscript function is missing in render_manuscript.py")
    return func


def _check_scoreboard(scoreboard_path: Path) -> Tuple[str, str, List[str], List[Dict[str, str]]]:
    needs: List[str] = []
    attention: List[Dict[str, str]] = []

    scoreboard, err = _safe_read_json(scoreboard_path)
    rel = "results/scoreboard.json"
    if scoreboard is None:
        reason = f"Missing/invalid scoreboard at {rel}: {err}"
        attention.append({"file": rel, "location": "", "reason": reason})
        return "fail", reason, needs, attention

    pm = scoreboard.get("primary_metric")
    if not isinstance(pm, dict):
        reason = "scoreboard.primary_metric is missing or not an object"
        attention.append({"file": rel, "location": "primary_metric", "reason": reason})
        return "fail", reason, needs, attention

    required = ["name", "direction", "current"]
    missing = [k for k in required if k not in pm]
    if missing:
        reason = f"Missing required primary_metric keys: {', '.join(missing)}"
        attention.append({"file": rel, "location": "primary_metric", "reason": reason})
        return "fail", reason, needs, attention

    if "baseline" not in pm:
        msg = "Missing required primary_metric key: baseline"
        needs.append(msg)
        attention.append({"file": rel, "location": "primary_metric.baseline", "reason": msg})

    nullish = [k for k in [*required, "baseline"] if k in pm and (pm.get(k) is None or (k == "name" and str(pm.get("name") or "").strip() == ""))]
    if nullish:
        reason = f"primary_metric contains null/empty required value(s): {', '.join(nullish)}"
        attention.append({"file": rel, "location": "primary_metric", "reason": reason})
        return "fail", reason, needs, attention

    direction = str(pm.get("direction") or "").strip().lower()
    if direction not in {"maximize", "minimize"}:
        reason = f"primary_metric.direction must be maximize/minimize (got: {pm.get('direction')})"
        attention.append({"file": rel, "location": "primary_metric.direction", "reason": reason})
        return "fail", reason, needs, attention

    cur = pm.get("current")
    n_runs = None
    has_ci = False
    if isinstance(cur, dict):
        raw_n = cur.get("n_runs")
        if isinstance(raw_n, (int, float)) and not isinstance(raw_n, bool):
            n_runs = int(raw_n)
        elif isinstance(raw_n, str):
            try:
                n_runs = int(float(raw_n))
            except ValueError:
                n_runs = None
        has_ci = any(k in cur for k in ("ci", "ci_95", "ci95", "confidence_interval"))
    if not has_ci:
        has_ci = any(
            isinstance(pm.get(k), (int, float, str)) and not isinstance(pm.get(k), bool)
            for k in ("ci", "ci_95", "ci95", "confidence_interval")
        )
    if not has_ci:
        metrics = scoreboard.get("metrics")
        metric_sources = metrics if isinstance(metrics, dict) else {}
        has_ci = any(
            key in metric_sources or key in scoreboard
            for key in (
                "primary_metric_ci_95",
                "primary_metric_ci95",
                "primary_metric_ci",
                "primary_metric_confidence_interval",
                "ci_95",
                "ci95",
                "ci",
                "confidence_interval",
            )
        )

    if n_runs is None:
        runs = scoreboard.get("runs")
        if isinstance(runs, list):
            n_runs = len(runs)
    if n_runs is None:
        for key in ("run_count", "n_runs"):
            raw_n = pm.get(key)
            if isinstance(raw_n, (int, float)) and not isinstance(raw_n, bool):
                n_runs = int(raw_n)
                break
            if isinstance(raw_n, str):
                try:
                    n_runs = int(float(raw_n))
                    break
                except ValueError:
                    continue

    if n_runs is None:
        msg = "n_runs could not be determined from primary_metric.current, primary_metric aliases, or runs[]"
        needs.append(msg)
        attention.append({"file": rel, "location": "primary_metric.current", "reason": msg})
    elif n_runs < 3:
        msg = f"n_runs={n_runs} is below recommended minimum (3)"
        needs.append(msg)
        attention.append({"file": rel, "location": "primary_metric.current.n_runs", "reason": msg})

    if n_runs is not None and not has_ci:
        msg = "CI field is missing in primary_metric.current or scoreboard aliases (expected ci_95/ci95/ci/confidence_interval)"
        needs.append(msg)
        attention.append({"file": rel, "location": "primary_metric.current", "reason": msg})

    detail = "scoreboard primary_metric keys are present"
    if needs:
        detail += "; " + "; ".join(needs)
        return "needs_human", detail, needs, attention
    return "pass", detail, needs, attention


def _check_claims(workspace: Path) -> Tuple[str, str, List[Dict[str, str]]]:
    claims_dir = workspace / "claims"
    if not claims_dir.exists():
        return "pass", "claims/ directory not found (skipped)", []

    claim_files = sorted(claims_dir.glob("*.md"))
    if not claim_files:
        return "pass", "No claim markdown files found", []

    missing: List[str] = []
    attention: List[Dict[str, str]] = []
    for cp in claim_files:
        try:
            text = cp.read_text(encoding="utf-8")
        except OSError:
            missing.append(cp.relative_to(workspace).as_posix())
            attention.append(
                {
                    "file": cp.relative_to(workspace).as_posix(),
                    "location": "",
                    "reason": "Could not read claim file",
                }
            )
            continue
        ids = _extract_claim_evidence_ids(text)
        if not ids:
            rel = cp.relative_to(workspace).as_posix()
            missing.append(rel)
            attention.append(
                {
                    "file": rel,
                    "location": "evidence_ids",
                    "reason": "Claim is missing populated evidence_id/evidence_ids",
                }
            )

    if missing:
        detail = f"{len(missing)}/{len(claim_files)} claim file(s) missing evidence IDs: {', '.join(missing[:5])}"
        return "fail", detail, attention
    return "pass", f"All {len(claim_files)} claim file(s) include evidence IDs", attention


def _run_compile_check(workspace: Path, repo_root: Path) -> Tuple[str, str, Optional[Path], str]:
    script_path = repo_root / "scripts" / "compile_paper.py"
    log_path = workspace / "results" / "submission_compile.log"

    if not script_path.exists():
        detail = f"compile script not found (skipped): {script_path}"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(detail + "\n", encoding="utf-8")
        return "needs_human", detail, log_path, detail

    cmd = [sys.executable, str(script_path), str(workspace)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as exc:
        timeout_stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        timeout_stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        log_text = "\n".join(
            [
                f"command: {' '.join(cmd)}",
                "return_code: timeout",
                "timeout_seconds: 120",
                "",
                "--- stdout ---",
                timeout_stdout,
                "",
                "--- stderr ---",
                timeout_stderr,
            ]
        ).strip() + "\n"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(log_text, encoding="utf-8")
        detail = "compile_paper.py timed out after 120 seconds"
        return "fail", detail, log_path, log_text
    log_text = "\n".join(
        [
            f"command: {' '.join(cmd)}",
            f"return_code: {proc.returncode}",
            "",
            "--- stdout ---",
            proc.stdout,
            "",
            "--- stderr ---",
            proc.stderr,
        ]
    ).strip() + "\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text, encoding="utf-8")

    if proc.returncode == 0:
        return "pass", "compile_paper.py completed successfully", log_path, log_text
    return "fail", f"compile_paper.py failed with exit code {proc.returncode}", log_path, log_text


def _write_report(
    workspace: Path,
    *,
    project_id: str,
    mode: str,
    generated_at: str,
    verdict: str,
    checks: List[Dict[str, str]],
    attention_map: List[Dict[str, str]],
    needs_human_items: List[str],
) -> Path:
    report_path = workspace / "results" / "submission_verification_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fail_details = [c["detail"] for c in checks if c["status"] == "fail"]

    lines: List[str] = []
    lines.append("# Submission Verification Report")
    lines.append("## Verdict")
    lines.append(f"- project_id: `{project_id}`")
    lines.append(f"- mode: `{mode}`")
    lines.append(f"- generated_at: `{generated_at}`")
    lines.append(f"- verdict: `{verdict}`")
    lines.append("")

    lines.append("## Failure Reasons")
    if fail_details:
        for idx, detail in enumerate(fail_details, start=1):
            lines.append(f"{idx}. {detail}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Needs Human")
    if needs_human_items:
        for idx, item in enumerate(needs_human_items, start=1):
            lines.append(f"{idx}. {item}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Attention Map")
    if attention_map:
        lines.append("| file | location | reason |")
        lines.append("|---|---|---|")
        for item in attention_map:
            file_path = str(item.get("file") or "")
            location = str(item.get("location") or "")
            reason = str(item.get("reason") or "")
            lines.append(f"| {file_path} | {location} | {reason} |")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Checks")
    if not checks:
        lines.append("- (none)")
    else:
        for chk in checks:
            lines.append(f"- [{chk['status'].upper()}] {chk['name']}: {chk['detail']}")

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def _create_submission_bundle(
    workspace: Path,
    *,
    artifacts_list_path: Path,
) -> Path:
    zip_path = workspace / "paper" / "submission_bundle.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    include_candidates = [
        workspace / "paper" / "manuscript.md",
        workspace / "paper" / "output" / "manuscript.pdf",
        workspace / "results" / "scoreboard.json",
        workspace / "results" / "manuscript_consistency_report.md",
        workspace / "reviews" / "verification_checklist.md",
        workspace / "results" / "submission_verification_report.md",
        workspace / "results" / "submission_verification.json",
        workspace / "results" / "submission_compile.log",
        artifacts_list_path,
    ]

    claims_dir = workspace / "claims"
    if claims_dir.exists():
        include_candidates.extend(sorted(claims_dir.glob("*.md")))

    evidence_dir = workspace / "evidence"
    if evidence_dir.exists():
        include_candidates.extend(sorted(evidence_dir.glob("*.json")))

    added: set[str] = set()
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in include_candidates:
            if not path.exists() or not path.is_file():
                continue
            rel = path.resolve().relative_to(workspace).as_posix()
            if rel in added:
                continue
            zf.write(path, arcname=rel)
            added.add(rel)

    return zip_path


def verify_submission(ledger: Any, project_id: str, mode: str = "quick") -> Dict[str, Any]:
    mode_norm = str(mode or "quick").strip().lower()
    if mode_norm not in {"quick", "full"}:
        raise ValueError("mode must be 'quick' or 'full'")

    project = ledger.get_project(project_id)
    workspace = Path(project["repo_path"]).resolve()
    (workspace / "results").mkdir(parents=True, exist_ok=True)
    (workspace / "paper").mkdir(parents=True, exist_ok=True)
    (workspace / "reviews").mkdir(parents=True, exist_ok=True)

    generated_at = utc_now_iso()
    checks: List[Dict[str, str]] = []
    attention_map: List[Dict[str, str]] = []
    needs_human_items: List[str] = []

    # 1) manuscript consistency
    try:
        consistency = check_manuscript_consistency(workspace)
        consistency_out = write_consistency_report(workspace, consistency)
        if consistency.failed_checks > 0:
            _add_check(
                checks,
                "manuscript_consistency",
                "fail",
                f"{consistency.failed_checks} consistency check(s) failed. See {consistency_out.relative_to(workspace).as_posix()}.",
            )
            for failed in [c for c in consistency.checks if not c.passed][:10]:
                _add_attention(
                    attention_map,
                    file_path="paper/manuscript.md",
                    location=failed.location or failed.check_id,
                    reason=failed.message,
                )
        else:
            _add_check(checks, "manuscript_consistency", "pass", "All manuscript consistency checks passed")
    except Exception as exc:  # noqa: BLE001
        _add_check(checks, "manuscript_consistency", "fail", f"Failed to run manuscript checker: {exc}")
        _add_attention(
            attention_map,
            file_path="paper/manuscript.md",
            reason=f"manuscript checker error: {exc}",
        )

    # 2) verification checklist
    try:
        checklist = generate_verification_checklist(workspace_dir=workspace, project_id=str(project_id), include_manuscript_checks=True)
        checklist_out = write_checklist(workspace, checklist)
        if checklist.fail_count > 0:
            _add_check(
                checks,
                "verification_checklist",
                "fail",
                f"Checklist reported {checklist.fail_count} fail item(s). See {checklist_out.relative_to(workspace).as_posix()}.",
            )
        elif checklist.needs_human_count > 0:
            _add_check(
                checks,
                "verification_checklist",
                "needs_human",
                f"Checklist has {checklist.needs_human_count} needs_human item(s). See {checklist_out.relative_to(workspace).as_posix()}.",
            )
        else:
            _add_check(checks, "verification_checklist", "pass", "Verification checklist passed without fail/needs_human")

        for item in checklist.items:
            if item.auto_status == "needs_human":
                needs_human_items.append(f"{item.id}: {item.auto_evidence}")
            if item.auto_status in {"fail", "needs_human"}:
                _add_attention(
                    attention_map,
                    file_path="reviews/verification_checklist.md",
                    location=item.id,
                    reason=item.auto_evidence,
                )
    except Exception as exc:  # noqa: BLE001
        _add_check(checks, "verification_checklist", "fail", f"Failed to generate checklist: {exc}")
        _add_attention(
            attention_map,
            file_path="reviews/verification_checklist.md",
            reason=f"checklist generation error: {exc}",
        )

    # 3) compile paper (best effort if script exists)
    repo_root = Path(ledger.paths.root).resolve()
    compile_status, compile_detail, compile_log_path, _ = _run_compile_check(workspace, repo_root)
    _add_check(checks, "compile_paper", compile_status, compile_detail)
    if compile_status != "pass":
        _add_attention(attention_map, file_path="paper/manuscript.md", reason=compile_detail)
        if compile_status == "needs_human":
            needs_human_items.append(compile_detail)

    # 4) scoreboard required keys
    sb_status, sb_detail, sb_needs, sb_attention = _check_scoreboard(workspace / "results" / "scoreboard.json")
    _add_check(checks, "scoreboard_required_fields", sb_status, sb_detail)
    attention_map.extend(sb_attention)
    needs_human_items.extend(sb_needs)

    # 5) claim evidence IDs
    claims_status, claims_detail, claims_attention = _check_claims(workspace)
    _add_check(checks, "claims_evidence_ids", claims_status, claims_detail)
    attention_map.extend(claims_attention)

    # 6) full mode template rendering
    template_path = workspace / "paper" / "manuscript.template.md"
    if mode_norm == "full" and template_path.exists():
        try:
            render_fn = _load_render_function(repo_root)
            rendered = render_fn(workspace)
            unresolved = bool(_PLACEHOLDER_ANY_RE.search(rendered))
            if unresolved:
                raise ValueError("unresolved placeholders remained after rendering")
            _add_check(checks, "render_manuscript", "pass", "manuscript.template.md rendered successfully")
        except Exception as exc:  # noqa: BLE001
            _add_check(checks, "render_manuscript", "fail", f"render_manuscript failed: {exc}")
            _add_attention(
                attention_map,
                file_path="paper/manuscript.template.md",
                reason=f"template rendering failed: {exc}",
            )
    elif mode_norm == "full":
        _add_check(checks, "render_manuscript", "pass", "Template not present; render step skipped")

    # artifacts list for bundle
    artifacts_json_path = workspace / "results" / "submission_artifacts.json"
    artifacts_rows = []
    try:
        raw_rows = ledger.list_artifacts(str(project_id), prefix=None, limit=500)
        for row in raw_rows:
            item = dict(row)
            item["meta"] = _parse_meta_json(item.get("meta_json"))
            artifacts_rows.append(item)
    except Exception as exc:  # noqa: BLE001
        artifacts_rows = [{"error": f"Failed to load artifacts from ledger: {exc}"}]

    artifacts_json_path.write_text(json.dumps({"artifacts": artifacts_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    has_fail = any(c["status"] == "fail" for c in checks)
    has_needs = any(c["status"] == "needs_human" for c in checks) or bool(needs_human_items)
    if has_fail:
        verdict = "fail"
    elif has_needs:
        verdict = "needs_human"
    else:
        verdict = "pass"

    # de-duplicate needs_human and attention items
    if needs_human_items:
        dedup_needs: List[str] = []
        seen_needs: set[str] = set()
        for item in needs_human_items:
            if item in seen_needs:
                continue
            seen_needs.add(item)
            dedup_needs.append(item)
        needs_human_items = dedup_needs

    if attention_map:
        dedup_attention: List[Dict[str, str]] = []
        seen_attention: set[Tuple[str, str, str]] = set()
        for item in attention_map:
            t = (str(item.get("file") or ""), str(item.get("location") or ""), str(item.get("reason") or ""))
            if t in seen_attention:
                continue
            seen_attention.add(t)
            dedup_attention.append({"file": t[0], "location": t[1], "reason": t[2]})
        attention_map = dedup_attention

    report_path = _write_report(
        workspace,
        project_id=str(project_id),
        mode=mode_norm,
        generated_at=generated_at,
        verdict=verdict,
        checks=checks,
        attention_map=attention_map,
        needs_human_items=needs_human_items,
    )

    result: Dict[str, Any] = {
        "project_id": str(project_id),
        "mode": mode_norm,
        "generated_at": generated_at,
        "verdict": verdict,
        "checks": checks,
        "attention_map": attention_map,
        "needs_human_items": needs_human_items,
        "report_path": report_path.relative_to(workspace).as_posix(),
        "json_path": "results/submission_verification.json",
        "bundle_path": "paper/submission_bundle.zip",
    }
    if compile_log_path is not None and compile_log_path.exists():
        result["compile_log_path"] = compile_log_path.relative_to(workspace).as_posix()

    json_out_path = workspace / "results" / "submission_verification.json"
    json_out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bundle_path = _create_submission_bundle(workspace, artifacts_list_path=artifacts_json_path)
    result["bundle_path"] = bundle_path.relative_to(workspace).as_posix()

    # Refresh JSON to capture the final bundle path.
    json_out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return result


__all__ = ["verify_submission"]
