from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from resorch.evidence_store import validate_evidence_url
from resorch.ledger import Ledger
from resorch.manuscript_checker import check_manuscript_consistency
from resorch.paths import RepoPaths, find_repo_root
from resorch.utils import utc_now_iso


@dataclass
class ChecklistItem:
    id: str
    category: str
    question: str
    auto_status: str
    auto_evidence: str
    human_verified: bool = False
    human_notes: str = ""


@dataclass
class VerificationChecklist:
    project_id: str
    generated_at: str
    items: List[ChecklistItem]
    auto_pass_count: int
    needs_human_count: int
    fail_count: int
    coverage: float
    summary: str


_PLACEHOLDER_RE = re.compile(r"\b(?:TODO|FIXME|TBD)\b", re.IGNORECASE)
_EVIDENCE_ID_RE = re.compile(r"\b[0-9a-f]{32}\b", re.IGNORECASE)
_P_VALUE_RE = re.compile(r"\bp\s*(?:<=|>=|=|<|>)\s*(?:[0-9]*\.[0-9]+|[0-9]+)\b", re.IGNORECASE)
_EFFECT_SIZE_RE = re.compile(
    r"\b(?:effect size|cohen'?s?\s*d|eta\s*(?:squared|\^?2)|odds ratio|hazard ratio)\b"
    r"|(?:\bd\s*=|\br\s*=|\bor\s*=|\bhr\s*=|\bbeta\s*=|eta2|eta\^?2)",
    re.IGNORECASE,
)


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        return _as_float(value.get("mean"))
    return None


def _metric_value(scoreboard: Dict[str, Any], key: str) -> Optional[float]:
    metrics = scoreboard.get("metrics")
    if isinstance(metrics, dict):
        v = _as_float(metrics.get(key))
        if v is not None:
            return v
    return _as_float(scoreboard.get(key))


def _primary_metric(scoreboard: Dict[str, Any]) -> Dict[str, Any]:
    pm = scoreboard.get("primary_metric")
    return pm if isinstance(pm, dict) else {}


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
    return _EVIDENCE_ID_RE.findall(text)


def _collect_evidence_items(workspace: Path, ledger_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()

    evidence_dir = workspace / "evidence"
    files = sorted(evidence_dir.glob("*.json")) if evidence_dir.exists() else []
    for ep in files:
        rel = ep.resolve().relative_to(workspace).as_posix()
        try:
            payload = json.loads(_safe_read_text(ep))
        except json.JSONDecodeError:
            items.append({"source": rel, "url": "", "error": "invalid_json"})
            continue
        rows: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            rows = [payload]
        elif isinstance(payload, list):
            rows = [r for r in payload if isinstance(r, dict)]
        if not rows:
            items.append({"source": rel, "url": "", "error": "no_rows"})
            continue
        for idx, row in enumerate(rows, start=1):
            url = str(row.get("url") or "").strip()
            key = f"{rel}#{idx}:{url}"
            if key in seen:
                continue
            seen.add(key)
            items.append({"source": f"{rel}#{idx}", "url": url})

    for row in ledger_rows:
        url = str(row.get("url") or "").strip()
        rid = str(row.get("id") or "row")
        key = f"ledger:{rid}:{url}"
        if key in seen:
            continue
        seen.add(key)
        items.append({"source": f"ledger:{rid}", "url": url})

    return items


def _check_url(url: str, timeout: float = 4.0) -> Dict[str, Any]:
    req = Request(url, method="HEAD", headers={"User-Agent": "resorch-verifier/1.0"})
    try:
        with urlopen(req, timeout=timeout) as response:
            status_code = getattr(response, "status", None) or response.getcode()
            return {"ok": int(status_code) < 400, "status_code": int(status_code)}
    except HTTPError as exc:
        return {"ok": int(exc.code) < 400, "status_code": int(exc.code)}
    except URLError as exc:
        return {"ok": False, "error": str(exc), "network_error": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "network_error": True}


def _looks_like_data_sources_section(method_text: str) -> bool:
    lower = method_text.lower()
    if "data source" in lower or "dataset" in lower:
        return True
    return bool(re.search(r"^\s*##+\s+.*data.*$", method_text, flags=re.IGNORECASE | re.MULTILINE))


def _looks_like_preprocessing_section(method_text: str) -> bool:
    lower = method_text.lower()
    keywords = (
        "preprocess",
        "pre-processing",
        "normalization",
        "tokenization",
        "cleaning",
        "filtering",
    )
    if any(k in lower for k in keywords):
        return True
    return bool(re.search(r"^\s*##+\s+.*preprocess.*$", method_text, flags=re.IGNORECASE | re.MULTILINE))


def _resolve_manuscript(workspace: Path) -> Path:
    default = workspace / "paper" / "manuscript.md"
    if default.exists():
        return default
    paper_dir = workspace / "paper"
    if paper_dir.exists():
        md_files = sorted(paper_dir.glob("*.md"))
        if md_files:
            return md_files[0]
    return default


def _load_ledger_context(workspace: Path, project_id: str) -> Dict[str, Any]:
    root = find_repo_root(workspace)
    if root is None:
        return {}
    db_path = root / ".orchestrator" / "ledger.db"
    if not db_path.exists():
        return {}

    ledger = Ledger(RepoPaths(root=root))
    try:
        ledger.init()
        pid = str(project_id or "").strip()
        if not pid:
            for p in ledger.list_projects():
                repo_path = str(p.get("repo_path") or "").strip()
                if not repo_path:
                    continue
                try:
                    if Path(repo_path).resolve() == workspace:
                        pid = str(p["id"])
                        break
                except OSError:
                    continue
        if not pid:
            return {}
        artifacts = ledger.list_artifacts(pid, prefix=None, limit=500)
        evidence_rows = ledger.list_evidence(project_id=pid, idea_id=None, limit=500)
        return {"project_id": pid, "artifacts": artifacts, "evidence_rows": evidence_rows}
    except Exception:  # noqa: BLE001
        return {}
    finally:
        ledger.close()


def _find_metric_keyword_hits(method_text: str, scoreboard: Dict[str, Any]) -> int:
    lower = method_text.lower()
    names: List[str] = []
    pm = _primary_metric(scoreboard)
    pm_name = str(pm.get("name") or "").strip()
    if pm_name:
        names.append(pm_name)
    metrics = scoreboard.get("metrics")
    if isinstance(metrics, dict):
        names.extend(str(k) for k in metrics.keys())

    hits = 0
    for name in names:
        n = name.lower().strip()
        if not n:
            continue
        if n in lower or n.replace("_", " ") in lower:
            hits += 1
    return hits


def _consistency_check_index(workspace: Path) -> Dict[str, Any]:
    report = check_manuscript_consistency(workspace)
    out: Dict[str, Any] = {"report": report, "checks": {}}
    for chk in report.checks:
        out["checks"][chk.check_id] = chk
    return out


def generate_verification_checklist(
    workspace_dir: Path,
    project_id: str = "",
    include_manuscript_checks: bool = True,
    lightweight: bool = False,
) -> VerificationChecklist:
    workspace = Path(workspace_dir).resolve()
    generated_at = utc_now_iso()

    checklist_items: List[ChecklistItem] = []

    def add_item(item_id: str, category: str, question: str, status: str, evidence: str) -> None:
        checklist_items.append(
            ChecklistItem(
                id=item_id,
                category=category,
                question=question,
                auto_status=status,
                auto_evidence=evidence,
            )
        )

    scoreboard = _safe_read_json(workspace / "results" / "scoreboard.json")
    method_path = workspace / "notes" / "method.md"
    method_text = _safe_read_text(method_path)
    manuscript_path = _resolve_manuscript(workspace)
    manuscript_text = _safe_read_text(manuscript_path) if manuscript_path.exists() else ""
    pm = _primary_metric(scoreboard)

    ledger_ctx = _load_ledger_context(workspace, project_id=project_id)
    ledger_evidence_rows = ledger_ctx.get("evidence_rows") if isinstance(ledger_ctx, dict) else []
    if not isinstance(ledger_evidence_rows, list):
        ledger_evidence_rows = []

    consistency: Dict[str, Any] = {}
    if include_manuscript_checks:
        consistency = _consistency_check_index(workspace)
    consistency_checks = consistency.get("checks") if isinstance(consistency, dict) else {}
    if not isinstance(consistency_checks, dict):
        consistency_checks = {}

    current_v = _as_float(pm.get("current"))
    baseline_v = _as_float(pm.get("baseline"))
    direction = str(pm.get("direction") or "maximize").strip().lower()

    # 1. metric_baseline_stated
    if baseline_v is None:
        add_item(
            "metric_baseline_stated",
            "metrics",
            "Is the baseline metric value documented?",
            "fail",
            "Baseline is missing from results/scoreboard.json primary_metric.baseline.",
        )
    else:
        add_item(
            "metric_baseline_stated",
            "metrics",
            "Is the baseline metric value documented?",
            "pass",
            f"scoreboard.json primary_metric.baseline={baseline_v:g}.",
        )

    # 2. metric_current_vs_baseline
    if current_v is None or baseline_v is None:
        add_item(
            "metric_current_vs_baseline",
            "metrics",
            "Does the current metric show improvement over baseline?",
            "fail",
            "Cannot compare current vs baseline: one or both values are missing in scoreboard.json.",
        )
    elif direction not in {"maximize", "minimize"}:
        add_item(
            "metric_current_vs_baseline",
            "metrics",
            "Does the current metric show improvement over baseline?",
            "needs_human",
            f"Direction is '{direction}' (expected maximize/minimize); current={current_v:g}, baseline={baseline_v:g}.",
        )
    else:
        improved = current_v > baseline_v if direction == "maximize" else current_v < baseline_v
        status = "pass" if improved else "fail"
        add_item(
            "metric_current_vs_baseline",
            "metrics",
            "Does the current metric show improvement over baseline?",
            status,
            f"direction={direction}, current={current_v:g}, baseline={baseline_v:g}.",
        )

    # 3. metric_reproducible
    cur_obj = pm.get("current")
    n_runs: Optional[int] = None
    has_ci = False
    if isinstance(cur_obj, dict):
        raw_n = cur_obj.get("n_runs")
        if isinstance(raw_n, bool):
            raw_n = None
        if isinstance(raw_n, (int, float)):
            n_runs = int(raw_n)
        elif isinstance(raw_n, str):
            try:
                n_runs = int(float(raw_n))
            except ValueError:
                n_runs = None
        has_ci = any(k in cur_obj for k in ("ci_95", "ci95", "ci", "confidence_interval"))
    if n_runs is None:
        runs = scoreboard.get("runs")
        if isinstance(runs, list):
            n_runs = len(runs)
    if n_runs is None:
        add_item(
            "metric_reproducible",
            "metrics",
            "Were metrics computed from \u22653 runs with CI?",
            "needs_human",
            "Could not determine n_runs from scoreboard primary_metric.current.n_runs or runs[].",
        )
    elif n_runs < 3:
        add_item(
            "metric_reproducible",
            "metrics",
            "Were metrics computed from \u22653 runs with CI?",
            "fail",
            f"Only {n_runs} run(s) recorded; expected at least 3 with confidence interval.",
        )
    elif not has_ci:
        add_item(
            "metric_reproducible",
            "metrics",
            "Were metrics computed from \u22653 runs with CI?",
            "fail",
            f"n_runs={n_runs}, but no CI field found (expected ci_95/ci95/ci/confidence_interval).",
        )
    else:
        add_item(
            "metric_reproducible",
            "metrics",
            "Were metrics computed from \u22653 runs with CI?",
            "pass",
            f"n_runs={n_runs} with CI field present in primary_metric.current.",
        )

    # 4. metric_defined_operationally
    method_lower = method_text.lower()
    has_metric_kw = "metric" in method_lower
    has_def_kw = any(k in method_lower for k in ("definition", "defined", "direction", "baseline"))
    metric_hits = _find_metric_keyword_hits(method_text, scoreboard) if method_text else 0
    if not method_text.strip():
        add_item(
            "metric_defined_operationally",
            "metrics",
            "Are all metrics defined in method.md?",
            "fail",
            f"Missing or empty method file: {method_path.as_posix()}",
        )
    elif has_metric_kw and has_def_kw and (metric_hits > 0 or "metric definitions" in method_lower):
        add_item(
            "metric_defined_operationally",
            "metrics",
            "Are all metrics defined in method.md?",
            "pass",
            "method.md includes metric definitions/direction language and metric-name matches.",
        )
    else:
        add_item(
            "metric_defined_operationally",
            "metrics",
            "Are all metrics defined in method.md?",
            "fail",
            "method.md does not clearly define metric names, definitions, and direction.",
        )

    # 5. claims_have_evidence
    claims_dir = workspace / "claims"
    claim_files = sorted(claims_dir.glob("*.md")) if claims_dir.exists() else []
    if not claim_files:
        add_item(
            "claims_have_evidence",
            "claims",
            "Do all claims reference evidence?",
            "not_applicable",
            "No claim markdown files found in claims/.",
        )
    else:
        missing_claims: List[str] = []
        for cp in claim_files:
            ids = _extract_claim_evidence_ids(_safe_read_text(cp))
            if not ids:
                missing_claims.append(cp.resolve().relative_to(workspace).as_posix())
        if missing_claims:
            add_item(
                "claims_have_evidence",
                "claims",
                "Do all claims reference evidence?",
                "fail",
                f"{len(missing_claims)}/{len(claim_files)} claim file(s) missing evidence IDs: {', '.join(missing_claims[:5])}.",
            )
        else:
            add_item(
                "claims_have_evidence",
                "claims",
                "Do all claims reference evidence?",
                "pass",
                f"All {len(claim_files)} claim file(s) include evidence IDs.",
            )

    # 6. claims_evidence_accessible
    evidence_items = _collect_evidence_items(workspace, ledger_rows=ledger_evidence_rows)
    if not evidence_items:
        add_item(
            "claims_evidence_accessible",
            "claims",
            "Are evidence URLs reachable?",
            "not_applicable",
            "No evidence records found in evidence/ or ledger.",
        )
    elif lightweight:
        add_item(
            "claims_evidence_accessible",
            "claims",
            "Are evidence URLs reachable?",
            "needs_human",
            f"Lightweight mode: skipped network URL probes for {len(evidence_items)} evidence item(s).",
        )
    else:
        total_checked = 0
        reachable = 0
        invalid = 0
        hard_fail = 0
        uncertain = 0
        max_urls = 12
        for item in evidence_items[:max_urls]:
            url = str(item.get("url") or "").strip()
            if not url:
                invalid += 1
                continue
            fmt = validate_evidence_url(url)
            if not fmt.get("valid"):
                invalid += 1
                continue
            total_checked += 1
            probe = _check_url(url)
            if probe.get("ok"):
                reachable += 1
                continue
            if probe.get("network_error"):
                uncertain += 1
            else:
                hard_fail += 1
        skipped = max(0, len(evidence_items) - max_urls)
        evidence_msg = (
            f"checked={total_checked}, reachable={reachable}, invalid={invalid}, "
            f"unreachable={hard_fail}, uncertain={uncertain}, skipped={skipped}."
        )
        if invalid > 0 or hard_fail > 0:
            add_item(
                "claims_evidence_accessible",
                "claims",
                "Are evidence URLs reachable?",
                "fail",
                evidence_msg,
            )
        elif uncertain > 0 or skipped > 0:
            add_item(
                "claims_evidence_accessible",
                "claims",
                "Are evidence URLs reachable?",
                "needs_human",
                evidence_msg,
            )
        else:
            add_item(
                "claims_evidence_accessible",
                "claims",
                "Are evidence URLs reachable?",
                "pass",
                evidence_msg,
            )

    # 7. code_tests_pass
    test_fail_count = None
    for key in ("test_fail_count", "tests_failed"):
        test_fail_count = _metric_value(scoreboard, key)
        if test_fail_count is not None:
            break
    test_pass_count = None
    for key in ("test_pass_count", "tests_passed"):
        test_pass_count = _metric_value(scoreboard, key)
        if test_pass_count is not None:
            break
    if test_fail_count is None:
        add_item(
            "code_tests_pass",
            "code",
            "Do all tests pass?",
            "needs_human",
            "scoreboard missing test_fail_count/tests_failed metric.",
        )
    elif test_fail_count <= 0:
        detail = f"test_fail_count={test_fail_count:g}"
        if test_pass_count is not None:
            detail += f", test_pass_count={test_pass_count:g}"
        add_item("code_tests_pass", "code", "Do all tests pass?", "pass", detail + ".")
    else:
        add_item(
            "code_tests_pass",
            "code",
            "Do all tests pass?",
            "fail",
            f"scoreboard reports test_fail_count={test_fail_count:g}.",
        )

    # 8. code_no_regressions
    baseline_pass = None
    for key in ("test_pass_count_baseline", "baseline_test_pass_count"):
        baseline_pass = _metric_value(scoreboard, key)
        if baseline_pass is not None:
            break
    if baseline_pass is None:
        first_run_pass: Optional[float] = None
        runs = scoreboard.get("runs")
        if isinstance(runs, list) and runs:
            for run in runs:
                if not isinstance(run, dict):
                    continue
                run_metrics = run.get("metrics")
                if isinstance(run_metrics, dict):
                    first_run_pass = _as_float(run_metrics.get("test_pass_count"))
                if first_run_pass is None:
                    first_run_pass = _as_float(run.get("test_pass_count"))
                if first_run_pass is not None:
                    break
        baseline_pass = first_run_pass
    if test_pass_count is None or baseline_pass is None:
        add_item(
            "code_no_regressions",
            "code",
            "Is test_pass_count >= baseline?",
            "needs_human",
            "Could not determine current/baseline test_pass_count from scoreboard metrics.",
        )
    elif test_pass_count >= baseline_pass:
        add_item(
            "code_no_regressions",
            "code",
            "Is test_pass_count >= baseline?",
            "pass",
            f"current={test_pass_count:g} >= baseline={baseline_pass:g}.",
        )
    else:
        add_item(
            "code_no_regressions",
            "code",
            "Is test_pass_count >= baseline?",
            "fail",
            f"current={test_pass_count:g} < baseline={baseline_pass:g}.",
        )

    # 9. code_in_version_control
    src_git = workspace / "src" / ".git"
    root_git = workspace / ".git"
    if src_git.exists():
        add_item(
            "code_in_version_control",
            "code",
            "Does .git exist in src/?",
            "pass",
            "Found src/.git.",
        )
    elif root_git.exists():
        add_item(
            "code_in_version_control",
            "code",
            "Does .git exist in src/?",
            "pass",
            "Found workspace-level .git (repository root); src/.git is not present.",
        )
    else:
        add_item(
            "code_in_version_control",
            "code",
            "Does .git exist in src/?",
            "fail",
            "No .git directory found in src/ or workspace root.",
        )

    # 10. data_sources_documented
    if not method_text.strip():
        add_item(
            "data_sources_documented",
            "data",
            "Are data sources listed in method.md?",
            "fail",
            f"Missing or empty method file: {method_path.as_posix()}",
        )
    elif _looks_like_data_sources_section(method_text):
        add_item(
            "data_sources_documented",
            "data",
            "Are data sources listed in method.md?",
            "needs_human",
            "method.md includes data-source/dataset language; verify completeness and provenance manually.",
        )
    else:
        add_item(
            "data_sources_documented",
            "data",
            "Are data sources listed in method.md?",
            "fail",
            "No clear data sources/datasets section found in method.md.",
        )

    # 11. data_preprocessing_described
    if not method_text.strip():
        add_item(
            "data_preprocessing_described",
            "data",
            "Is data preprocessing described?",
            "fail",
            f"Missing or empty method file: {method_path.as_posix()}",
        )
    elif _looks_like_preprocessing_section(method_text):
        add_item(
            "data_preprocessing_described",
            "data",
            "Is data preprocessing described?",
            "needs_human",
            "Preprocessing keywords/section detected in method.md; verify adequacy and reproducibility manually.",
        )
    else:
        add_item(
            "data_preprocessing_described",
            "data",
            "Is data preprocessing described?",
            "fail",
            "No clear preprocessing description detected in method.md.",
        )

    # 12. stats_tests_appropriate
    if manuscript_text.strip():
        detected = sorted(
            {
                k
                for k in (
                    "t-test" if re.search(r"\bt[-\s]?test\b", manuscript_text, re.IGNORECASE) else "",
                    "anova" if re.search(r"\banova\b", manuscript_text, re.IGNORECASE) else "",
                    "wilcoxon" if re.search(r"\bwilcoxon\b", manuscript_text, re.IGNORECASE) else "",
                    "mann-whitney" if re.search(r"\bmann[-\s]?whitney\b", manuscript_text, re.IGNORECASE) else "",
                    "chi-square" if re.search(r"\bchi[-\s]?square\b", manuscript_text, re.IGNORECASE) else "",
                )
                if k
            }
        )
        if detected:
            evidence = f"Detected statistical test terms: {', '.join(detected)}. Human judgment required for appropriateness."
        else:
            evidence = "No explicit statistical-test terms detected; human review required."
        add_item(
            "stats_tests_appropriate",
            "statistics",
            "Are statistical tests appropriate for the data?",
            "needs_human",
            evidence,
        )
    else:
        add_item(
            "stats_tests_appropriate",
            "statistics",
            "Are statistical tests appropriate for the data?",
            "fail",
            f"Manuscript not found or empty: {manuscript_path.as_posix()}",
        )

    # 13. stats_effect_sizes_reported
    stats_effect_chk = consistency_checks.get("stats_have_effect_sizes")
    if stats_effect_chk is not None:
        status = "pass" if bool(getattr(stats_effect_chk, "passed", False)) else "fail"
        add_item(
            "stats_effect_sizes_reported",
            "statistics",
            "Are effect sizes reported alongside p-values?",
            status,
            str(getattr(stats_effect_chk, "message", "")),
        )
    elif not manuscript_text.strip():
        add_item(
            "stats_effect_sizes_reported",
            "statistics",
            "Are effect sizes reported alongside p-values?",
            "fail",
            f"Manuscript not found or empty: {manuscript_path.as_posix()}",
        )
    else:
        has_p = bool(_P_VALUE_RE.search(manuscript_text))
        has_effect = bool(_EFFECT_SIZE_RE.search(manuscript_text))
        if not has_p:
            add_item(
                "stats_effect_sizes_reported",
                "statistics",
                "Are effect sizes reported alongside p-values?",
                "not_applicable",
                "No p-values detected in manuscript text.",
            )
        elif has_effect:
            add_item(
                "stats_effect_sizes_reported",
                "statistics",
                "Are effect sizes reported alongside p-values?",
                "pass",
                "Detected both p-values and effect-size indicators in manuscript text.",
            )
        else:
            add_item(
                "stats_effect_sizes_reported",
                "statistics",
                "Are effect sizes reported alongside p-values?",
                "fail",
                "Detected p-values but no effect-size indicators in manuscript text.",
            )

    # 14. stats_multiple_comparisons
    if manuscript_text.strip():
        has_correction = bool(
            re.search(
                r"\b(bonferroni|holm|hochberg|benjamini[-\s]?hochberg|fdr|false discovery rate)\b",
                manuscript_text,
                re.IGNORECASE,
            )
        )
        evidence = (
            "Multiple-comparisons correction keywords detected; verify that correction matches test family."
            if has_correction
            else "No explicit correction keywords detected; verify whether correction is required."
        )
        add_item(
            "stats_multiple_comparisons",
            "statistics",
            "Is multiple comparison correction applied where needed?",
            "needs_human",
            evidence,
        )
    else:
        add_item(
            "stats_multiple_comparisons",
            "statistics",
            "Is multiple comparison correction applied where needed?",
            "fail",
            f"Manuscript not found or empty: {manuscript_path.as_posix()}",
        )

    # 15. manuscript_figures_consistent
    if include_manuscript_checks:
        fig_ref = consistency_checks.get("fig_ref_exists")
        fig_file = consistency_checks.get("fig_file_referenced")
        if fig_ref is None:
            add_item(
                "manuscript_figures_consistent",
                "manuscript",
                "Do all figure references resolve?",
                "fail",
                "Consistency report missing fig_ref_exists check.",
            )
        else:
            passed = bool(getattr(fig_ref, "passed", False))
            detail = str(getattr(fig_ref, "message", ""))
            if fig_file is not None and not bool(getattr(fig_file, "passed", False)):
                passed = False
                detail = f"{detail} {str(getattr(fig_file, 'message', ''))}".strip()
            add_item(
                "manuscript_figures_consistent",
                "manuscript",
                "Do all figure references resolve?",
                "pass" if passed else "fail",
                detail or "Figure consistency check completed.",
            )
    else:
        add_item(
            "manuscript_figures_consistent",
            "manuscript",
            "Do all figure references resolve?",
            "not_applicable",
            "Manuscript checks disabled by include_manuscript_checks=False.",
        )

    # 16. manuscript_no_placeholders
    if include_manuscript_checks:
        if not manuscript_text.strip():
            add_item(
                "manuscript_no_placeholders",
                "manuscript",
                "No TODO/FIXME in text?",
                "fail",
                f"Manuscript not found or empty: {manuscript_path.as_posix()}",
            )
        else:
            matches = _PLACEHOLDER_RE.findall(manuscript_text)
            if matches:
                add_item(
                    "manuscript_no_placeholders",
                    "manuscript",
                    "No TODO/FIXME in text?",
                    "fail",
                    f"Found placeholder tokens: {', '.join(matches[:5])}.",
                )
            else:
                add_item(
                    "manuscript_no_placeholders",
                    "manuscript",
                    "No TODO/FIXME in text?",
                    "pass",
                    "No TODO/FIXME/TBD tokens found in manuscript text.",
                )
    else:
        add_item(
            "manuscript_no_placeholders",
            "manuscript",
            "No TODO/FIXME in text?",
            "not_applicable",
            "Manuscript checks disabled by include_manuscript_checks=False.",
        )

    # 17. manuscript_refs_have_dois
    if include_manuscript_checks:
        doi_chk = consistency_checks.get("refs_have_dois")
        if doi_chk is None:
            add_item(
                "manuscript_refs_have_dois",
                "manuscript",
                "Do references have DOIs?",
                "fail",
                "Consistency report missing refs_have_dois check.",
            )
        else:
            add_item(
                "manuscript_refs_have_dois",
                "manuscript",
                "Do references have DOIs?",
                "pass" if bool(getattr(doi_chk, "passed", False)) else "fail",
                str(getattr(doi_chk, "message", "")),
            )
    else:
        add_item(
            "manuscript_refs_have_dois",
            "manuscript",
            "Do references have DOIs?",
            "not_applicable",
            "Manuscript checks disabled by include_manuscript_checks=False.",
        )

    total = len(checklist_items)
    auto_pass_count = sum(1 for i in checklist_items if i.auto_status == "pass")
    needs_human_count = sum(1 for i in checklist_items if i.auto_status == "needs_human")
    fail_count = sum(1 for i in checklist_items if i.auto_status == "fail")
    coverage = float(auto_pass_count) / float(total) if total else 1.0
    summary = (
        f"Auto-verified {auto_pass_count}/{total}; "
        f"needs human review {needs_human_count}; failed {fail_count}."
    )

    effective_project_id = str(project_id).strip()
    if not effective_project_id and isinstance(ledger_ctx, dict):
        effective_project_id = str(ledger_ctx.get("project_id") or "").strip()
    if not effective_project_id:
        effective_project_id = workspace.name

    return VerificationChecklist(
        project_id=effective_project_id,
        generated_at=generated_at,
        items=checklist_items,
        auto_pass_count=auto_pass_count,
        needs_human_count=needs_human_count,
        fail_count=fail_count,
        coverage=coverage,
        summary=summary,
    )


def format_checklist_markdown(checklist: VerificationChecklist) -> str:
    lines: List[str] = []
    total = len(checklist.items)
    lines.append("# Verification Checklist")
    lines.append("## Summary")
    lines.append(
        f"- Auto-verified: {checklist.auto_pass_count}/{total} | "
        f"Needs human review: {checklist.needs_human_count} | Failed: {checklist.fail_count}"
    )
    lines.append(f"- Coverage: {checklist.coverage * 100:.1f}%")
    lines.append(f"- Project: `{checklist.project_id}`")
    lines.append(f"- Generated: `{checklist.generated_at}`")
    lines.append("")

    category_order = ["metrics", "claims", "code", "data", "statistics", "manuscript"]
    category_title = {
        "metrics": "Metrics",
        "claims": "Claims",
        "code": "Code",
        "data": "Data",
        "statistics": "Statistics",
        "manuscript": "Manuscript",
    }

    status_label = {
        "pass": "PASS",
        "fail": "FAIL",
        "needs_human": "NEEDS HUMAN",
        "not_applicable": "N/A",
    }
    checks = checklist.items
    by_category: Dict[str, List[ChecklistItem]] = {}
    for item in checks:
        by_category.setdefault(item.category, []).append(item)

    for category in category_order:
        items = by_category.get(category)
        if not items:
            continue
        lines.append(f"## {category_title.get(category, category.title())}")
        for item in items:
            checked = "x" if item.auto_status == "pass" else " "
            lines.append(f"- [{checked}] {item.id}: {item.question}")
            lines.append(f"  - Auto: {status_label.get(item.auto_status, item.auto_status)} - {item.auto_evidence}")
            if item.human_verified or item.human_notes:
                verified = "YES" if item.human_verified else "NO"
                lines.append(f"  - Human verified: {verified}")
            if item.human_notes:
                lines.append(f"  - Human notes: {item.human_notes}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_checklist(
    workspace_dir: Path,
    checklist: VerificationChecklist,
    output_path: Optional[Path] = None,
) -> Path:
    workspace = Path(workspace_dir).resolve()
    out = output_path or (workspace / "reviews" / "verification_checklist.md")
    out = out if out.is_absolute() else (workspace / out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_checklist_markdown(checklist), encoding="utf-8")
    return out


__all__ = [
    "ChecklistItem",
    "VerificationChecklist",
    "generate_verification_checklist",
    "format_checklist_markdown",
    "write_checklist",
]
