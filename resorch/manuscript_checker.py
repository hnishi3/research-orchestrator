from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from resorch.evidence_store import validate_evidence_url
from resorch.paths import resolve_within_workspace


FIG_REF_RE = re.compile(r"\b(?:Figure|Fig\.?)\s*(\d+)\b", re.IGNORECASE)
TABLE_REF_RE = re.compile(r"\bTable\s+(\d+)\b", re.IGNORECASE)
MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
FIG_NUM_IN_TEXT_RE = re.compile(r"\b(?:Figure|Fig\.?)\s*(\d+)\b", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^\s*(?:#{2,6}\s+)?(?:\*\*)?\s*Table\s+(\d+)\b", re.IGNORECASE)
INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
P_VALUE_ANY_RE = re.compile(r"\bp\s*(<=|>=|=|<|>)\s*([0-9]*\.[0-9]+|[0-9]+)\b", re.IGNORECASE)
PERCENT_CLAIM_RE = re.compile(
    r"\b(improved?|increased?|reduced?|decreased?)\b(?:\s+\w+){0,3}?\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
PERCENT_NOUN_CLAIM_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*%\s*(improvement|increase|reduction|decrease)\b",
    re.IGNORECASE,
)
SIGNIFICANCE_CLAIM_RE = re.compile(
    r"\b(?:statistically\s+significant|significantly\s+different|significant\s+(?:result|improvement)|significantly)\b",
    re.IGNORECASE,
)
DOI_RE = re.compile(r"\b(?:doi:\s*|https?://(?:dx\.)?doi\.org/)\s*10\.\d{4,9}/\S+", re.IGNORECASE)
REF_PLACEHOLDER_RE = re.compile(r"\[\?\]|\b(?:TODO|FIXME|TBD)\b", re.IGNORECASE)
EFFECT_SIZE_RE = re.compile(
    r"\b(?:effect size|cohen'?s?\s*d|eta\s*(?:squared|\^?2)|odds ratio|hazard ratio)\b"
    r"|(?:\bd\s*=|\br\s*=|\bor\s*=|\bhr\s*=|\bbeta\s*=|β\s*=|η²|η2)",
    re.IGNORECASE,
)


@dataclass
class CheckResult:
    check_id: str
    category: str
    severity: str
    passed: bool
    message: str
    location: Optional[str] = None
    applicable: bool = True


@dataclass
class ConsistencyReport:
    checks: List[CheckResult] = field(default_factory=list)
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    applicable_checks: int = 0
    not_applicable_checks: int = 0
    consistency_score: float = 0.0
    effective_score: float = 0.0
    categories: Dict[str, Dict[str, int]] = field(default_factory=dict)
    summary: str = ""


def _strip_heading_number(text: str) -> str:
    return re.sub(r"^\s*\d+(?:\.\d+)*\.?\s+", "", text).strip()


def _resolve_manuscript_path(workspace_dir: Path, manuscript_path: Optional[Path]) -> Path:
    if manuscript_path is not None:
        p = Path(manuscript_path)
        return p if p.is_absolute() else (workspace_dir / p)
    default = workspace_dir / "paper" / "manuscript.md"
    if default.exists():
        return default
    paper_dir = workspace_dir / "paper"
    if paper_dir.exists():
        md_files = sorted(paper_dir.glob("*.md"))
        if md_files:
            return md_files[0]
    return default


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _extract_heading(lines: Sequence[str], prefix: str = "##") -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    pattern = re.compile(rf"^\s*{re.escape(prefix)}+\s+(.+?)\s*$")
    for idx, line in enumerate(lines):
        m = pattern.match(line)
        if not m:
            continue
        out.append((idx, _strip_heading_number(m.group(1))))
    return out


def _extract_references(lines: Sequence[str]) -> Tuple[List[str], str]:
    start_idx: Optional[int] = None
    for idx, heading in _extract_heading(lines, prefix="##"):
        if heading.lower() == "references":
            start_idx = idx + 1
            break
    if start_idx is None:
        return [], ""

    end_idx = len(lines)
    for idx in range(start_idx, len(lines)):
        if re.match(r"^\s*##+\s+", lines[idx]):
            end_idx = idx
            break

    section_lines = list(lines[start_idx:end_idx])
    entries: List[str] = []
    for raw in section_lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("<!--") and line.endswith("-->"):
            continue
        if re.match(r"^\d+\.\s+", line):
            entries.append(re.sub(r"^\d+\.\s+", "", line).strip())
            continue
        if entries:
            entries[-1] = entries[-1] + " " + line
        else:
            entries.append(line)
    return entries, "\n".join(section_lines)


def _find_section_bounds(lines: Sequence[str], section_name: str) -> Optional[Tuple[int, int]]:
    headings = _extract_heading(lines, prefix="##")
    target = section_name.strip().lower()
    for idx, (line_idx, heading) in enumerate(headings):
        if heading.strip().lower() != target:
            continue
        start_idx = line_idx + 1
        end_idx = headings[idx + 1][0] if (idx + 1) < len(headings) else len(lines)
        return start_idx, end_idx
    return None


def _extract_inline_images(manuscript_text: str) -> List[Tuple[str, str]]:
    return [(m.group(1).strip(), m.group(2).strip()) for m in MD_IMAGE_RE.finditer(manuscript_text)]


def _extract_figure_numbers_from_name(name: str) -> List[int]:
    nums: List[int] = []
    patterns = [
        re.compile(r"\bfig(?:ure)?[_\-\s]?(\d+)\b", re.IGNORECASE),
        re.compile(r"^(\d+)\b"),
    ]
    for pat in patterns:
        for m in pat.finditer(name):
            try:
                nums.append(int(m.group(1)))
            except ValueError:
                continue
    return nums


def _extract_table_numbers(lines: Sequence[str]) -> Tuple[List[int], int]:
    explicit_nums: List[int] = []
    table_blocks = 0
    pending_caption_num: Optional[int] = None

    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        cap = TABLE_CAPTION_RE.match(stripped)
        if cap:
            try:
                pending_caption_num = int(cap.group(1))
            except ValueError:
                pending_caption_num = None

        if stripped.startswith("|"):
            table_blocks += 1
            if pending_caption_num is not None:
                explicit_nums.append(pending_caption_num)
            pending_caption_num = None
            idx += 1
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                idx += 1
            continue

        idx += 1

    return explicit_nums, table_blocks


def _sequential_gap(nums: Iterable[int]) -> List[int]:
    uniq = sorted({n for n in nums if n > 0})
    if not uniq:
        return []
    expected = set(range(1, max(uniq) + 1))
    return sorted(expected - set(uniq))


def _extract_paragraphs_with_context(manuscript_text: str) -> List[Dict[str, Any]]:
    lines = manuscript_text.splitlines()
    section = "Document"
    per_section_count: Dict[str, int] = {}
    paragraphs: List[Dict[str, Any]] = []
    current: List[str] = []
    current_start_line = 1
    in_code = False

    def flush() -> None:
        nonlocal current, current_start_line
        text = " ".join(x.strip() for x in current if x.strip()).strip()
        current = []
        if not text:
            return
        per_section_count[section] = per_section_count.get(section, 0) + 1
        paragraphs.append(
            {
                "section": section,
                "paragraph": per_section_count[section],
                "text": text,
                "line": current_start_line,
            }
        )

    for idx, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if stripped.startswith("```"):
            if in_code:
                in_code = False
            else:
                flush()
                in_code = True
            continue
        if in_code:
            continue
        hm = re.match(r"^\s*##+\s+(.+?)\s*$", raw)
        if hm:
            flush()
            section = _strip_heading_number(hm.group(1)) or "Section"
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        if not stripped:
            flush()
            continue
        if not current:
            current_start_line = idx
        current.append(raw)
    flush()
    return paragraphs


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        return _to_float(value.get("mean"))
    return None


def _flatten_scoreboard_metrics(scoreboard: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    metrics = scoreboard.get("metrics")
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            fv = _to_float(v)
            if fv is not None:
                out[str(k)] = fv

    pm = scoreboard.get("primary_metric")
    if isinstance(pm, dict):
        pm_name = str(pm.get("name") or "").strip()
        current_v = _to_float(pm.get("current"))
        best_v = _to_float(pm.get("best"))
        baseline_v = _to_float(pm.get("baseline"))
        delta_v = _to_float(pm.get("delta_vs_baseline"))
        if pm_name and current_v is not None:
            out[pm_name] = current_v
        if current_v is not None:
            out["primary_metric.current"] = current_v
        if best_v is not None:
            out["primary_metric.best"] = best_v
        if baseline_v is not None:
            out["primary_metric.baseline"] = baseline_v
        if delta_v is not None:
            out["primary_metric.delta_vs_baseline"] = delta_v
    return out


def _metric_name_variants(metric_name: str) -> List[str]:
    base = metric_name.strip().lower()
    if not base:
        return []
    variants = {
        base,
        base.replace("_", " "),
        base.replace(".", " "),
    }
    if base == "test_pass_count":
        variants.update({"tests pass", "tests passed", "test pass count"})
    return sorted({v for v in variants if v})


def _regex_from_metric_variant(variant: str) -> str:
    escaped = re.escape(variant)
    escaped = escaped.replace(r"\ ", r"\s+")
    return escaped


def _line_from_pos(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _normalize_change_concept(token: str) -> str:
    t = token.strip().lower()
    if t.startswith("improv"):
        return "improved"
    if t.startswith("increas"):
        return "increased"
    if t.startswith("reduc"):
        return "reduced"
    if t.startswith("decreas"):
        return "decreased"
    return t


def _extract_percent_claims(text: str) -> Dict[str, set[float]]:
    claims: Dict[str, set[float]] = {}
    for m in PERCENT_CLAIM_RE.finditer(text):
        concept = _normalize_change_concept(m.group(1))
        try:
            value = float(m.group(2))
        except ValueError:
            continue
        claims.setdefault(concept, set()).add(value)
    for m in PERCENT_NOUN_CLAIM_RE.finditer(text):
        noun = m.group(2).lower()
        if noun.startswith("improv"):
            concept = "improved"
        elif noun.startswith("increas"):
            concept = "increased"
        elif noun.startswith("reduct"):
            concept = "reduced"
        else:
            concept = "decreased"
        try:
            value = float(m.group(1))
        except ValueError:
            continue
        claims.setdefault(concept, set()).add(value)
    return claims


def _is_explicitly_gt_005(op: str, value: float) -> bool:
    if op == "=":
        return value > 0.05
    if op == ">":
        return value >= 0.05
    if op == ">=":
        return value > 0.05
    return False


def _build_report(checks: List[CheckResult]) -> ConsistencyReport:
    total = len(checks)
    passed = sum(1 for c in checks if c.passed)
    failed = total - passed
    score = (float(passed) / float(total)) if total else 1.0
    applicable_checks = sum(1 for c in checks if c.applicable)
    not_applicable_checks = sum(1 for c in checks if not c.applicable)
    passed_applicable = sum(1 for c in checks if c.applicable and c.passed)
    effective_score = (float(passed_applicable) / float(applicable_checks)) if applicable_checks else 1.0

    categories: Dict[str, Dict[str, int]] = {}
    for chk in checks:
        bucket = categories.setdefault(chk.category, {"passed": 0, "failed": 0})
        if chk.passed:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1

    error_count = sum(1 for c in checks if (not c.passed) and c.severity == "error")
    warning_count = sum(1 for c in checks if (not c.passed) and c.severity == "warning")
    info_count = sum(1 for c in checks if (not c.passed) and c.severity == "info")
    summary = (
        f"{passed}/{total} checks passed ({score * 100:.1f}%). "
        f"{failed} issues found ({error_count} error, {warning_count} warning, {info_count} info)."
    )

    return ConsistencyReport(
        checks=checks,
        total_checks=total,
        passed_checks=passed,
        failed_checks=failed,
        applicable_checks=applicable_checks,
        not_applicable_checks=not_applicable_checks,
        consistency_score=score,
        effective_score=effective_score,
        categories=categories,
        summary=summary,
    )


def check_manuscript_consistency(
    workspace_dir: Path,
    manuscript_path: Optional[Path] = None,
) -> ConsistencyReport:
    workspace = Path(workspace_dir).resolve()
    manuscript = _resolve_manuscript_path(workspace, manuscript_path).resolve()
    manuscript_exists = manuscript.exists() and manuscript.is_file()
    manuscript_text = _safe_read_text(manuscript) if manuscript_exists else ""
    manuscript_lines = manuscript_text.splitlines()

    checks: List[CheckResult] = []

    def add(
        check_id: str,
        category: str,
        severity: str,
        passed: bool,
        message: str,
        location: Optional[str] = None,
        applicable: bool = True,
    ) -> None:
        checks.append(
            CheckResult(
                check_id=check_id,
                category=category,
                severity=severity,
                passed=passed,
                message=message,
                location=location,
                applicable=applicable,
            )
        )

    # Shared figure/table parsing.
    fig_refs = sorted({int(x) for x in FIG_REF_RE.findall(manuscript_text)}) if manuscript_text else []
    table_refs = sorted({int(x) for x in TABLE_REF_RE.findall(manuscript_text)}) if manuscript_text else []
    inline_images = _extract_inline_images(manuscript_text) if manuscript_text else []

    fig_dir = workspace / "results" / "fig"
    figure_files = []
    if fig_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg", "*.pdf"):
            figure_files.extend(sorted(fig_dir.glob(ext)))
    figure_files = [p for p in figure_files if p.is_file()]

    inline_fig_nums = []
    inline_image_paths: List[Path] = []
    for alt, path_text in inline_images:
        inline_fig_nums.extend(int(x) for x in FIG_NUM_IN_TEXT_RE.findall(alt))
        p = Path(path_text)
        if p.is_absolute():
            inline_image_paths.append(p)
        else:
            inline_image_paths.append((manuscript.parent / p).resolve())

    fig_file_nums: List[int] = []
    for p in figure_files:
        fig_file_nums.extend(_extract_figure_numbers_from_name(p.stem))

    explicit_fig_nums = sorted(set(inline_fig_nums + fig_file_nums))
    figure_assets = {str(p.resolve()) for p in figure_files}
    figure_assets.update(str(p.resolve()) for p in inline_image_paths)
    total_figure_assets = len(figure_assets)

    # 1. fig_ref_exists
    if not manuscript_exists:
        add(
            "fig_ref_exists",
            "figures",
            "error",
            False,
            f"Manuscript not found: {manuscript}",
        )
    elif not fig_refs:
        add(
            "fig_ref_exists",
            "figures",
            "info",
            True,
            "No figure references found in manuscript text.",
            applicable=False,
        )
    else:
        missing_refs: List[int] = []
        for n in fig_refs:
            if n in explicit_fig_nums:
                continue
            if 1 <= n <= total_figure_assets:
                continue
            missing_refs.append(n)
        if missing_refs:
            add(
                "fig_ref_exists",
                "figures",
                "error",
                False,
                f"Referenced figures missing corresponding assets: {', '.join(str(n) for n in missing_refs)}.",
            )
        else:
            add("fig_ref_exists", "figures", "error", True, "All referenced figures have matching assets.")

    # 2. fig_file_referenced
    if not manuscript_exists:
        add(
            "fig_file_referenced",
            "figures",
            "warning",
            False,
            f"Cannot verify figure file references because manuscript is missing: {manuscript}",
        )
    elif not figure_files:
        add("fig_file_referenced", "figures", "info", True, "No figure files found in results/fig/.")
    else:
        text_lower = manuscript_text.lower()
        unreferenced: List[str] = []
        for p in figure_files:
            rel = p.resolve().relative_to(workspace).as_posix().lower()
            name = p.name.lower()
            stem = p.stem.lower()
            nums = _extract_figure_numbers_from_name(stem)
            mentioned_by_number = any(n in fig_refs for n in nums)
            mentioned_by_text = (rel in text_lower) or (name in text_lower) or (stem in text_lower)
            if not (mentioned_by_text or mentioned_by_number):
                unreferenced.append(p.resolve().relative_to(workspace).as_posix())
        if unreferenced:
            add(
                "fig_file_referenced",
                "figures",
                "warning",
                False,
                f"{len(unreferenced)} figure file(s) are not referenced in manuscript text: {', '.join(unreferenced[:5])}.",
            )
        else:
            add("fig_file_referenced", "figures", "warning", True, "All figure files are referenced.")

    # 3. fig_numbering_sequential
    if not manuscript_exists:
        add(
            "fig_numbering_sequential",
            "figures",
            "warning",
            False,
            f"Cannot verify figure numbering because manuscript is missing: {manuscript}",
        )
    elif not fig_refs:
        add(
            "fig_numbering_sequential",
            "figures",
            "info",
            True,
            "No numbered figure references found.",
            applicable=False,
        )
    else:
        gaps = _sequential_gap(fig_refs)
        if gaps:
            add(
                "fig_numbering_sequential",
                "figures",
                "warning",
                False,
                f"Figure numbering has gaps: missing {', '.join(str(g) for g in gaps)}.",
            )
        else:
            add("fig_numbering_sequential", "figures", "warning", True, "Figure numbering is sequential.")

    # Table parsing shared.
    explicit_table_nums, table_block_count = _extract_table_numbers(manuscript_lines) if manuscript_text else ([], 0)
    explicit_table_num_set = set(explicit_table_nums)

    # 4. table_ref_exists
    if not manuscript_exists:
        add(
            "table_ref_exists",
            "tables",
            "error",
            False,
            f"Manuscript not found: {manuscript}",
        )
    elif not table_refs:
        add(
            "table_ref_exists",
            "tables",
            "info",
            True,
            "No table references found in manuscript text.",
            applicable=False,
        )
    else:
        missing_table_refs: List[int] = []
        for n in table_refs:
            if n in explicit_table_num_set:
                continue
            if 1 <= n <= table_block_count:
                continue
            missing_table_refs.append(n)
        if missing_table_refs:
            add(
                "table_ref_exists",
                "tables",
                "error",
                False,
                f"Referenced tables missing corresponding table blocks: {', '.join(str(n) for n in missing_table_refs)}.",
            )
        else:
            add("table_ref_exists", "tables", "error", True, "All referenced tables have corresponding table blocks.")

    # 5. table_numbering_sequential
    if not manuscript_exists:
        add(
            "table_numbering_sequential",
            "tables",
            "warning",
            False,
            f"Cannot verify table numbering because manuscript is missing: {manuscript}",
        )
    elif not table_refs:
        add(
            "table_numbering_sequential",
            "tables",
            "info",
            True,
            "No numbered table references found.",
            applicable=False,
        )
    else:
        gaps = _sequential_gap(table_refs)
        if gaps:
            add(
                "table_numbering_sequential",
                "tables",
                "warning",
                False,
                f"Table numbering has gaps: missing {', '.join(str(g) for g in gaps)}.",
            )
        else:
            add("table_numbering_sequential", "tables", "warning", True, "Table numbering is sequential.")

    # 6. claims_have_evidence
    claims_dir = workspace / "claims"
    claim_files = sorted(claims_dir.glob("*.md")) if claims_dir.exists() else []
    if not claims_dir.exists() or not claim_files:
        add(
            "claims_have_evidence",
            "claims",
            "info",
            True,
            "No claim markdown files found to validate.",
            applicable=False,
        )
    else:
        missing_claims: List[str] = []
        for claim in claim_files:
            txt = _safe_read_text(claim)
            lines = txt.splitlines()
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
            if not evidence_ids:
                fallback = re.findall(r"\b[0-9a-f]{32}\b", txt, flags=re.IGNORECASE)
                evidence_ids.extend(fallback)
            if not evidence_ids:
                missing_claims.append(claim.resolve().relative_to(workspace).as_posix())
        if missing_claims:
            add(
                "claims_have_evidence",
                "claims",
                "error",
                False,
                f"{len(missing_claims)} claim file(s) have no evidence IDs: {', '.join(missing_claims[:5])}.",
            )
        else:
            add("claims_have_evidence", "claims", "error", True, "All claim files include evidence IDs.")

    # 7. evidence_urls_valid
    evidence_dir = workspace / "evidence"
    evidence_files = sorted(evidence_dir.glob("*.json")) if evidence_dir.exists() else []
    if not evidence_dir.exists() or not evidence_files:
        add(
            "evidence_urls_valid",
            "claims",
            "info",
            True,
            "No evidence JSON files found to validate URLs.",
            applicable=False,
        )
    else:
        invalid_items: List[str] = []
        for ep in evidence_files:
            rel = ep.resolve().relative_to(workspace).as_posix()
            try:
                payload = json.loads(_safe_read_text(ep))
            except json.JSONDecodeError:
                invalid_items.append(f"{rel} (invalid JSON)")
                continue
            rows: List[Dict[str, Any]] = []
            if isinstance(payload, dict):
                rows = [payload]
            elif isinstance(payload, list):
                rows = [r for r in payload if isinstance(r, dict)]
            else:
                invalid_items.append(f"{rel} (unsupported JSON type)")
                continue
            for idx, row in enumerate(rows, start=1):
                url = str(row.get("url") or "").strip()
                if not url:
                    invalid_items.append(f"{rel}#{idx} (missing url)")
                    continue
                v = validate_evidence_url(url)
                if not bool(v.get("valid")):
                    invalid_items.append(f"{rel}#{idx} ({url})")
        if invalid_items:
            add(
                "evidence_urls_valid",
                "claims",
                "error",
                False,
                f"Found invalid evidence URLs in {len(invalid_items)} item(s): {', '.join(invalid_items[:5])}.",
            )
        else:
            add("evidence_urls_valid", "claims", "error", True, "All evidence URLs pass format validation.")

    paragraphs = _extract_paragraphs_with_context(manuscript_text) if manuscript_text else []

    # 8. stats_have_effect_sizes
    if not manuscript_exists:
        add(
            "stats_have_effect_sizes",
            "statistics",
            "warning",
            False,
            f"Cannot evaluate statistics because manuscript is missing: {manuscript}",
        )
    else:
        pvalue_paragraphs = [p for p in paragraphs if P_VALUE_ANY_RE.search(str(p["text"]))]
        if not pvalue_paragraphs:
            add(
                "stats_have_effect_sizes",
                "statistics",
                "info",
                True,
                "No p-values found in manuscript text.",
                applicable=False,
            )
            pvalue_paragraphs = []
        missing_effect_size_locs: List[str] = []
        for p in pvalue_paragraphs:
            txt = str(p["text"])
            if EFFECT_SIZE_RE.search(txt):
                continue
            loc = f"{p['section']}, paragraph {p['paragraph']}"
            missing_effect_size_locs.append(loc)
        if pvalue_paragraphs and missing_effect_size_locs:
            add(
                "stats_have_effect_sizes",
                "statistics",
                "warning",
                False,
                f"P-value paragraph(s) missing effect sizes: {', '.join(missing_effect_size_locs[:5])}.",
                location=missing_effect_size_locs[0],
            )
        elif pvalue_paragraphs:
            add(
                "stats_have_effect_sizes",
                "statistics",
                "warning",
                True,
                "All paragraphs with p-values include effect size indicators.",
            )

    # 9. stats_p_value_format
    if not manuscript_exists:
        add(
            "stats_p_value_format",
            "statistics",
            "warning",
            False,
            f"Cannot evaluate p-value format because manuscript is missing: {manuscript}",
        )
    else:
        pvalue_matches = list(P_VALUE_ANY_RE.finditer(manuscript_text))
        if not pvalue_matches:
            add(
                "stats_p_value_format",
                "statistics",
                "info",
                True,
                "No p-values found in manuscript text.",
                applicable=False,
            )
            pvalue_matches = []
        bad_formats: List[str] = []
        for match in pvalue_matches:
            op = match.group(1)
            value = match.group(2)
            is_bad = False
            reason = ""
            if value.startswith("."):
                is_bad = True
                reason = "missing leading zero"
            try:
                fval = float(value)
            except ValueError:
                is_bad = True
                reason = "non-numeric p-value"
                fval = -1.0
            if not is_bad and (fval < 0.0 or fval > 1.0):
                is_bad = True
                reason = "outside [0,1]"
            if not is_bad and op == "=" and fval == 0.0:
                is_bad = True
                reason = "p cannot be exactly 0"
            if not is_bad and op in {"<", "<="} and re.fullmatch(r"0\.0{3,}", value):
                is_bad = True
                reason = "over-precise zero threshold"

            if is_bad:
                line = _line_from_pos(manuscript_text, match.start())
                bad_formats.append(f"line {line}: p {op} {value} ({reason})")
        if pvalue_matches and bad_formats:
            add(
                "stats_p_value_format",
                "statistics",
                "warning",
                False,
                f"Invalid p-value formats detected: {', '.join(bad_formats[:5])}.",
                location=bad_formats[0].split(":")[0],
            )
        elif pvalue_matches:
            add("stats_p_value_format", "statistics", "warning", True, "All p-values use accepted formatting.")

    # References parsing (shared).
    ref_entries, references_section_text = _extract_references(manuscript_lines) if manuscript_text else ([], "")

    # 10. refs_have_dois
    if not manuscript_exists:
        add("refs_have_dois", "references", "error", False, f"Cannot check references: manuscript missing {manuscript}")
    elif not ref_entries:
        add("refs_have_dois", "references", "error", False, "No reference entries found in References section.")
    else:
        doi_count = sum(1 for e in ref_entries if DOI_RE.search(e))
        total_refs = len(ref_entries)
        fraction = (float(doi_count) / float(total_refs)) if total_refs else 0.0
        if fraction >= 0.8:
            add(
                "refs_have_dois",
                "references",
                "error",
                True,
                f"{doi_count}/{total_refs} references include DOI identifiers ({fraction * 100:.1f}%).",
            )
        else:
            add(
                "refs_have_dois",
                "references",
                "error",
                False,
                f"Only {doi_count}/{total_refs} references include DOI identifiers ({fraction * 100:.1f}%).",
            )

    # 11. refs_no_placeholder
    if not manuscript_exists:
        add(
            "refs_no_placeholder",
            "references",
            "error",
            False,
            f"Cannot check placeholder references: manuscript missing {manuscript}",
        )
    elif not references_section_text.strip():
        add("refs_no_placeholder", "references", "warning", False, "References section not found or empty.")
    else:
        m = REF_PLACEHOLDER_RE.search(references_section_text)
        if m:
            line_offset = references_section_text[: m.start()].count("\n") + 1
            add(
                "refs_no_placeholder",
                "references",
                "error",
                False,
                f"Reference section contains placeholder token: {m.group(0)}.",
                location=f"References section, line {line_offset}",
            )
        else:
            add("refs_no_placeholder", "references", "error", True, "No placeholder tokens found in references.")

    # Scoreboard shared for checks 12, 13, 16.
    scoreboard_path = workspace / "results" / "scoreboard.json"
    scoreboard_obj: Optional[Dict[str, Any]] = None
    scoreboard_err: Optional[str] = None
    if not scoreboard_path.exists():
        scoreboard_err = f"Missing file: {scoreboard_path.resolve().relative_to(workspace).as_posix()}"
    else:
        try:
            raw = _safe_read_text(scoreboard_path)
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                scoreboard_obj = parsed
            else:
                scoreboard_err = "scoreboard.json must be a JSON object"
        except json.JSONDecodeError as exc:
            scoreboard_err = f"Invalid JSON: {exc}"

    # 12. scoreboard_exists
    if scoreboard_obj is not None:
        add("scoreboard_exists", "reproducibility", "error", True, "results/scoreboard.json exists and is valid JSON.")
    else:
        add("scoreboard_exists", "reproducibility", "error", False, f"results/scoreboard.json invalid: {scoreboard_err}.")

    # 13. scoreboard_has_primary_metric
    if scoreboard_obj is None:
        add(
            "scoreboard_has_primary_metric",
            "reproducibility",
            "error",
            False,
            "Cannot validate primary_metric.current because scoreboard.json is invalid or missing.",
        )
    else:
        pm = scoreboard_obj.get("primary_metric")
        cur = pm.get("current") if isinstance(pm, dict) else None
        cur_v = _to_float(cur)
        if cur is not None and (cur_v is not None or isinstance(cur, dict)):
            add(
                "scoreboard_has_primary_metric",
                "reproducibility",
                "error",
                True,
                "scoreboard.primary_metric.current is present and non-null.",
            )
        else:
            add(
                "scoreboard_has_primary_metric",
                "reproducibility",
                "error",
                False,
                "scoreboard.primary_metric.current is missing or null.",
            )

    # 14. method_describes_metrics
    method_path = workspace / "notes" / "method.md"
    if not method_path.exists():
        add("method_describes_metrics", "reproducibility", "warning", False, "notes/method.md not found.")
    else:
        method_text = _safe_read_text(method_path)
        method_lower = method_text.lower()
        has_metric_keyword = "metric" in method_lower
        has_def_keyword = any(k in method_lower for k in ("definition", "primary metric", "direction", "baseline"))
        scoreboard_metrics = _flatten_scoreboard_metrics(scoreboard_obj or {})
        metric_name_hits = 0
        if scoreboard_metrics:
            for name in scoreboard_metrics.keys():
                n = name.lower()
                if n and (n in method_lower or n.replace("_", " ") in method_lower):
                    metric_name_hits += 1
        passed = has_metric_keyword and has_def_keyword and (metric_name_hits > 0 or "metric definitions" in method_lower)
        if passed:
            add("method_describes_metrics", "reproducibility", "warning", True, "notes/method.md includes metric definitions.")
        else:
            add(
                "method_describes_metrics",
                "reproducibility",
                "warning",
                False,
                "notes/method.md does not clearly define metrics (missing definitions or metric names).",
            )

    # 15. code_in_workspace
    src_dir = workspace / "src"
    py_files = sorted(src_dir.rglob("*.py")) if src_dir.exists() else []
    if src_dir.exists() and py_files:
        add("code_in_workspace", "reproducibility", "error", True, f"Found {len(py_files)} Python files under src/.")
    elif not src_dir.exists():
        add("code_in_workspace", "reproducibility", "error", False, "src/ directory is missing.")
    else:
        add("code_in_workspace", "reproducibility", "error", False, "src/ exists but no Python files were found.")

    # 16. text_numbers_match_scoreboard
    if not manuscript_exists:
        add(
            "text_numbers_match_scoreboard",
            "consistency",
            "warning",
            False,
            f"Cannot verify text metrics because manuscript is missing: {manuscript}",
        )
    elif scoreboard_obj is None:
        add(
            "text_numbers_match_scoreboard",
            "consistency",
            "warning",
            False,
            "Cannot verify manuscript metric values because scoreboard.json is invalid or missing.",
        )
    else:
        metric_map = _flatten_scoreboard_metrics(scoreboard_obj)
        if not metric_map:
            add(
                "text_numbers_match_scoreboard",
                "consistency",
                "info",
                True,
                "No numeric metrics found in scoreboard to cross-check against manuscript text.",
            )
        else:
            mentioned_count = 0
            mismatches: List[str] = []
            lower_text = manuscript_text.lower()
            for metric_name, expected_value in metric_map.items():
                variants = _metric_name_variants(metric_name)
                for variant in variants:
                    frag = _regex_from_metric_variant(variant)
                    patterns = [
                        re.compile(rf"\b{frag}\b\s*(?::|=|is|was|of|at)?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
                        re.compile(rf"(-?\d+(?:\.\d+)?)\s*(?:for|on|in)?\s*\b{frag}\b", re.IGNORECASE),
                    ]
                    for pat in patterns:
                        for m in pat.finditer(lower_text):
                            mentioned_count += 1
                            found = float(m.group(1))
                            if float(expected_value).is_integer():
                                tolerance = 0.0
                            else:
                                tolerance = max(0.01, abs(float(expected_value)) * 0.02)
                            if abs(found - float(expected_value)) > tolerance:
                                line = _line_from_pos(manuscript_text, m.start())
                                mismatches.append(
                                    f"line {line}: {metric_name}={found:g} (scoreboard {float(expected_value):g})"
                                )
            if mentioned_count == 0:
                add(
                    "text_numbers_match_scoreboard",
                    "consistency",
                    "info",
                    True,
                    "No explicit scoreboard metric values were detected in manuscript text.",
                    applicable=False,
                )
            elif mismatches:
                add(
                    "text_numbers_match_scoreboard",
                    "consistency",
                    "warning",
                    False,
                    f"Metric value mismatches found: {', '.join(mismatches[:5])}.",
                    location=mismatches[0].split(":")[0],
                )
            else:
                add(
                    "text_numbers_match_scoreboard",
                    "consistency",
                    "warning",
                    True,
                    f"Detected {mentioned_count} metric value mention(s); all match scoreboard values.",
                )

    # 17. refs_citations_exist
    if not manuscript_exists:
        add(
            "refs_citations_exist",
            "references",
            "error",
            False,
            f"Cannot validate inline citations because manuscript is missing: {manuscript}",
        )
    else:
        citation_nums = [int(x) for x in INLINE_CITATION_RE.findall(manuscript_text)]
        if not citation_nums:
            add(
                "refs_citations_exist",
                "references",
                "error",
                True,
                "No inline numeric citations found to validate.",
                applicable=False,
            )
        else:
            total_refs = len(ref_entries)
            orphaned = sorted({n for n in citation_nums if n > total_refs})
            if orphaned:
                first = orphaned[0]
                m = re.search(rf"\[{first}\]", manuscript_text)
                line = _line_from_pos(manuscript_text, m.start()) if m else None
                add(
                    "refs_citations_exist",
                    "references",
                    "error",
                    False,
                    (
                        f"Inline citation(s) exceed reference count ({total_refs}): "
                        f"{', '.join(str(n) for n in orphaned)}."
                    ),
                    location=(f"line {line}" if line is not None else None),
                )
            else:
                add(
                    "refs_citations_exist",
                    "references",
                    "error",
                    True,
                    f"All inline citations map to existing references (total references: {total_refs}).",
                )

    # 18. abstract_body_consistency
    if not manuscript_exists:
        add(
            "abstract_body_consistency",
            "consistency",
            "warning",
            False,
            f"Cannot compare abstract/body numeric claims because manuscript is missing: {manuscript}",
        )
    else:
        abstract_bounds = _find_section_bounds(manuscript_lines, "abstract")
        if abstract_bounds is None:
            add(
                "abstract_body_consistency",
                "consistency",
                "warning",
                True,
                "No Abstract section found; abstract-vs-body percentage consistency not applicable.",
            )
        else:
            abs_start, abs_end = abstract_bounds
            abstract_text = "\n".join(manuscript_lines[abs_start:abs_end]).strip()
            body_text = "\n".join(manuscript_lines[abs_end:]).strip()
            abstract_claims = _extract_percent_claims(abstract_text)
            body_claims = _extract_percent_claims(body_text)
            if not abstract_claims or not body_claims:
                add(
                    "abstract_body_consistency",
                    "consistency",
                    "warning",
                    True,
                    "No comparable percentage claims found across Abstract and body.",
                    applicable=False,
                )
            else:
                contradictions: List[str] = []
                for concept in sorted(set(abstract_claims) & set(body_claims)):
                    abs_vals = sorted(abstract_claims[concept])
                    body_vals = sorted(body_claims[concept])
                    if abs_vals != body_vals:
                        contradictions.append(
                            f"{concept}: abstract={','.join(f'{v:g}' for v in abs_vals)}% "
                            f"vs body={','.join(f'{v:g}' for v in body_vals)}%"
                        )
                if contradictions:
                    add(
                        "abstract_body_consistency",
                        "consistency",
                        "warning",
                        False,
                        f"Contradictory percentage claims between Abstract and body: {', '.join(contradictions[:5])}.",
                        location="Abstract",
                    )
                else:
                    add(
                        "abstract_body_consistency",
                        "consistency",
                        "warning",
                        True,
                        "Abstract and body percentage claims are consistent for overlapping concepts.",
                    )

    # 19. stats_significance_claim_valid
    if not manuscript_exists:
        add(
            "stats_significance_claim_valid",
            "statistics",
            "error",
            False,
            f"Cannot validate significance claims because manuscript is missing: {manuscript}",
        )
    else:
        misuse_locs: List[str] = []
        significance_combo_count = 0
        for p in paragraphs:
            txt = str(p["text"])
            if not SIGNIFICANCE_CLAIM_RE.search(txt):
                continue
            for m in P_VALUE_ANY_RE.finditer(txt):
                significance_combo_count += 1
                op = m.group(1)
                try:
                    pval = float(m.group(2))
                except ValueError:
                    continue
                if _is_explicitly_gt_005(op, pval):
                    misuse_locs.append(f"{p['section']}, paragraph {p['paragraph']} (p {op} {pval:g})")
                    break
        if significance_combo_count == 0:
            add(
                "stats_significance_claim_valid",
                "statistics",
                "info",
                True,
                "No p-value + significance combinations found.",
                applicable=False,
            )
        elif misuse_locs:
            add(
                "stats_significance_claim_valid",
                "statistics",
                "error",
                False,
                f"Significance claimed with non-significant p-value(s): {', '.join(misuse_locs[:5])}.",
                location=misuse_locs[0],
            )
        else:
            add(
                "stats_significance_claim_valid",
                "statistics",
                "error",
                True,
                "All significance claims with p-values use significant thresholds (<= 0.05).",
            )

    return _build_report(checks)


def format_consistency_report(report: ConsistencyReport) -> str:
    checks = report.checks
    failed_checks = [c for c in checks if not c.passed]
    error_count = sum(1 for c in failed_checks if c.severity == "error")
    warning_count = sum(1 for c in failed_checks if c.severity == "warning")

    lines: List[str] = []
    lines.append("# Manuscript Consistency Report")
    lines.append("## Summary")
    lines.append(f"- Score: {report.passed_checks}/{report.total_checks} checks passed ({report.consistency_score * 100:.1f}%)")
    lines.append(f"- {report.failed_checks} issues found ({error_count} error, {warning_count} warning)")
    lines.append("")

    category_order = [
        "figures",
        "tables",
        "claims",
        "statistics",
        "references",
        "reproducibility",
        "consistency",
    ]
    category_title = {
        "figures": "Figures",
        "tables": "Tables",
        "claims": "Claims",
        "statistics": "Statistics",
        "references": "References",
        "reproducibility": "Reproducibility",
        "consistency": "Internal Consistency",
    }

    checks_by_category: Dict[str, List[CheckResult]] = {}
    for chk in checks:
        checks_by_category.setdefault(chk.category, []).append(chk)

    for category in category_order:
        if category not in checks_by_category:
            continue
        cks = checks_by_category[category]
        passed_count = sum(1 for c in cks if c.passed)
        total_count = len(cks)
        status = "✓" if passed_count == total_count else "!"
        lines.append(f"## {category_title.get(category, category.title())} ({passed_count}/{total_count} {status})")
        for c in cks:
            if c.passed:
                icon = "✓"
            elif c.severity == "error":
                icon = "❌"
            elif c.severity == "warning":
                icon = "⚠️"
            else:
                icon = "ℹ️"
            loc = f" ({c.location})" if c.location else ""
            lines.append(f"- {icon} {c.check_id}: {c.message}{loc}")
        lines.append("")

    lines.append("## Issues")
    if not failed_checks:
        lines.append("1. ✓ No issues found.")
    else:
        for idx, c in enumerate(failed_checks, start=1):
            icon = "❌" if c.severity == "error" else ("⚠️" if c.severity == "warning" else "ℹ️")
            loc = f" ({c.location})" if c.location else ""
            lines.append(f"{idx}. {icon} [{c.severity}] {c.check_id}: {c.message}{loc}")
    lines.append("")
    return "\n".join(lines)


def write_consistency_report(
    workspace_dir: Path,
    report: ConsistencyReport,
    output_path: Optional[Path] = None,
) -> Path:
    workspace = Path(workspace_dir).resolve()
    out = resolve_within_workspace(
        workspace, output_path or "results/manuscript_consistency_report.md", label="consistency report output path"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_consistency_report(report), encoding="utf-8")
    return out


__all__ = [
    "CheckResult",
    "ConsistencyReport",
    "check_manuscript_consistency",
    "format_consistency_report",
    "write_consistency_report",
]
