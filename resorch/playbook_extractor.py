from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from resorch.ledger import Ledger

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


log = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)(.+?)\s*$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("mean", "value", "current", "best", "baseline"):
            if key in value:
                out = _to_float(value.get(key))
                if out is not None:
                    return out
        return None
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _safe_read_json(path: Path) -> Dict[str, Any]:
    raw = _safe_read_text(path).strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _clean_line(line: str) -> str:
    out = _LINK_RE.sub(r"\1", line or "")
    out = out.replace("`", "").strip("-:* ")
    out = _SPACE_RE.sub(" ", out).strip()
    return out


def _is_placeholder(line: str) -> bool:
    s = (line or "").strip().lower()
    if not s:
        return True
    if s in {"-", "*"}:
        return True
    if "(fill" in s:
        return True
    if "created:" in s:
        return True
    return False


def _dedupe_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        clean = _clean_line(value)
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _parse_markdown_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {"root": []}
    current = "root"
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            current = _clean_line(match.group(1)).lower() or "root"
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line.rstrip())
    return sections


def _extract_section_items(sections: Dict[str, List[str]], section_name: str) -> List[str]:
    out: List[str] = []
    for line in sections.get(section_name, []):
        line = line.rstrip()
        match = _BULLET_RE.match(line)
        if match:
            candidate = _clean_line(match.group(1))
        else:
            candidate = _clean_line(line)
        if _is_placeholder(candidate):
            continue
        if candidate:
            out.append(candidate)
    return out


def _extract_learning_summary(sections: Dict[str, List[str]]) -> str:
    preferred = ("latest", "summary", "key learnings", "learnings", "notes", "root")
    for section_name in preferred:
        items = _extract_section_items(sections, section_name)
        if not items:
            continue
        line = items[0]
        if len(line) > 180:
            line = line[:177].rstrip() + "..."
        return line

    for section_name in sections:
        items = _extract_section_items(sections, section_name)
        if items:
            line = items[0]
            if len(line) > 180:
                line = line[:177].rstrip() + "..."
            return line
    return ""


def _extract_when_to_apply(sections: Dict[str, List[str]], learning_summary: str, mode: str) -> List[str]:
    max_items = 6 if mode == "full" else 3
    section_keywords = ("when", "condition", "failure", "latest", "finding", "note", "lesson")
    condition_keywords = ("if", "when", "unless", "under", "fails", "failure", "degrad", "regress", "drift")

    candidates: List[str] = []
    for section_name in sections:
        items = _extract_section_items(sections, section_name)
        if any(k in section_name for k in section_keywords):
            candidates.extend(items)
            continue
        for item in items:
            low = item.lower()
            if any(k in low for k in condition_keywords):
                candidates.append(item)

    candidates = _dedupe_keep_order(candidates)
    if not candidates and learning_summary:
        candidates = [learning_summary]
    return candidates[:max_items]


def _extract_steps(sections: Dict[str, List[str]], mode: str) -> List[str]:
    max_items = 8 if mode == "full" else 4
    step_section_keywords = ("next actions", "next action", "next steps", "action", "todo", "recommend", "plan")
    imperative_prefixes = (
        "run",
        "add",
        "check",
        "validate",
        "measure",
        "compare",
        "ablate",
        "collect",
        "test",
        "tune",
        "switch",
        "use",
        "try",
    )

    candidates: List[str] = []
    for section_name in sections:
        items = _extract_section_items(sections, section_name)
        if any(k in section_name for k in step_section_keywords):
            candidates.extend(items)
            continue
        for item in items:
            first = item.lower().split(" ", 1)[0]
            if first in imperative_prefixes:
                candidates.append(item)

    return _dedupe_keep_order(candidates)[:max_items]


def _review_files(workspace: Path) -> List[Path]:
    reviews_dir = workspace / "reviews"
    if not reviews_dir.exists() or not reviews_dir.is_dir():
        return []

    files: List[Tuple[float, Path]] = []
    for path in reviews_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        files.append((mtime, path))

    files.sort(key=lambda t: (t[0], t[1].name), reverse=True)
    return [p for _mtime, p in files]


def _anti_patterns_from_review_json(data: Dict[str, Any]) -> List[str]:
    findings = data.get("findings")
    out: List[str] = []
    if isinstance(findings, list):
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            severity = str(finding.get("severity") or "").strip().lower()
            message = _clean_line(str(finding.get("message") or ""))
            if not message:
                continue
            if severity in {"major", "blocker"}:
                out.append(message)
                continue
            low = message.lower()
            if any(k in low for k in ("avoid", "do not", "don't", "leak", "overfit", "cherry-pick", "regression", "failure")):
                out.append(message)

    overall = _clean_line(str(data.get("overall") or ""))
    if overall and any(k in overall.lower() for k in ("avoid", "do not", "don't", "risk", "failure")):
        out.append(overall)
    return out


def _anti_patterns_from_review_markdown(text: str) -> List[str]:
    sections = _parse_markdown_sections(text)
    out: List[str] = []
    anti_keys = ("anti", "avoid", "risk", "blocker", "failure")
    for section_name in sections:
        items = _extract_section_items(sections, section_name)
        if any(k in section_name for k in anti_keys):
            out.extend(items)
            continue
        for item in items:
            low = item.lower()
            if any(k in low for k in ("avoid", "do not", "don't", "never", "leak", "overfit", "cherry-pick")):
                out.append(item)
    return out


def _extract_anti_patterns(workspace: Path, mode: str) -> Tuple[List[str], List[str]]:
    max_files = 20 if mode == "full" else 8
    max_items = 8 if mode == "full" else 4

    anti_patterns: List[str] = []
    source_paths: List[str] = []

    for path in _review_files(workspace)[:max_files]:
        try:
            rel = path.resolve().relative_to(workspace.resolve()).as_posix()
        except ValueError:
            continue
        source_paths.append(rel)

        if path.suffix.lower() == ".json":
            raw = _safe_read_json(path)
            if raw:
                anti_patterns.extend(_anti_patterns_from_review_json(raw))
            continue

        if path.suffix.lower() in {".md", ".txt"}:
            anti_patterns.extend(_anti_patterns_from_review_markdown(_safe_read_text(path)))

    return _dedupe_keep_order(anti_patterns)[:max_items], _dedupe_keep_order(source_paths)


def _extract_tags(domain: str, title: str) -> List[str]:
    stop = {
        "a",
        "an",
        "and",
        "for",
        "from",
        "the",
        "with",
        "using",
        "study",
        "project",
        "task",
    }
    tokens = _TOKEN_RE.findall(f"{domain} {title}".lower())
    out: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) < 3 or token in stop:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out[:10]


def extract_playbook_entry(
    ledger: Ledger,
    project_id: str,
    mode: str = "compact",
    timestamp: str | None = None,
) -> Dict[str, Any]:
    mode = "full" if str(mode).lower() == "full" else "compact"

    project = ledger.get_project(project_id)
    workspace = Path(str(project.get("repo_path") or "")).resolve()
    title = str(project.get("title") or project_id).strip() or project_id
    domain = str(project.get("domain") or "").strip()

    digest_path = workspace / "notes" / "analysis_digest.md"
    scoreboard_path = workspace / "results" / "scoreboard.json"

    digest_text = _safe_read_text(digest_path)
    digest_sections = _parse_markdown_sections(digest_text) if digest_text.strip() else {"root": []}

    scoreboard = _safe_read_json(scoreboard_path)
    pm = scoreboard.get("primary_metric")
    if not isinstance(pm, dict):
        pm = {}

    current = _to_float(pm.get("current"))
    baseline = _to_float(pm.get("baseline"))
    best = _to_float(pm.get("best"))
    delta = _to_float(pm.get("delta_vs_baseline"))
    if delta is None and current is not None and baseline is not None:
        delta = current - baseline

    learning_summary = _extract_learning_summary(digest_sections)
    when_to_apply = _extract_when_to_apply(digest_sections, learning_summary, mode)
    steps = _extract_steps(digest_sections, mode)
    anti_patterns, review_artifacts = _extract_anti_patterns(workspace, mode)

    claim_artifacts: List[str] = []
    claims_dir = workspace / "claims"
    if claims_dir.exists() and claims_dir.is_dir():
        claim_paths = sorted([p for p in claims_dir.rglob("*") if p.is_file()])
        for p in claim_paths:
            try:
                claim_artifacts.append(p.resolve().relative_to(workspace.resolve()).as_posix())
            except ValueError:
                continue

    key_artifacts: List[str] = []
    if digest_path.exists():
        key_artifacts.append("notes/analysis_digest.md")
    if scoreboard_path.exists():
        key_artifacts.append("results/scoreboard.json")
    key_artifacts.extend(review_artifacts)
    key_artifacts.extend(claim_artifacts)
    key_artifacts = _dedupe_keep_order(key_artifacts)
    key_artifacts = key_artifacts[: (24 if mode == "full" else 12)]

    has_signal = any(
        [
            bool(learning_summary),
            bool(when_to_apply),
            bool(steps),
            bool(anti_patterns),
            bool(str(pm.get("name") or "").strip()),
            current is not None,
            baseline is not None,
        ]
    )

    if not learning_summary:
        learning_summary = "manual extraction required"

    stamp = str(timestamp or "").strip() or _stamp_utc()
    entry_id = f"pb_{project_id}_{stamp}"
    entry: Dict[str, Any] = {
        "id": entry_id,
        "title": f"{title} - {learning_summary}",
        "when_to_apply": when_to_apply,
        "steps": steps,
        "anti_patterns": anti_patterns,
        "evidence": {
            "project_id": project_id,
            "primary_metric": {
                "name": str(pm.get("name") or "").strip(),
                "direction": str(pm.get("direction") or "").strip(),
                "current": current,
                "baseline": baseline,
                "best": best,
            },
            "delta_vs_baseline": delta,
            "key_artifacts": key_artifacts,
        },
        "tags": _extract_tags(domain, title),
        "topic": domain,
    }

    if not has_signal:
        entry["needs_human"] = True
        if not entry["when_to_apply"]:
            entry["when_to_apply"] = ["Insufficient structured evidence in analysis_digest/reviews."]
        if not entry["steps"]:
            entry["steps"] = ["Review this project manually before reusing its strategy."]

    return entry


def extract_and_save(ledger: Ledger, project_id: str, mode: str = "compact") -> Dict[str, Any]:
    if yaml is None:
        raise SystemExit("PyYAML is required for playbook extraction: pip install pyyaml")

    project = ledger.get_project(project_id)
    workspace = Path(str(project.get("repo_path") or "")).resolve()

    stamp = _stamp_utc()
    entry = extract_playbook_entry(ledger=ledger, project_id=project_id, mode=mode, timestamp=stamp)

    out_dir = workspace / "playbook" / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / f"{stamp}_{project_id}.yaml"
    yaml_path.write_text(yaml.safe_dump(entry, allow_unicode=True, sort_keys=False), encoding="utf-8")

    topic = str(entry.get("topic") or "").strip() or str(project.get("domain") or "").strip() or project_id
    ledger_stored = False
    try:
        ledger.upsert_playbook_entry(entry_id=str(entry["id"]), topic=topic, rule=entry)
        ledger_stored = True
    except Exception:  # noqa: BLE001
        log.exception("Failed to upsert extracted playbook entry: project_id=%s entry_id=%s", project_id, entry.get("id"))

    return {
        "entry_id": str(entry["id"]),
        "yaml_path": str(yaml_path),
        "ledger_stored": ledger_stored,
    }
