from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.artifacts import register_artifact
from resorch.ideas import get_idea as get_idea_fn
from resorch.ideas import set_idea_status as set_idea_status_fn
from resorch.ledger import Ledger
from resorch.utils import utc_now_iso


def _fmt_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for it in items:
        if isinstance(it, str):
            s = it.strip()
            if s:
                out.append(s)
    return out


def _md_escape(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _render_evidence(evidence: Any) -> List[str]:
    if not isinstance(evidence, list):
        return ["- (none)"]
    lines: List[str] = []
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        title = str(ev.get("title") or "").strip() or "(untitled)"
        url = str(ev.get("url") or "").strip()
        summary = str(ev.get("summary") or "").strip()
        retrieved_at = str(ev.get("retrieved_at") or "").strip()
        relevance = ev.get("relevance")
        rel_s = ""
        try:
            if relevance is not None:
                rel_s = f" (relevance={float(relevance):.2f})"
        except (TypeError, ValueError):
            rel_s = ""

        head = f"- {title}"
        if url:
            head = f"- [{title}]({url})"
        tail_parts = []
        if retrieved_at:
            tail_parts.append(f"retrieved_at={retrieved_at}")
        if rel_s:
            tail_parts.append(rel_s.strip())
        tail = ""
        if tail_parts:
            tail = " — " + ", ".join(tail_parts)
        if summary:
            tail = (tail + " — " if tail else " — ") + summary
        lines.append(head + tail)

    return lines or ["- (none)"]


def _render_smoke_section(smoke: Optional[Dict[str, Any]]) -> List[str]:
    if smoke is None:
        return [
            "## Smoke Test",
            "",
            "- (no smoke test results recorded)",
            "",
        ]

    result = smoke.get("result") or {}
    if not isinstance(result, dict):
        result = {}
    verdict = str(smoke.get("verdict") or result.get("verdict") or "").strip()
    started_at = str(smoke.get("started_at") or result.get("started_at") or "").strip()
    completed_at = str(smoke.get("completed_at") or result.get("completed_at") or "").strip()

    lines = [
        "## Smoke Test",
        "",
        f"- verdict: `{verdict or 'unknown'}`",
    ]
    if started_at:
        lines.append(f"- started_at: `{started_at}`")
    if completed_at:
        lines.append(f"- completed_at: `{completed_at}`")

    metrics = result.get("metrics")
    if isinstance(metrics, list) and metrics:
        lines.append("")
        lines.append("### Metrics")
        for m in metrics:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or "").strip()
            if not name:
                continue
            unit = m.get("unit")
            unit_s = f" {unit}" if isinstance(unit, str) and unit.strip() else ""
            try:
                val = float(m.get("value"))
                lines.append(f"- {name}: {val}{unit_s}")
            except (TypeError, ValueError):
                continue

    checkpoints = result.get("checkpoints")
    if isinstance(checkpoints, list) and checkpoints:
        lines.append("")
        lines.append("### Checkpoints")
        for c in checkpoints:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "").strip()
            status = str(c.get("status") or "").strip()
            if not name or not status:
                continue
            notes = str(c.get("notes") or "").strip()
            extra = f" — {notes}" if notes else ""
            lines.append(f"- {name}: `{status}`{extra}")

    lines.append("")
    return lines


def write_topic_brief(
    *,
    ledger: Ledger,
    project_id: str,
    idea_id: str,
    output_path: str = "topic_brief.md",
    register_as_artifact: bool = True,
    set_selected: bool = False,
) -> Dict[str, Any]:
    project = ledger.get_project(project_id)
    workspace = Path(project["repo_path"]).resolve()

    idea = get_idea_fn(ledger=ledger, idea_id=idea_id)
    if str(idea.get("project_id")) != project_id:
        raise SystemExit(f"Idea {idea_id} does not belong to project {project_id}.")

    data = idea.get("data") or {}
    if not isinstance(data, dict):
        data = {}

    title = str(data.get("title") or "").strip() or f"Idea {idea_id}"
    claim = str(data.get("one_sentence_claim") or "").strip()
    novelty = str(data.get("novelty_statement") or "").strip()
    contrib = str(data.get("contribution_type") or "").strip()
    venues = _fmt_list(data.get("target_venues"))

    eval_plan = data.get("evaluation_plan") or {}
    if not isinstance(eval_plan, dict):
        eval_plan = {}
    datasets = _fmt_list(eval_plan.get("datasets"))
    metrics = _fmt_list(eval_plan.get("metrics"))
    baselines = _fmt_list(eval_plan.get("baselines"))
    ablations = _fmt_list(eval_plan.get("ablations"))

    feasibility = data.get("feasibility") or {}
    if not isinstance(feasibility, dict):
        feasibility = {}
    risks = data.get("risks") or {}
    if not isinstance(risks, dict):
        risks = {}

    smoke_rows = ledger.list_smoke_tests(project_id=project_id, idea_id=idea_id, limit=1)
    smoke_latest = None
    if smoke_rows:
        smoke_latest = dict(smoke_rows[0])
        smoke_latest["result"] = json.loads(smoke_latest.pop("result_json") or "{}")

    lines: List[str] = []
    lines.extend([f"# Topic Brief — {title}", ""])
    lines.extend([f"- idea_id: `{idea_id}`", f"- generated_at: `{utc_now_iso()}`", ""])

    lines.extend(["## Title Ideas", "", f"1. {title}", "2. (alt title)", "3. (alt title)", ""])

    lines.extend(["## One-Sentence Claim", "", _md_escape(claim) if claim else "(missing)", ""])

    lines.extend(["## Contribution Type", "", f"- `{contrib or 'unknown'}`", ""])

    lines.extend(["## Target Venues", ""])
    if venues:
        lines.extend([f"- {v}" for v in venues])
    else:
        lines.append("- (none)")
    lines.append("")

    lines.extend(["## Novelty / Positioning", "", _md_escape(novelty) if novelty else "(missing)", ""])

    lines.extend(["## Key Evidence", ""])
    lines.extend(_render_evidence(data.get("evidence")))
    lines.append("")

    lines.extend(["## Evaluation Plan", ""])
    lines.extend(["### Datasets", ""] + ([f"- {d}" for d in datasets] if datasets else ["- (none)"]) + [""])
    lines.extend(["### Metrics", ""] + ([f"- {m}" for m in metrics] if metrics else ["- (none)"]) + [""])
    lines.extend(["### Baselines", ""] + ([f"- {b}" for b in baselines] if baselines else ["- (none)"]) + [""])
    lines.extend(["### Ablations", ""] + ([f"- {a}" for a in ablations] if ablations else ["- (none)"]) + [""])

    lines.extend(["## Feasibility", ""])
    for k in ("estimated_gpu_hours", "estimated_calendar_days"):
        if k in feasibility:
            lines.append(f"- {k}: {feasibility.get(k)}")
    deps = _fmt_list(feasibility.get("blocking_dependencies"))
    lines.append(f"- blocking_dependencies: {', '.join(deps) if deps else '(none)'}")
    notes = str(feasibility.get("notes") or "").strip()
    if notes:
        lines.append(f"- notes: {_md_escape(notes)}")
    lines.append("")

    lines.extend(_render_smoke_section(smoke_latest))

    lines.extend(["## Risks", ""])
    for k in ("ethics", "license", "safety", "reproducibility"):
        v = str(risks.get(k) or "").strip()
        lines.append(f"- {k}: {v or '(missing)'}")
    lines.append("")

    lines.extend(
        [
            "## 2-Week Plan (Draft)",
            "",
            "- Day 1-2: tighten novelty statement, pick baselines",
            "- Day 3-5: implement / reproduce baseline + smoke test",
            "- Day 6-9: main experiments + ablations",
            "- Day 10-12: write draft + figures",
            "- Day 13-14: redteam review + fixes",
            "",
        ]
    )

    out_p = Path(output_path)
    if not out_p.is_absolute():
        out_p = (workspace / out_p).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    artifact = None
    if register_as_artifact:
        artifact = register_artifact(
            ledger=ledger,
            project={"id": project_id, "repo_path": str(workspace)},
            kind="topic_brief_md",
            relative_path=out_p.resolve().relative_to(workspace).as_posix(),
            meta={"idea_id": idea_id},
        )

    updated_idea = None
    if set_selected:
        updated_idea = set_idea_status_fn(ledger=ledger, idea_id=idea_id, status="selected")

    return {
        "output_path": str(out_p),
        "artifact": artifact,
        "idea": updated_idea,
    }

