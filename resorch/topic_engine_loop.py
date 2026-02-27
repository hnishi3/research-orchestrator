from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from resorch.ideas import dedupe_ideas_jsonl as dedupe_ideas_jsonl_fn
from resorch.ideas import import_ideas_jsonl as import_ideas_jsonl_fn
from resorch.ideas import list_ideas as list_ideas_fn
from resorch.ideas import score_ideas as score_ideas_fn
from resorch.ideas import set_idea_status as set_idea_status_fn
from resorch.tasks import create_task as create_task_fn
from resorch.tasks import run_task as run_task_fn
from resorch.topic_brief import write_topic_brief as write_topic_brief_fn
from resorch.utils import utc_now_iso

log = logging.getLogger(__name__)


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            count += 1
    return count


def _normalize_title(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().strip().split())


def _simple_title_dedupe_jsonl(*, input_abs: Path, output_abs: Path) -> Path:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for raw in input_abs.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        idea_id = str(rec.get("id") or "").strip()
        if not idea_id:
            continue
        title_key = _normalize_title(rec.get("title")) or idea_id.lower()
        if title_key in seen:
            continue
        seen.add(title_key)
        kept.append(rec)

    output_abs.parent.mkdir(parents=True, exist_ok=True)
    with output_abs.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return output_abs


def _build_generation_prompt(*, repo_root: Path, output_rel: str) -> str:
    prompt_path = repo_root / "prompts" / "idea_generation.md"
    if prompt_path.exists():
        base = prompt_path.read_text(encoding="utf-8")
    else:
        base = (
            "Generate at least 10 research idea records as JSONL. "
            "Each line must be an IdeaRecord-compatible JSON object."
        )
        log.warning("Idea generation prompt not found at %s; using fallback prompt.", prompt_path)

    instructions = [
        "",
        "Additional execution instructions:",
        f"- Write JSONL output to workspace-relative path: `{output_rel}`",
        "- Create parent directories if needed.",
        "- Ensure each line is a single JSON object with an `id` field.",
        "- Final response must be task_result JSON and include the generated file in `artifacts_created`.",
    ]
    return base.rstrip() + "\n" + "\n".join(instructions) + "\n"


def _generate_ideas_via_codex(*, ledger: Any, project: Dict[str, Any], cycle: int) -> Optional[str]:
    workspace = Path(project["repo_path"]).resolve()
    ts = utc_now_iso().replace(":", "").replace("-", "")
    output_rel = f"ideas/generated/{ts}_r{cycle:02d}.jsonl"
    prompt = _build_generation_prompt(repo_root=ledger.paths.root.resolve(), output_rel=output_rel)

    task = create_task_fn(
        ledger=ledger,
        project_id=str(project["id"]),
        task_type="codex_exec",
        spec={
            "cd": ".",
            "sandbox": "networking-off",
            "prompt": prompt,
            "append_schema": True,
        },
    )
    run = run_task_fn(ledger=ledger, project=project, task=task)
    status = str(((run.get("task") or {}).get("status") or "")).strip()
    if status != "success":
        raise RuntimeError(f"Idea generation task status={status or 'unknown'}")

    output_abs = (workspace / output_rel).resolve()
    if not output_abs.exists():
        raise RuntimeError(f"Idea generation output not found: {output_abs}")
    return output_rel


def _dedupe_generated_file(
    *,
    ledger: Any,
    project_id: str,
    input_rel: str,
    cycle: int,
) -> str:
    project = ledger.get_project(project_id)
    workspace = Path(project["repo_path"]).resolve()
    input_abs = (workspace / input_rel).resolve()
    output_rel = f"ideas/generated/deduped_r{cycle:02d}.jsonl"
    output_abs = (workspace / output_rel).resolve()
    mapping_rel = f"ideas/generated/deduped_r{cycle:02d}.mapping.json"

    if callable(dedupe_ideas_jsonl_fn):
        try:
            out = dedupe_ideas_jsonl_fn(
                ledger=ledger,
                project_id=project_id,
                input_path=input_rel,
                output_path=output_rel,
                mapping_path=mapping_rel,
                threshold=0.9,
            )
            return str(Path(out.get("output_path", output_rel)).resolve().relative_to(workspace).as_posix())
        except Exception as exc:  # noqa: BLE001
            log.warning("dedupe_ideas_jsonl failed (%s). Falling back to title-based dedupe.", exc)

    _simple_title_dedupe_jsonl(input_abs=input_abs, output_abs=output_abs)
    return output_rel


def _activate_top_k(*, ledger: Any, project_id: str, top_k: int) -> None:
    if top_k <= 0:
        return
    ideas = list_ideas_fn(ledger=ledger, project_id=project_id, limit=max(top_k, 1))
    for row in ideas[:top_k]:
        idea_id = str(row.get("id") or "").strip()
        if not idea_id:
            continue
        status = str(row.get("status") or "").strip()
        if status in {"selected", "smoke_passed", "done"}:
            continue
        try:
            set_idea_status_fn(ledger=ledger, idea_id=idea_id, status="active")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to activate idea %s: %s", idea_id, exc)


def _select_smoke_passed(*, ledger: Any, project_id: str) -> Tuple[Optional[str], Optional[str]]:
    smoke_passed = list_ideas_fn(ledger=ledger, project_id=project_id, status="smoke_passed", limit=500)
    if not smoke_passed:
        return None, None

    def _to_float(value: Any) -> Optional[float]:
        if isinstance(value, dict):
            for key in ("value", "total", "score"):
                if key in value:
                    return _to_float(value.get(key))
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _idea_score(row: Dict[str, Any]) -> float:
        score = _to_float(row.get("score_total"))
        if score is not None:
            return score
        data = row.get("data")
        if isinstance(data, dict):
            scores = data.get("scores")
            if isinstance(scores, dict):
                nested = _to_float(scores.get("total"))
                if nested is not None:
                    return nested
        return float("-inf")

    smoke_passed.sort(
        key=lambda row: (
            _idea_score(row),
            str(row.get("updated_at") or ""),
            str(row.get("id") or ""),
        ),
        reverse=True,
    )
    best = smoke_passed[0]
    idea_id = str(best.get("id") or "").strip()
    if not idea_id:
        return None, None
    set_idea_status_fn(ledger=ledger, idea_id=idea_id, status="selected")

    topic_out = write_topic_brief_fn(
        ledger=ledger,
        project_id=project_id,
        idea_id=idea_id,
        output_path="topic_brief.md",
        register_as_artifact=True,
        set_selected=False,
    )
    return idea_id, str(topic_out.get("output_path")) if topic_out.get("output_path") else None


def run_topic_engine(
    ledger: Any,
    project_id: str,
    rounds: int = 3,
    dry_run: bool = False,
    top_k: int = 10,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "cycles_run": 0,
        "ideas_generated": 0,
        "ideas_imported": 0,
        "ideas_scored": 0,
        "selected_idea_id": None,
        "topic_brief_path": None,
        "stopped_reason": "max_rounds",
    }

    try:
        project = ledger.get_project(project_id)
    except Exception as exc:  # noqa: BLE001
        log.error("Topic engine failed to load project %s: %s", project_id, exc)
        result["stopped_reason"] = "error"
        return result

    total_rounds = max(0, int(rounds))
    if total_rounds == 0:
        return result

    if dry_run:
        try:
            existing = list_ideas_fn(ledger=ledger, project_id=project_id, limit=1)
        except Exception as exc:  # noqa: BLE001
            log.warning("Dry-run precheck failed while listing ideas: %s", exc)
            existing = []
        if not existing:
            log.info("Dry-run mode: no existing ideas for project %s; loop skipped.", project_id)
            return result

    for cycle in range(1, total_rounds + 1):
        result["cycles_run"] += 1
        generated_rel: Optional[str] = None
        import_input_rel: Optional[str] = None

        if dry_run:
            log.info("Cycle %d: dry-run mode, skipping idea generation.", cycle)
        else:
            try:
                generated_rel = _generate_ideas_via_codex(ledger=ledger, project=project, cycle=cycle)
                if generated_rel:
                    workspace = Path(project["repo_path"]).resolve()
                    result["ideas_generated"] += _count_jsonl_rows((workspace / generated_rel).resolve())
            except Exception as exc:  # noqa: BLE001
                log.error("Cycle %d generation failed: %s", cycle, exc)

        if generated_rel:
            try:
                import_input_rel = _dedupe_generated_file(
                    ledger=ledger,
                    project_id=project_id,
                    input_rel=generated_rel,
                    cycle=cycle,
                )
            except Exception as exc:  # noqa: BLE001
                log.error("Cycle %d dedupe failed: %s", cycle, exc)
                import_input_rel = generated_rel

        if import_input_rel:
            try:
                imported = import_ideas_jsonl_fn(
                    ledger=ledger,
                    project_id=project_id,
                    input_path=import_input_rel,
                )
                result["ideas_imported"] += int(imported.get("imported") or 0)
            except Exception as exc:  # noqa: BLE001
                log.error("Cycle %d import failed: %s", cycle, exc)

        try:
            scored = score_ideas_fn(
                ledger=ledger,
                project_id=project_id,
                rubric_path="rubrics/idea_score_rubric.yaml",
                output_path=f"ideas/ranked_r{cycle:02d}.jsonl",
            )
            result["ideas_scored"] += int(scored.get("count") or 0)
        except Exception as exc:  # noqa: BLE001
            log.warning("Cycle %d score step failed: %s", cycle, exc)

        try:
            _activate_top_k(ledger=ledger, project_id=project_id, top_k=int(top_k))
        except Exception as exc:  # noqa: BLE001
            log.warning("Cycle %d activate-top-k step failed: %s", cycle, exc)

        try:
            selected_id, brief_path = _select_smoke_passed(ledger=ledger, project_id=project_id)
            if selected_id:
                result["selected_idea_id"] = selected_id
                result["topic_brief_path"] = brief_path
                result["stopped_reason"] = "selected_found"
                break
        except Exception as exc:  # noqa: BLE001
            log.error("Cycle %d selection step failed: %s", cycle, exc)

    return result
