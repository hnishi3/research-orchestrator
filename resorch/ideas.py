from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.artifacts import register_artifact
from resorch.idea_dedupe import dedupe_ideas as dedupe_ideas_fn
from resorch.ledger import Ledger

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

try:
    from resorch.providers.claude_code_cli import (
        ClaudeCodeCliConfig,
        extract_structured_output,
        run_claude_code_print_json,
    )
except ImportError:  # pragma: no cover
    ClaudeCodeCliConfig = None  # type: ignore[assignment,misc]
    extract_structured_output = None  # type: ignore[assignment]
    run_claude_code_print_json = None  # type: ignore[assignment]


_IDEA_SCORE_AXES = ["novelty", "feasibility", "impact", "clarity", "reusability", "risk_penalty"]

_IDEA_SCORE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        axis: {"type": "number", "minimum": 0, "maximum": 5}
        for axis in _IDEA_SCORE_AXES
    },
    "required": _IDEA_SCORE_AXES,
}

_SCORE_PROMPT_TEMPLATE = """\
You are an expert research idea evaluator.

Score the following idea on each axis from 0 to 5 (integers or one-decimal floats).

## Scoring Rubric
- **novelty** (0-5): How original is the idea? 0 = well-known, 5 = highly novel.
- **feasibility** (0-5): How practical to execute? 0 = infeasible, 5 = straightforward.
- **impact** (0-5): Potential scientific or practical impact? 0 = negligible, 5 = transformative.
- **clarity** (0-5): How clearly is the idea articulated? 0 = vague, 5 = crystal clear.
- **reusability** (0-5): Can results/methods be reused in other projects? 0 = one-off, 5 = highly reusable.
- **risk_penalty** (0-5): How risky is execution? 0 = no risk, 5 = extremely risky.

## Idea
{idea_json}

Respond with a JSON object containing exactly these six keys with numeric values.
"""


def _score_idea_claude(rec: Dict[str, Any], workspace_dir: Path) -> Dict[str, float]:
    """Score a single idea using Claude Haiku via CLI. Returns dict of axis->score."""
    if run_claude_code_print_json is None:
        raise RuntimeError("claude_code_cli provider not available")

    idea_json = json.dumps(rec, ensure_ascii=False, indent=2)
    prompt = _SCORE_PROMPT_TEMPLATE.format(idea_json=idea_json)

    cfg = ClaudeCodeCliConfig(
        model="haiku",
        timeout_sec=120,
        tools="",
        allowed_tools="",
    )

    last_err: Optional[str] = None
    for _attempt in range(2):
        try:
            retry_hint = ""
            if last_err:
                retry_hint = f"\n\n(Previous attempt failed: {last_err}. Please output valid JSON.)\n"
            cli_json = run_claude_code_print_json(
                prompt=prompt + retry_hint,
                system_prompt=None,
                json_schema=_IDEA_SCORE_SCHEMA,
                workspace_dir=workspace_dir,
                config=cfg,
            )
            out = extract_structured_output(cli_json)
            if not isinstance(out, dict):
                raise ValueError("structured_output was not a dict")

            scores: Dict[str, float] = {}
            for axis in _IDEA_SCORE_AXES:
                try:
                    val = float(out[axis])
                    scores[axis] = max(0.0, min(5.0, val))
                except (KeyError, TypeError, ValueError):
                    scores[axis] = 2.5
            return scores
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)

    raise RuntimeError(f"Claude idea scoring failed after retries: {last_err}")


def _parse_idea_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["data"] = json.loads(out.pop("data_json") or "{}")
    return out


def import_ideas_jsonl(
    *,
    ledger: Ledger,
    project_id: str,
    input_path: str,
    register_as_artifact: bool = True,
) -> Dict[str, Any]:
    project = ledger.get_project(project_id)
    workspace = Path(project["repo_path"]).resolve()

    p = Path(input_path)
    if not p.is_absolute():
        p = (workspace / p).resolve()

    if not p.exists():
        raise SystemExit(f"Ideas file not found: {p}")

    raw_lines = p.read_text(encoding="utf-8").splitlines()
    imported = 0
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        idea_id = rec.get("id")
        if not idea_id:
            continue
        status = str(rec.get("status") or "candidate")
        score_total = None
        scores = rec.get("scores") or {}
        if isinstance(scores, dict) and "total" in scores:
            try:
                score_total = float(scores.get("total"))
            except (TypeError, ValueError):
                score_total = None
        ledger.upsert_idea(
            idea_id=str(idea_id),
            project_id=project_id,
            status=status,
            score_total=score_total,
            data=rec,
        )
        imported += 1

    artifact = None
    if register_as_artifact:
        try:
            rel = p.resolve().relative_to(workspace).as_posix()
        except ValueError:
            rel = None
        if rel is not None:
            artifact = register_artifact(ledger=ledger, project={"id": project_id, "repo_path": str(workspace)}, kind="ideas_jsonl", relative_path=rel, meta={})

    return {"imported": imported, "path": str(p), "artifact": artifact}


def list_ideas(
    *,
    ledger: Ledger,
    project_id: str,
    status: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    rows = ledger.list_ideas(project_id=project_id, status=status, limit=limit)
    return [_parse_idea_row(r) for r in rows]


def get_idea(*, ledger: Ledger, idea_id: str) -> Dict[str, Any]:
    row = ledger.get_idea(idea_id)
    return _parse_idea_row(row)


_IDEA_STATUSES = {"candidate", "active", "rejected", "smoke_passed", "selected", "in_progress", "parked", "done"}


def set_idea_status(*, ledger: Ledger, idea_id: str, status: str) -> Dict[str, Any]:
    status = str(status).strip()
    if status not in _IDEA_STATUSES:
        raise SystemExit(f"Invalid idea status: {status}. Expected one of: {', '.join(sorted(_IDEA_STATUSES))}")

    row = get_idea(ledger=ledger, idea_id=idea_id)
    data = row.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    data["status"] = status

    ledger.upsert_idea(
        idea_id=idea_id,
        project_id=str(row.get("project_id")),
        status=status,
        score_total=row.get("score_total"),
        data=data,
    )
    return get_idea(ledger=ledger, idea_id=idea_id)


def _compute_total(scores: Dict[str, Any], weights: Dict[str, Any]) -> float:
    total = 0.0
    total += float(weights["novelty"]) * float(scores.get("novelty", 0))
    total += float(weights["feasibility"]) * float(scores.get("feasibility", 0))
    total += float(weights["impact"]) * float(scores.get("impact", 0))
    total += float(weights["clarity"]) * float(scores.get("clarity", 0))
    total += float(weights["reusability"]) * float(scores.get("reusability", 0))
    total -= float(weights["risk_penalty"]) * float(scores.get("risk_penalty", 0))
    return float(total)


def score_ideas(
    *,
    ledger: Ledger,
    project_id: str,
    rubric_path: str,
    output_path: str,
    provider: str = "arithmetic",
    register_output_artifact: bool = True,
) -> Dict[str, Any]:
    if yaml is None:
        raise SystemExit("PyYAML is required for idea scoring: pip install pyyaml")

    project = ledger.get_project(project_id)
    workspace = Path(project["repo_path"]).resolve()

    rubric_p = Path(rubric_path)
    if not rubric_p.is_absolute():
        rubric_p = (ledger.paths.root / rubric_p).resolve()
    weights = yaml.safe_load(rubric_p.read_text(encoding="utf-8"))["weights"]

    use_claude = str(provider).strip().lower() == "claude"

    ideas = list_ideas(ledger=ledger, project_id=project_id, limit=500)
    updated: List[Dict[str, Any]] = []
    for row in ideas:
        rec = row["data"]
        scores = rec.get("scores") or {}
        if not isinstance(scores, dict):
            scores = {}

        if use_claude:
            try:
                llm_scores = _score_idea_claude(rec, workspace)
                scores.update(llm_scores)
            except Exception:  # noqa: BLE001  — fail-open
                for k in _IDEA_SCORE_AXES:
                    scores.setdefault(k, 2.5)
        else:
            for k in _IDEA_SCORE_AXES:
                scores.setdefault(k, 2.5)

        scores["total"] = _compute_total(scores, weights)
        rec["scores"] = scores

        ledger.upsert_idea(
            idea_id=str(rec.get("id")),
            project_id=project_id,
            status=str(rec.get("status") or "candidate"),
            score_total=float(scores["total"]),
            data=rec,
        )
        updated.append(rec)

    updated.sort(key=lambda r: float(((r.get("scores") or {}).get("total") or -1e9)), reverse=True)

    out_p = Path(output_path)
    if not out_p.is_absolute():
        out_p = (workspace / out_p).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for rec in updated:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    artifact = None
    if register_output_artifact:
        try:
            rel = out_p.resolve().relative_to(workspace).as_posix()
        except ValueError:
            rel = None
        if rel is not None:
            artifact = register_artifact(
                ledger=ledger,
                project={"id": project_id, "repo_path": str(workspace)},
                kind="ideas_ranked_jsonl",
                relative_path=rel,
                meta={"rubric_path": str(rubric_p)},
            )

    return {"count": len(updated), "output_path": str(out_p), "artifact": artifact}


def dedupe_ideas_jsonl(
    *,
    ledger: Ledger,
    project_id: str,
    input_path: str,
    output_path: str,
    mapping_path: Optional[str] = None,
    threshold: float = 0.9,
    register_output_artifacts: bool = True,
) -> Dict[str, Any]:
    project = ledger.get_project(project_id)
    workspace = Path(project["repo_path"]).resolve()

    inp = Path(input_path)
    if not inp.is_absolute():
        inp = (workspace / inp).resolve()
    if not inp.exists():
        raise SystemExit(f"Ideas file not found: {inp}")

    records: List[Dict[str, Any]] = []
    for line in inp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not isinstance(rec, dict):
            continue
        if not rec.get("id"):
            continue
        records.append(rec)

    deduped, mapping = dedupe_ideas_fn(records, threshold=threshold)

    out_p = Path(output_path)
    if not out_p.is_absolute():
        out_p = (workspace / out_p).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for rec in deduped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    map_p = Path(mapping_path) if mapping_path else out_p.with_suffix(".mapping.json")
    if not map_p.is_absolute():
        map_p = (workspace / map_p).resolve()
    map_p.parent.mkdir(parents=True, exist_ok=True)
    map_p.write_text(json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifact_out = None
    artifact_map = None
    if register_output_artifacts:
        try:
            _out_rel = out_p.resolve().relative_to(workspace).as_posix()
        except ValueError:
            _out_rel = None
        if _out_rel is not None:
            artifact_out = register_artifact(
                ledger=ledger,
                project={"id": project_id, "repo_path": str(workspace)},
                kind="ideas_deduped_jsonl",
                relative_path=_out_rel,
                meta={"threshold": float(threshold), "input_path": str(inp)},
            )
        try:
            _map_rel = map_p.resolve().relative_to(workspace).as_posix()
        except ValueError:
            _map_rel = None
        if _map_rel is not None:
            artifact_map = register_artifact(
                ledger=ledger,
                project={"id": project_id, "repo_path": str(workspace)},
                kind="ideas_dedupe_mapping",
                relative_path=_map_rel,
                meta={"threshold": float(threshold), "input_path": str(inp), "output_path": str(out_p)},
            )

    return {
        "before": len(records),
        "after": len(deduped),
        "output_path": str(out_p),
        "mapping_path": str(map_p),
        "artifact": artifact_out,
        "mapping_artifact": artifact_map,
    }
