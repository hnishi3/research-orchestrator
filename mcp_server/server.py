#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

# Make the repo root importable when running as `python mcp_server/server.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from resorch.ledger import Ledger
from resorch.paths import resolve_repo_paths
from resorch.retrieval import fetch as ro_fetch
from resorch.retrieval import search as ro_search
from resorch.projects import get_project, list_projects
from resorch.tasks import create_task, get_task, list_tasks, run_task
from resorch.reviews import write_review_request
from resorch.artifacts import list_artifacts, put_artifact
from resorch.ideas import dedupe_ideas_jsonl, get_idea, list_ideas, set_idea_status
from resorch.evidence_store import add_evidence, get_evidence, list_evidence
from resorch.smoke_tests import ingest_smoke_test_result, list_smoke_tests
from resorch.topic_brief import write_topic_brief


def _tool_defs() -> List[Dict[str, Any]]:
    manifest_path = Path(__file__).with_name("tool_manifest.yaml")
    if yaml is None:
        # Keep the server usable even if PyYAML isn't installed.
        # (Schemas will be omitted in this fallback mode.)
        return [
            {"name": "search", "description": "Search across ledger + artifacts."},
            {"name": "fetch", "description": "Fetch a single item by id."},
        ]

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    tools = []
    for t in manifest.get("tools", []):
        tool: Dict[str, Any] = {
            "name": t.get("name"),
            "description": (t.get("description") or "").strip(),
        }
        if "input_schema" in t:
            tool["input_schema"] = t["input_schema"]
            tool["inputSchema"] = t["input_schema"]
        if "output_schema" in t:
            tool["output_schema"] = t["output_schema"]
            tool["outputSchema"] = t["output_schema"]
        tools.append(tool)
    return tools


def _jsonrpc_response(req_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(req_id: Any, message: str, code: int = -32000, data: Any = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _handle_call(ledger: Ledger, name: str, arguments: Dict[str, Any]) -> Any:
    if name == "search":
        return ro_search(
            ledger,
            query=str(arguments.get("query", "")),
            project_id=arguments.get("project_id"),
            kind=arguments.get("kind"),
            limit=int(arguments.get("limit", 10)),
        )
    if name == "fetch":
        return ro_fetch(ledger, id=str(arguments.get("id", "")))
    if name == "review.create_packet":
        project = get_project(ledger, str(arguments.get("project_id", "")))
        out = write_review_request(
            ledger=ledger,
            project=project,
            stage=str(arguments.get("stage", "")),
            mode="balanced",
            targets=list(arguments.get("targets") or []),
            questions=["Any major issues? Any missing baselines/related work?"],
            rubric=None,
            time_budget_minutes=None,
        )
        return {"packet_path": out["packet_path"]}
    if name == "ledger.list_projects":
        limit = int(arguments.get("limit", 50))
        projects = list_projects(ledger)[:limit]
        return {"projects": projects}
    if name == "ledger.get_project":
        return {"project": get_project(ledger, str(arguments.get("project_id", "")))}
    if name == "ledger.list_tasks":
        project_id = arguments.get("project_id")
        status = arguments.get("status")
        limit = int(arguments.get("limit", 200))
        tasks = list_tasks(ledger, project_id=project_id)
        if status:
            tasks = [t for t in tasks if t.get("status") == status]
        return {"tasks": tasks[:limit]}
    if name == "ledger.get_task":
        return {"task": get_task(ledger, str(arguments.get("task_id", "")))}
    if name == "artifact.list":
        project_id = str(arguments.get("project_id", ""))
        prefix = arguments.get("prefix")
        limit = int(arguments.get("limit", 200))
        return {"artifacts": list_artifacts(ledger, project_id=project_id, prefix=prefix, limit=limit)}
    if name == "artifact.get":
        project_id = str(arguments.get("project_id", ""))
        path = str(arguments.get("path", ""))
        return ro_fetch(ledger, id=f"artifact:{project_id}/{path}")
    if name == "artifact.put":
        project_id = str(arguments.get("project_id", ""))
        path = str(arguments.get("path", ""))
        content = str(arguments.get("content", ""))
        mode = str(arguments.get("mode", "overwrite"))
        kind = arguments.get("kind")
        project = get_project(ledger, project_id)
        artifact = put_artifact(ledger=ledger, project=project, relative_path=path, content=content, mode=mode, kind=kind)
        return {"artifact": artifact}
    if name == "task.create":
        project_id = str(arguments.get("project_id", ""))
        task_type = str(arguments.get("type", ""))
        spec = arguments.get("spec") or {}
        if not isinstance(spec, dict):
            raise ValueError("task.create: spec must be an object")
        return {"task": create_task(ledger=ledger, project_id=project_id, task_type=task_type, spec=spec)}
    if name == "task.run":
        if not os.environ.get("RESORCH_MCP_ALLOW_TASK_RUN"):
            raise ValueError("task.run is disabled by default. Set RESORCH_MCP_ALLOW_TASK_RUN=1 to enable.")
        task_id = str(arguments.get("task_id", ""))
        task = get_task(ledger, task_id)
        project = get_project(ledger, task["project_id"])
        result = run_task(ledger=ledger, project=project, task=task)
        return {"task": result["task"], "run": result["run"]}
    if name == "idea.list":
        project_id = str(arguments.get("project_id", ""))
        status = arguments.get("status")
        limit = int(arguments.get("limit", 50))
        return {"ideas": list_ideas(ledger=ledger, project_id=project_id, status=status, limit=limit)}
    if name == "idea.get":
        return {"idea": get_idea(ledger=ledger, idea_id=str(arguments.get("idea_id", "")))}
    if name == "idea.set_status":
        return {
            "idea": set_idea_status(
                ledger=ledger,
                idea_id=str(arguments.get("idea_id", "")),
                status=str(arguments.get("status", "")),
            )
        }
    if name == "idea.dedupe_jsonl":
        project_id = str(arguments.get("project_id", ""))
        input_path = str(arguments.get("input_path", ""))
        output_path = str(arguments.get("output_path", "ideas/deduped.jsonl"))
        mapping_path = arguments.get("mapping_path")
        threshold = float(arguments.get("threshold", 0.9))
        return dedupe_ideas_jsonl(
            ledger=ledger,
            project_id=project_id,
            input_path=input_path,
            output_path=output_path,
            mapping_path=mapping_path,
            threshold=threshold,
        )
    if name == "smoke.ingest":
        project_id = str(arguments.get("project_id", ""))
        result_path = str(arguments.get("result_path", ""))
        store_path = arguments.get("store_path")
        return ingest_smoke_test_result(
            ledger=ledger,
            project_id=project_id,
            result_path=result_path,
            store_path=store_path,
        )
    if name == "smoke.list":
        project_id = str(arguments.get("project_id", ""))
        idea_id = arguments.get("idea_id")
        limit = int(arguments.get("limit", 50))
        return {"smoke_tests": list_smoke_tests(ledger=ledger, project_id=project_id, idea_id=idea_id, limit=limit)}
    if name == "topic.brief":
        project_id = str(arguments.get("project_id", ""))
        idea_id = str(arguments.get("idea_id", ""))
        output_path = str(arguments.get("output_path", "topic_brief.md"))
        set_selected = bool(arguments.get("set_selected", False))
        out = write_topic_brief(
            ledger=ledger,
            project_id=project_id,
            idea_id=idea_id,
            output_path=output_path,
            set_selected=set_selected,
        )
        return {"output_path": out["output_path"]}
    if name == "evidence.add":
        project_id = str(arguments.get("project_id", ""))
        kind = str(arguments.get("kind", ""))
        title = str(arguments.get("title", ""))
        url = str(arguments.get("url", ""))
        summary = str(arguments.get("summary", ""))
        meta = arguments.get("meta")
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            raise ValueError("evidence.add: meta must be an object")
        return add_evidence(
            ledger=ledger,
            project_id=project_id,
            idea_id=arguments.get("idea_id"),
            kind=kind,
            title=title,
            url=url,
            summary=summary,
            retrieved_at=arguments.get("retrieved_at"),
            relevance=arguments.get("relevance"),
            meta=meta,
            output_path=arguments.get("output_path"),
        )
    if name == "evidence.list":
        project_id = str(arguments.get("project_id", ""))
        idea_id = arguments.get("idea_id")
        limit = int(arguments.get("limit", 50))
        return {"evidence": list_evidence(ledger=ledger, project_id=project_id, idea_id=idea_id, limit=limit)}
    if name == "evidence.get":
        return {"evidence": get_evidence(ledger=ledger, evidence_id=str(arguments.get("evidence_id", "")))}
    raise ValueError(f"Unknown tool: {name}")


def run_stdio_server(ledger: Ledger) -> None:
    tools = _tool_defs()

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps(_jsonrpc_error(None, f"Invalid JSON: {e}")) + "\n")
            sys.stdout.flush()
            continue

        req_id = msg.get("id")
        method = msg.get("method")

        # Non-JSON-RPC: allow calling the tools by method name directly.
        if method in ("search", "fetch", "review.create_packet"):
            try:
                params = msg.get("params") or {}
                result = _handle_call(ledger, method, params)
                if req_id is not None:
                    sys.stdout.write(json.dumps(_jsonrpc_response(req_id, result), ensure_ascii=False) + "\n")
                    sys.stdout.flush()
            except Exception as e:  # noqa: BLE001
                if req_id is not None:
                    sys.stdout.write(json.dumps(_jsonrpc_error(req_id, str(e)), ensure_ascii=False) + "\n")
                    sys.stdout.flush()
            continue

        # JSON-RPC / MCP-ish methods.
        if method == "initialize":
            if req_id is None:
                continue
            sys.stdout.write(
                json.dumps(
                    _jsonrpc_response(
                        req_id,
                        {
                            "serverInfo": {"name": "research-orchestrator", "version": "0.1.0"},
                            "capabilities": {"tools": True},
                        },
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            sys.stdout.flush()
            continue

        if method == "tools/list":
            if req_id is None:
                continue
            sys.stdout.write(json.dumps(_jsonrpc_response(req_id, {"tools": tools}), ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue

        if method == "tools/call":
            if req_id is None:
                continue
            params = msg.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                sys.stdout.write(json.dumps(_jsonrpc_error(req_id, "params.arguments must be an object"), ensure_ascii=False) + "\n")
                sys.stdout.flush()
                continue
            try:
                result = _handle_call(ledger, str(name), arguments)
                sys.stdout.write(json.dumps(_jsonrpc_response(req_id, result), ensure_ascii=False) + "\n")
                sys.stdout.flush()
            except Exception as e:  # noqa: BLE001
                sys.stdout.write(json.dumps(_jsonrpc_error(req_id, str(e)), ensure_ascii=False) + "\n")
                sys.stdout.flush()
            continue

        # Unknown method: ignore notification, error on request.
        if req_id is not None:
            sys.stdout.write(json.dumps(_jsonrpc_error(req_id, f"Unknown method: {method}"), ensure_ascii=False) + "\n")
            sys.stdout.flush()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Research Orchestrator MCP-ish server (stdio)")
    ap.add_argument("--repo-root", default=None, help="Path to orchestrator repo root (defaults to auto-detect)")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    paths = resolve_repo_paths(args.repo_root)
    ledger = Ledger(paths)
    ledger.init()
    run_stdio_server(ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
