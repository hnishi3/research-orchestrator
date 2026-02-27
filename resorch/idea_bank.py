from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from resorch.ledger import Ledger
from resorch.projects import create_project
from resorch.utils import utc_now_iso


def _parse_idea_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["data"] = json.loads(out.pop("data_json") or "{}")
    return out


def _parse_idea_edge_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["meta"] = json.loads(out.pop("meta_json") or "{}")
    return out


def _ensure_list_of_str(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        out: List[str] = []
        for it in v:
            if it is None:
                continue
            s = str(it).strip()
            if s:
                out.append(s)
        return out
    s2 = str(v).strip()
    return [s2] if s2 else []


def link_ideas(
    *,
    ledger: Ledger,
    src_idea_id: str,
    dst_idea_id: str,
    relation: str,
    reason: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Validate existence (raises if missing).
    ledger.get_idea(str(src_idea_id))
    ledger.get_idea(str(dst_idea_id))

    edge = ledger.insert_idea_edge(
        src_idea_id=str(src_idea_id),
        dst_idea_id=str(dst_idea_id),
        relation=str(relation).strip(),
        reason=str(reason).strip() if isinstance(reason, str) and reason.strip() else None,
        meta=meta or {},
    )
    return _parse_idea_edge_row(edge)


def park_idea(
    *,
    ledger: Ledger,
    idea_id: str,
    parked_reason: str,
    unblock_conditions: Optional[List[str]] = None,
    next_check_date: Optional[str] = None,
) -> Dict[str, Any]:
    row = _parse_idea_row(ledger.get_idea(str(idea_id)))
    data = row.get("data") if isinstance(row.get("data"), dict) else {}
    data = dict(data)
    data["status"] = "parked"
    data["parked_reason"] = str(parked_reason).strip()
    if unblock_conditions is not None:
        data["unblock_conditions"] = _ensure_list_of_str(unblock_conditions)
    if next_check_date is not None:
        data["next_check_date"] = str(next_check_date).strip()
    data["updated_at"] = utc_now_iso()

    ledger.upsert_idea(
        idea_id=str(row.get("id")),
        project_id=str(row.get("project_id")),
        status="parked",
        score_total=row.get("score_total"),
        data=data,
    )
    return _parse_idea_row(ledger.get_idea(str(idea_id)))


def revive_idea_to_new_project(
    *,
    ledger: Ledger,
    idea_id: str,
    new_project_id: str,
    new_project_title: str,
    domain: str = "",
    stage: str = "intake",
    git_init: bool = False,
    new_idea_id: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    parent = _parse_idea_row(ledger.get_idea(str(idea_id)))
    parent_data = parent.get("data") if isinstance(parent.get("data"), dict) else {}

    project = create_project(
        ledger=ledger,
        project_id=str(new_project_id),
        title=str(new_project_title),
        domain=str(domain),
        stage=str(stage),
        git_init=bool(git_init),
    )

    spawned_id = str(new_idea_id).strip() if isinstance(new_idea_id, str) and new_idea_id.strip() else uuid4().hex
    data = deepcopy(parent_data) if isinstance(parent_data, dict) else {}
    data["id"] = spawned_id
    data["status"] = "candidate"
    data["parent_idea_id"] = str(idea_id)
    data.setdefault("revived_at", utc_now_iso())
    data.pop("scores", None)

    ledger.upsert_idea(
        idea_id=spawned_id,
        project_id=str(project["id"]),
        status="candidate",
        score_total=None,
        data=data,
    )

    edge = link_ideas(
        ledger=ledger,
        src_idea_id=str(idea_id),
        dst_idea_id=spawned_id,
        relation="revive",
        reason=reason,
        meta={"new_project_id": str(project["id"])},
    )

    return {
        "project": project,
        "idea": _parse_idea_row(ledger.get_idea(spawned_id)),
        "edge": edge,
    }


def _apply_mutation_operator(
    *,
    rec: Dict[str, Any],
    operator: str,
    narrow_focus: Optional[str],
    broaden_scope: Optional[str],
    reframe: Optional[str],
    baseline_add: Optional[str],
    metric_from: Optional[str],
    metric_to: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    op = str(operator).strip()
    meta: Dict[str, Any] = {"operator": op}

    title = str(rec.get("title") or "").strip()
    if op == "narrow":
        focus = str(narrow_focus or "").strip()
        if not focus:
            raise SystemExit("idea spawn --operator narrow requires --focus")
        rec["title"] = f"{title} (narrow: {focus})" if title else f"(narrow: {focus})"
        tags = _ensure_list_of_str(rec.get("tags"))
        tags.append(f"narrow:{focus}")
        rec["tags"] = sorted(set(tags))
        meta["focus"] = focus
    elif op == "broaden":
        scope = str(broaden_scope or "").strip()
        if not scope:
            raise SystemExit("idea spawn --operator broaden requires --scope")
        rec["title"] = f"{title} (broaden: {scope})" if title else f"(broaden: {scope})"
        tags = _ensure_list_of_str(rec.get("tags"))
        tags.append(f"broaden:{scope}")
        rec["tags"] = sorted(set(tags))
        meta["scope"] = scope
    elif op == "reframe":
        rf = str(reframe or "").strip()
        if not rf:
            raise SystemExit("idea spawn --operator reframe requires --reframe")
        rec["title"] = f"{title} (reframe: {rf})" if title else f"(reframe: {rf})"
        rec["novelty_statement"] = rf
        meta["reframe"] = rf
    elif op == "baseline_add":
        b = str(baseline_add or "").strip()
        if not b:
            raise SystemExit("idea spawn --operator baseline_add requires --baseline")
        ep = rec.get("evaluation_plan")
        if not isinstance(ep, dict):
            ep = {}
        baselines = _ensure_list_of_str(ep.get("baselines"))
        if b not in baselines:
            baselines.append(b)
        ep["baselines"] = baselines
        rec["evaluation_plan"] = ep
        meta["baseline"] = b
    elif op == "metric_swap":
        to = str(metric_to or "").strip()
        if not to:
            raise SystemExit("idea spawn --operator metric_swap requires --to")
        frm = str(metric_from or "").strip()
        ep2 = rec.get("evaluation_plan")
        if not isinstance(ep2, dict):
            ep2 = {}
        metrics = _ensure_list_of_str(ep2.get("metrics"))
        if frm and frm in metrics:
            metrics = [to if m == frm else m for m in metrics]
        elif to not in metrics:
            metrics.append(to)
        ep2["metrics"] = metrics
        rec["evaluation_plan"] = ep2
        meta["from"] = frm or None
        meta["to"] = to
    else:
        raise SystemExit("idea spawn --operator must be one of: narrow, broaden, reframe, baseline_add, metric_swap")

    return op, meta


def spawn_idea(
    *,
    ledger: Ledger,
    parent_idea_id: str,
    operator: str,
    project_id: Optional[str] = None,
    new_idea_id: Optional[str] = None,
    reason: Optional[str] = None,
    narrow_focus: Optional[str] = None,
    broaden_scope: Optional[str] = None,
    reframe: Optional[str] = None,
    baseline_add: Optional[str] = None,
    metric_from: Optional[str] = None,
    metric_to: Optional[str] = None,
) -> Dict[str, Any]:
    parent = _parse_idea_row(ledger.get_idea(str(parent_idea_id)))
    parent_data = parent.get("data") if isinstance(parent.get("data"), dict) else {}
    if not isinstance(parent_data, dict):
        parent_data = {}

    spawned_id = str(new_idea_id).strip() if isinstance(new_idea_id, str) and new_idea_id.strip() else uuid4().hex
    data = deepcopy(parent_data)
    data["id"] = spawned_id
    data["status"] = "candidate"
    data["parent_idea_id"] = str(parent_idea_id)
    data.setdefault("spawned_at", utc_now_iso())
    data.pop("scores", None)

    op, op_meta = _apply_mutation_operator(
        rec=data,
        operator=operator,
        narrow_focus=narrow_focus,
        broaden_scope=broaden_scope,
        reframe=reframe,
        baseline_add=baseline_add,
        metric_from=metric_from,
        metric_to=metric_to,
    )

    dst_project_id = str(project_id).strip() if isinstance(project_id, str) and project_id.strip() else str(parent.get("project_id"))
    ledger.upsert_idea(
        idea_id=spawned_id,
        project_id=dst_project_id,
        status="candidate",
        score_total=None,
        data=data,
    )

    edge = link_ideas(
        ledger=ledger,
        src_idea_id=str(parent_idea_id),
        dst_idea_id=spawned_id,
        relation=op,
        reason=reason,
        meta=op_meta,
    )

    return {
        "idea": _parse_idea_row(ledger.get_idea(spawned_id)),
        "edge": edge,
    }


def build_idea_graph(
    *,
    ledger: Ledger,
    project_id: Optional[str] = None,
    root_idea_id: Optional[str] = None,
    max_nodes: int = 200,
    max_edges: int = 500,
) -> Dict[str, Any]:
    if max_nodes < 1:
        max_nodes = 1
    if max_edges < 1:
        max_edges = 1

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def _add_node(idea_row: Dict[str, Any]) -> None:
        rec = _parse_idea_row(idea_row)
        iid = str(rec.get("id") or "")
        if not iid:
            return
        data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
        nodes[iid] = {
            "id": iid,
            "project_id": str(rec.get("project_id") or ""),
            "status": str(rec.get("status") or ""),
            "title": str(data.get("title") or ""),
        }

    if project_id:
        ideas = ledger.list_ideas(project_id=str(project_id), limit=500)
        idea_ids = [str(r.get("id") or "") for r in ideas if r.get("id")]
        for r in ideas:
            _add_node(r)
        raw_edges = ledger.list_idea_edges(idea_ids=idea_ids, limit=max_edges)
        for e in raw_edges:
            edges.append(_parse_idea_edge_row(e))
        return {"nodes": list(nodes.values()), "edges": edges}

    if root_idea_id:
        root_id = str(root_idea_id)
        seen: Set[str] = set()
        frontier: List[str] = [root_id]
        while frontier and len(seen) < max_nodes and len(edges) < max_edges:
            cur_id = frontier.pop(0)
            if cur_id in seen:
                continue
            seen.add(cur_id)
            _add_node(ledger.get_idea(cur_id))
            raw_edges2 = ledger.list_idea_edges(idea_ids=[cur_id], limit=max_edges)
            for e in raw_edges2:
                if len(edges) >= max_edges:
                    break
                pe = _parse_idea_edge_row(e)
                edges.append(pe)
                src = str(pe.get("src_idea_id") or "")
                dst = str(pe.get("dst_idea_id") or "")
                for nid in (src, dst):
                    if nid and nid not in seen and nid not in frontier and len(seen) + len(frontier) < max_nodes:
                        frontier.append(nid)
        return {"nodes": list(nodes.values()), "edges": edges}

    raise SystemExit("idea graph requires either --project or --root")


def _dot_escape(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace("\"", "\\\"")


def format_idea_graph_dot(graph: Dict[str, Any]) -> str:
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    edges = graph.get("edges") if isinstance(graph, dict) else None
    nodes_list = nodes if isinstance(nodes, list) else []
    edges_list = edges if isinstance(edges, list) else []

    lines: List[str] = []
    lines.append("digraph idea_bank {\n")
    lines.append("  rankdir=LR;\n")
    for n in nodes_list:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id") or "")
        if not nid:
            continue
        label = f"{nid}\\n{n.get('title')}"
        lines.append(f"  \"{_dot_escape(nid)}\" [label=\"{_dot_escape(label)}\"];\n")
    for e in edges_list:
        if not isinstance(e, dict):
            continue
        src = str(e.get("src_idea_id") or "")
        dst = str(e.get("dst_idea_id") or "")
        if not src or not dst:
            continue
        rel = str(e.get("relation") or "")
        lines.append(f"  \"{_dot_escape(src)}\" -> \"{_dot_escape(dst)}\" [label=\"{_dot_escape(rel)}\"];\n")
    lines.append("}\n")
    return "".join(lines)
