from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from resorch.ideas import list_ideas as list_ideas_fn
from resorch.ledger import Ledger

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class Unknown:
    reason: str = "unknown"


TriBool = Union[bool, Unknown]


def _is_unknown(v: Any) -> bool:
    return isinstance(v, Unknown)


def _truthy(v: TriBool) -> TriBool:
    return v


def eval_expr(expr: str, env: Dict[str, Any]) -> TriBool:
    # Accept YAML-style booleans.
    env = dict(env)
    env.setdefault("true", True)
    env.setdefault("false", False)

    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        ident = expr.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ident):
            return Unknown(f"missing:{ident}")
        return Unknown("invalid_syntax")
    return _eval_node(node.body, env)


def _eval_node(node: ast.AST, env: Dict[str, Any]) -> TriBool:
    if isinstance(node, ast.Constant):
        return node.value  # type: ignore[return-value]
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]  # type: ignore[return-value]
        return Unknown(f"missing:{node.id}")
    if isinstance(node, ast.Attribute):
        base = _eval_node(node.value, env)
        if _is_unknown(base):
            return base  # type: ignore[return-value]
        if isinstance(base, dict):
            if node.attr in base:
                return base[node.attr]  # type: ignore[return-value]
            return Unknown(f"missing:{node.attr}")
        return Unknown("attr_on_non_object")
    if isinstance(node, ast.List):
        items: List[Any] = []
        for elt in node.elts:
            v = _eval_node(elt, env)
            if _is_unknown(v):
                return v  # type: ignore[return-value]
            items.append(v)
        return items  # type: ignore[return-value]
    if isinstance(node, ast.Tuple):
        items2: List[Any] = []
        for elt in node.elts:
            v = _eval_node(elt, env)
            if _is_unknown(v):
                return v  # type: ignore[return-value]
            items2.append(v)
        return tuple(items2)  # type: ignore[return-value]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        v = _eval_node(node.operand, env)
        if _is_unknown(v):
            return v  # type: ignore[return-value]
        return (not bool(v))  # type: ignore[return-value]
    if isinstance(node, ast.BoolOp):
        vals = [_eval_node(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            # False short-circuit; Unknown if no False and at least one Unknown.
            saw_unknown = False
            for v in vals:
                if _is_unknown(v):
                    saw_unknown = True
                    continue
                if not bool(v):
                    return False
            return Unknown("and_unknown") if saw_unknown else True
        if isinstance(node.op, ast.Or):
            saw_unknown = False
            for v in vals:
                if _is_unknown(v):
                    saw_unknown = True
                    continue
                if bool(v):
                    return True
            return Unknown("or_unknown") if saw_unknown else False
        return Unknown("unsupported_boolop")
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        if _is_unknown(left):
            return left  # type: ignore[return-value]
        cur = left
        for op, comp in zip(node.ops, node.comparators):
            right = _eval_node(comp, env)
            if _is_unknown(right):
                return right  # type: ignore[return-value]
            ok: bool
            if isinstance(op, ast.Eq):
                ok = cur == right
            elif isinstance(op, ast.NotEq):
                ok = cur != right
            elif isinstance(op, ast.Gt):
                ok = float(cur) > float(right)
            elif isinstance(op, ast.GtE):
                ok = float(cur) >= float(right)
            elif isinstance(op, ast.Lt):
                ok = float(cur) < float(right)
            elif isinstance(op, ast.LtE):
                ok = float(cur) <= float(right)
            elif isinstance(op, ast.In):
                ok = cur in right  # type: ignore[operator]
            elif isinstance(op, ast.NotIn):
                ok = cur not in right  # type: ignore[operator]
            else:
                return Unknown("unsupported_compare")
            if not ok:
                return False
            cur = right
        return True

    return Unknown(f"unsupported_node:{type(node).__name__}")


def compute_gate_env(*, ledger: Ledger, project_id: str) -> Dict[str, Any]:
    project = ledger.get_project(project_id)
    ws = Path(project["repo_path"]).resolve()

    constraints_yaml_exists = (ws / "constraints.yaml").exists()
    ideas = list_ideas_fn(ledger=ledger, project_id=project_id, limit=500)
    idea_count = len(ideas)

    # Aggregate idea-level minimums for the YAML gate rules.
    evidence_counts = []
    novelty_empty = False
    score_totals = []
    for it in ideas:
        data = it.get("data") or {}
        ev = data.get("evidence") or []
        if isinstance(ev, list):
            evidence_counts.append(len(ev))
        novelty = str(data.get("novelty_statement") or "")
        if novelty.strip() == "":
            novelty_empty = True
        scores = data.get("scores") or {}
        if isinstance(scores, dict) and "total" in scores:
            try:
                score_totals.append(float(scores.get("total")))
            except (TypeError, ValueError):
                pass

    evidence_count = min(evidence_counts) if evidence_counts else Unknown("no_evidence")
    novelty_statement = "" if novelty_empty else "ok"
    all_ideas_meet_minimums = bool(evidence_counts) and (min(evidence_counts) >= 3) and (not novelty_empty)

    # Reviews: consider a "redteam" review as completed if any review exists with stage == "redteam".
    reviews = ledger.list_reviews(project_id)
    redteam_done = any((r.get("stage") or "") == "redteam" for r in reviews)

    score_total_min = min(score_totals) if score_totals else Unknown("no_scores")

    smoke_rows = ledger.list_smoke_tests(project_id=project_id, limit=500)
    latest_by_idea: Dict[str, Dict[str, Any]] = {}
    for r in smoke_rows:
        iid = str(r.get("idea_id") or "")
        if not iid:
            continue
        if iid not in latest_by_idea:
            latest_by_idea[iid] = r

    verdicts = [str(r.get("verdict") or "") for r in latest_by_idea.values()]
    smoke_verdict: Any
    if not verdicts:
        smoke_verdict = Unknown("no_smoke_tests")
    elif "pass" in verdicts:
        smoke_verdict = "pass"
    elif "timeout" in verdicts:
        smoke_verdict = "timeout"
    elif "fail" in verdicts:
        smoke_verdict = "fail"
    else:
        smoke_verdict = Unknown("unknown_verdict")

    smoke_counts = {"pass": verdicts.count("pass"), "fail": verdicts.count("fail"), "timeout": verdicts.count("timeout")}

    return {
        "constraints_yaml_exists": constraints_yaml_exists,
        "idea_count": idea_count,
        "evidence_count": evidence_count,
        "novelty_statement": novelty_statement,
        "all_ideas_meet_minimums": all_ideas_meet_minimums,
        "smoke_test": {"verdict": smoke_verdict, "counts": smoke_counts},
        "redteam_review": {"completed": redteam_done},
        "score": {"total": score_total_min},
    }


def evaluate_transitions(*, config: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
    transitions = config.get("transitions") or {}
    out: Dict[str, Any] = {"env": env, "transitions": {}}

    for name, spec in transitions.items():
        spec = spec or {}
        auto_pass_if = list(spec.get("auto_pass_if") or [])
        auto_reject_if = list(spec.get("auto_reject_if") or [])
        requires = list(spec.get("requires") or [])
        manual_gate = bool(spec.get("manual_gate", False))

        checks: Dict[str, Any] = {"auto_pass_if": [], "auto_reject_if": [], "requires": [], "manual_gate": manual_gate}

        def _eval_list(exprs: List[str]) -> Tuple[List[Dict[str, Any]], bool, bool]:
            results = []
            any_true = False
            any_unknown = False
            for e in exprs:
                r = eval_expr(str(e), env)
                results.append({"expr": e, "result": (None if _is_unknown(r) else bool(r)), "unknown": (_is_unknown(r))})
                if _is_unknown(r):
                    any_unknown = True
                elif bool(r):
                    any_true = True
            return results, any_true, any_unknown

        checks["auto_reject_if"], any_reject_true, any_reject_unknown = _eval_list(auto_reject_if)
        checks["auto_pass_if"], all_pass_true, any_pass_unknown = _eval_list(auto_pass_if)
        # For pass_if, require all expressions true if provided; empty means no auto-pass rule.
        if auto_pass_if:
            all_pass_true = all((not c["unknown"]) and bool(c["result"]) for c in checks["auto_pass_if"])
        else:
            all_pass_true = False

        checks["requires"], _req_any_true, req_any_unknown = _eval_list(requires)
        req_all_true = all((not c["unknown"]) and bool(c["result"]) for c in checks["requires"]) if requires else True

        decision = "pending"
        if any_reject_true:
            decision = "auto_reject"
        elif any_reject_unknown:
            decision = "unknown"
        elif not req_all_true:
            decision = "blocked"
        elif manual_gate:
            decision = "manual"
        elif all_pass_true:
            decision = "auto_pass"
        elif any_pass_unknown or req_any_unknown:
            decision = "unknown"

        out["transitions"][name] = {"decision": decision, "checks": checks}

    return out


def load_stage_transitions(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise SystemExit("PyYAML is required for stage transitions: pip install pyyaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))
