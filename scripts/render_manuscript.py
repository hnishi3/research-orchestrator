#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_PLACEHOLDER_KEY_RE = re.compile(r"{{\s*([A-Za-z0-9_.]+)\s*}}")
_UNRESOLVED_PLACEHOLDER_RE = re.compile(r"{{[^{}]+}}")
_VALID_PLACEHOLDER_KEY_RE = re.compile(r"^[A-Za-z0-9_.]+$")


def _safe_read_json(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw)


def _as_number(value: Any) -> float | None:
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
        return _as_number(value.get("mean"))
    return None


def _sanitize_segment(key: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(key)).strip("_")
    return text or "key"


def _set_mapping_value(mapping: Dict[str, Any], parts: Iterable[str], value: Any) -> None:
    segments = [p for p in parts if p]
    if not segments:
        return
    underscore_key = "_".join(segments)
    dot_key = ".".join(segments)
    mapping[underscore_key] = value
    mapping[dot_key] = value


def _flatten_json(value: Any, mapping: Dict[str, Any], parts: List[str]) -> None:
    if isinstance(value, dict):
        if not value:
            _set_mapping_value(mapping, parts, {})
            return
        for key, inner in value.items():
            _flatten_json(inner, mapping, [*parts, _sanitize_segment(str(key))])
        return
    if isinstance(value, list):
        _set_mapping_value(mapping, parts, value)
        for idx, inner in enumerate(value):
            _flatten_json(inner, mapping, [*parts, str(idx)])
        return
    _set_mapping_value(mapping, parts, value)


def _scoreboard_template_vars(scoreboard: Dict[str, Any]) -> Dict[str, Any]:
    vars_map: Dict[str, Any] = {}
    _flatten_json(scoreboard, vars_map, [])

    pm = scoreboard.get("primary_metric") if isinstance(scoreboard.get("primary_metric"), dict) else {}
    pm_name = str(pm.get("name") or "")
    pm_direction = str(pm.get("direction") or "")
    pm_current = _as_number(pm.get("current"))
    pm_baseline = _as_number(pm.get("baseline"))
    pm_delta = _as_number(pm.get("delta_vs_baseline"))

    if pm_delta is None and pm_current is not None and pm_baseline is not None:
        direction = pm_direction.strip().lower()
        if direction == "minimize":
            pm_delta = pm_baseline - pm_current
        else:
            pm_delta = pm_current - pm_baseline

    vars_map["primary_metric_name"] = pm_name
    vars_map["primary_metric_direction"] = pm_direction
    vars_map["primary_metric_current"] = pm_current
    vars_map["primary_metric_baseline"] = pm_baseline
    vars_map["delta_vs_baseline"] = pm_delta

    metrics = scoreboard.get("metrics")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            num = _as_number(value)
            vars_map[f"metric_{_sanitize_segment(str(key))}"] = num if num is not None else value

    test_pass_count = scoreboard.get("test_pass_count")
    if test_pass_count is None and isinstance(metrics, dict):
        test_pass_count = metrics.get("test_pass_count")
    test_fail_count = scoreboard.get("test_fail_count")
    if test_fail_count is None and isinstance(metrics, dict):
        test_fail_count = metrics.get("test_fail_count")

    vars_map["test_pass_count"] = _as_number(test_pass_count)
    vars_map["test_fail_count"] = _as_number(test_fail_count)

    return vars_map


def _load_result_json_vars(results_dir: Path) -> Dict[str, Any]:
    vars_map: Dict[str, Any] = {}
    if not results_dir.exists():
        return vars_map

    for json_path in sorted(results_dir.glob("*.json")):
        stem = _sanitize_segment(json_path.stem)
        try:
            payload = _safe_read_json(json_path)
        except (OSError, json.JSONDecodeError):
            continue
        vars_map[f"{stem}_json"] = json.dumps(payload, ensure_ascii=False)
        _flatten_json(payload, vars_map, [stem])

    return vars_map


def _to_template_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_manuscript(workspace_path: str | Path) -> str:
    workspace = Path(workspace_path).resolve()
    template_path = workspace / "paper" / "manuscript.template.md"
    if not template_path.exists():
        raise ValueError(f"Missing manuscript template: {template_path}")

    scoreboard_path = workspace / "results" / "scoreboard.json"
    if not scoreboard_path.exists():
        raise ValueError(f"Missing scoreboard: {scoreboard_path}")

    try:
        scoreboard_obj = _safe_read_json(scoreboard_path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in scoreboard: {scoreboard_path} ({exc})") from exc

    if not isinstance(scoreboard_obj, dict):
        raise ValueError(f"scoreboard.json must be a JSON object: {scoreboard_path}")

    template_vars: Dict[str, Any] = {}
    template_vars.update(_scoreboard_template_vars(scoreboard_obj))
    template_vars.update(_load_result_json_vars(workspace / "results"))

    template_text = template_path.read_text(encoding="utf-8")

    unknown: List[str] = []
    for match in _PLACEHOLDER_KEY_RE.finditer(template_text):
        key = match.group(1)
        if key not in template_vars:
            unknown.append(key)
    if unknown:
        missing = ", ".join(sorted(set(unknown)))
        raise ValueError(f"Unknown template key(s): {missing}")

    rendered = _PLACEHOLDER_KEY_RE.sub(lambda m: _to_template_string(template_vars[m.group(1)]), template_text)

    invalid_placeholders: List[str] = []
    for placeholder in sorted(set(_UNRESOLVED_PLACEHOLDER_RE.findall(rendered))):
        key = placeholder[2:-2].strip()
        if not _VALID_PLACEHOLDER_KEY_RE.fullmatch(key):
            invalid_placeholders.append(placeholder)
    if invalid_placeholders:
        bad = invalid_placeholders[0]
        raise ValueError(
            f"Invalid placeholder format: {bad} - keys may only contain alphanumeric, dot, and underscore characters"
        )

    unresolved = sorted(set(_UNRESOLVED_PLACEHOLDER_RE.findall(rendered)))
    if unresolved:
        preview = ", ".join(unresolved[:5])
        raise ValueError(f"Unresolved placeholder(s) remain: {preview}")

    out_path = workspace / "paper" / "manuscript.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return rendered


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render manuscript.md from manuscript.template.md and results JSON")
    parser.add_argument("workspace_path", help="Path to workspace")
    args = parser.parse_args(argv)
    try:
        render_manuscript(args.workspace_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
