from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.ledger import Ledger

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


def _parse_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["rule"] = json.loads(out.pop("rule_json") or "{}")
    return out


def put_playbook_entry(
    *,
    ledger: Ledger,
    entry_path: str,
    topic: Optional[str] = None,
) -> Dict[str, Any]:
    if yaml is None:
        raise SystemExit("PyYAML is required for playbook operations: pip install pyyaml")

    p = Path(entry_path).expanduser()
    if not p.exists():
        raise SystemExit(f"Playbook entry file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Playbook entry must be a YAML object.")
    entry_id = data.get("id")
    if not entry_id:
        raise SystemExit("Playbook entry must include `id`.")

    inferred_topic = topic
    if inferred_topic is None:
        name = str(data.get("name") or "").strip()
        domain = str(data.get("domain") or "").strip()
        inferred_topic = f"{domain}:{name}".strip(":") or str(entry_id)

    row = ledger.upsert_playbook_entry(entry_id=str(entry_id), topic=inferred_topic, rule=data)
    return _parse_row(row)


def get_playbook_entry(ledger: Ledger, entry_id: str) -> Dict[str, Any]:
    return _parse_row(ledger.get_playbook_entry(entry_id))


def list_playbook_entries(ledger: Ledger, *, topic: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    return [_parse_row(r) for r in ledger.list_playbook_entries(topic=topic, limit=limit)]

