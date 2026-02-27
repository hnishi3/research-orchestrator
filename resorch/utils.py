from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


def slugify(value: str) -> str:
    v = value.strip().lower()
    v = _SLUG_RE.sub("-", v)
    v = v.strip("-")
    return v or "project"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a single JSON object from a model output string.

    This is a best-effort helper to deal with common wrappers like markdown
    fences or surrounding prose.
    """

    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1]
            if "\n" in t:
                _, rest = t.split("\n", 1)
                t = rest
            t = t.strip()

    try:
        parsed = json.loads(t)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    decoder = json.JSONDecoder()
    idx = 0
    while True:
        idx = t.find("{", idx)
        if idx < 0:
            break
        try:
            obj, _end = decoder.raw_decode(t, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict):
            return obj
        idx += 1
    raise ValueError("No JSON object found in model output.")
