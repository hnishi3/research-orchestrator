from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


_NON_WORD_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def idea_fingerprint(rec: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for k in ("title", "one_sentence_claim", "novelty_statement"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    if not parts:
        v2 = rec.get("id")
        if isinstance(v2, str):
            parts.append(v2)
    return normalize_text(" ".join(parts))


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()
    ta = set(a.split())
    tb = set(b.split())
    jac = 0.0
    if ta or tb:
        jac = (len(ta & tb) / float(len(ta | tb))) if (ta | tb) else 0.0
    return float(max(seq, jac))


_STATUS_PRIORITY = {
    "done": 60,
    "selected": 50,
    "smoke_passed": 40,
    "active": 35,
    "candidate": 30,
    "parked": 20,
    "rejected": 10,
}


def _pick_status(a: Optional[str], b: Optional[str]) -> Optional[str]:
    a = (a or "").strip() or None
    b = (b or "").strip() or None
    if a is None:
        return b
    if b is None:
        return a
    return a if _STATUS_PRIORITY.get(a, 0) >= _STATUS_PRIORITY.get(b, 0) else b


def _merge_unique_list(base: List[Any], other: Iterable[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for it in list(base) + list(other):
        key = repr(it)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _merge_evidence(base: Any, other: Any) -> List[Dict[str, Any]]:
    base_list = base if isinstance(base, list) else []
    other_list = other if isinstance(other, list) else []

    seen = set()
    merged: List[Dict[str, Any]] = []

    def _key(ev: Mapping[str, Any]) -> str:
        url = ev.get("url")
        if isinstance(url, str) and url.strip():
            return f"url:{url.strip()}"
        title = ev.get("title")
        if isinstance(title, str) and title.strip():
            return f"title:{title.strip().lower()}"
        return repr(dict(ev))

    for ev in base_list + other_list:
        if not isinstance(ev, dict):
            continue
        k = _key(ev)
        if k in seen:
            continue
        seen.add(k)
        merged.append(dict(ev))
    return merged


def merge_idea_records(base: MutableMapping[str, Any], other: Mapping[str, Any]) -> Dict[str, Any]:
    base_id = str(base.get("id") or "")
    other_id = str(other.get("id") or "")

    # Prefer non-empty text fields.
    for k in ("title", "one_sentence_claim", "novelty_statement"):
        bv = base.get(k)
        ov = other.get(k)
        if (not isinstance(bv, str) or not bv.strip()) and isinstance(ov, str) and ov.strip():
            base[k] = ov

    # Merge list fields.
    if "target_venues" in other and isinstance(other.get("target_venues"), list):
        base["target_venues"] = _merge_unique_list(list(base.get("target_venues") or []), other.get("target_venues") or [])

    base["evidence"] = _merge_evidence(base.get("evidence"), other.get("evidence"))

    # Merge status with a simple priority ordering.
    base["status"] = _pick_status(base.get("status"), other.get("status"))

    # Merge scores: keep max per key when both numeric.
    bs = base.get("scores")
    os = other.get("scores")
    if isinstance(os, dict):
        if not isinstance(bs, dict):
            bs = {}
        for k, v in os.items():
            if k not in bs:
                bs[k] = v
                continue
            try:
                bs[k] = max(float(bs[k]), float(v))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                # Prefer existing if non-numeric.
                pass
        base["scores"] = bs

    # Track merged ids for provenance.
    meta = base.get("_dedupe")
    if not isinstance(meta, dict):
        meta = {}
    merged_from = meta.get("merged_from")
    merged_ids = set(merged_from) if isinstance(merged_from, list) else set()
    if base_id:
        merged_ids.add(base_id)
    if other_id:
        merged_ids.add(other_id)
    meta["merged_from"] = sorted(merged_ids)
    base["_dedupe"] = meta

    return dict(base)


def dedupe_ideas(
    records: List[Dict[str, Any]],
    *,
    threshold: float = 0.9,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    reps: List[Dict[str, Any]] = []
    rep_fps: List[str] = []
    mapping: Dict[str, str] = {}

    for rec in records:
        if not isinstance(rec, dict):
            continue
        rec_id = str(rec.get("id") or "").strip()
        if not rec_id:
            continue

        fp = idea_fingerprint(rec)
        best_idx: Optional[int] = None
        best_score = 0.0
        for i, rep_fp in enumerate(rep_fps):
            s = similarity(fp, rep_fp)
            if s > best_score:
                best_score = s
                best_idx = i

        if best_idx is not None and best_score >= float(threshold):
            rep = reps[best_idx]
            rep_id = str(rep.get("id") or "").strip() or rec_id
            mapping[rec_id] = rep_id
            reps[best_idx] = merge_idea_records(rep, rec)
            rep_fps[best_idx] = idea_fingerprint(reps[best_idx])
            continue

        reps.append(dict(rec))
        rep_fps.append(fp)
        mapping[rec_id] = rec_id

    return reps, mapping
