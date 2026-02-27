"""
scripts/dedupe_ideas.py

Idea candidates often contain near-duplicates.
This script provides a lightweight, dependency-free baseline deduper.

Algorithm (baseline):
- fingerprint: normalize(title + one_sentence_claim + novelty_statement)
- similarity: max(SequenceMatcher ratio, token Jaccard)
- merge: union evidence/venues, keep best-ish scores

Outputs:
- deduped ideas JSONL
- mapping JSON (old_id -> representative_id)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from resorch.idea_dedupe import dedupe_ideas


def load_jsonl(path: str) -> List[dict]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def save_jsonl(items: List[dict], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--mapping", default=None, help="Path to mapping JSON (default: <out>.mapping.json)")
    ap.add_argument("--threshold", type=float, default=0.9, help="Similarity threshold (default: 0.9)")
    args = ap.parse_args()

    ideas = load_jsonl(args.inp)

    deduped, mapping = dedupe_ideas(ideas, threshold=float(args.threshold))

    save_jsonl(deduped, args.out)
    mapping_path = Path(args.mapping) if args.mapping else Path(args.out).with_suffix(".mapping.json")
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Deduped {len(ideas)} -> {len(deduped)} ideas. Mapping: {mapping_path}")


if __name__ == "__main__":
    main()
