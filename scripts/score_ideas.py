#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""score_ideas.py

Usage:
  python scripts/score_ideas.py --input ideas.jsonl --rubric rubrics/idea_score_rubric.yaml --output ranked.jsonl

This is a lightweight baseline scorer.
In practice, you will fill `scores` using LLM-based rubric grading + smoke-test results.
"""

import argparse, json
from pathlib import Path

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML is required: pip install pyyaml")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()

def compute_total(scores, weights):
    total = 0.0
    total += weights["novelty"] * scores.get("novelty", 0)
    total += weights["feasibility"] * scores.get("feasibility", 0)
    total += weights["impact"] * scores.get("impact", 0)
    total += weights["clarity"] * scores.get("clarity", 0)
    total += weights["reusability"] * scores.get("reusability", 0)
    total -= weights["risk_penalty"] * scores.get("risk_penalty", 0)
    return float(total)

def main():
    args = parse_args()
    rubric = yaml.safe_load(Path(args.rubric).read_text(encoding="utf-8"))
    weights = rubric["weights"]

    ideas = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ideas.append(json.loads(line))

    for idea in ideas:
        scores = idea.get("scores", {}) or {}
        # If scores missing, set neutral defaults.
        for k in ["novelty","feasibility","impact","clarity","reusability","risk_penalty"]:
            scores.setdefault(k, 2.5)
        scores["total"] = compute_total(scores, weights)
        idea["scores"] = scores

    ideas.sort(key=lambda x: x.get("scores", {}).get("total", -1e9), reverse=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for idea in ideas:
            f.write(json.dumps(idea, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ideas)} ranked ideas to {out}")

if __name__ == "__main__":
    main()
