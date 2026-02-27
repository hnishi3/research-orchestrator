"""
scripts/make_review_packet.py

Small helper: given a project_id + target paths, assemble a review packet file.
This is intentionally tiny; orchestration logic lives elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime


def make_review_packet(project_id: str, stage: str, targets: list[str], out_path: str) -> str:
    ts = datetime.utcnow().isoformat() + "Z"
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Review Packet\n")
    lines.append(f"- Project: {project_id}\n- Stage: {stage}\n- Created: {ts}\n\n")
    lines.append("## Targets\n")
    for t in targets:
        lines.append(f"- {t}\n")
    p.write_text("".join(lines), encoding="utf-8")
    return str(p)


if __name__ == "__main__":
    # Minimal manual usage example
    packet = make_review_packet("demo_project", "analysis", ["paper/draft.md"], "review/packets/demo_project_stage_analysis.md")
    print(packet)
