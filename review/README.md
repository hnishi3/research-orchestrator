# Reviews

Templates for running external reviews (another model / human / future self) in a consistent format.

## Why

- Review findings must link back to artifacts; otherwise fixes get lost.
- Severity, confidence, and fix cost must be separated so priorities are clear.

## Recommended Workflow

1. The orchestrator generates `review_packet.md` (from `review/review_packet_template.md`).
2. The reviewer produces `review_result.json` (schema: `review/review_result.schema.json`).
3. The orchestrator converts findings into `review_fix` tasks and feeds them into stage gates.

## Review Layers

- **L0**: Automated checkers (lint, tests, schema, reproducibility)
- **L1**: Same model (self-consistency / self-critique)
- **L2**: Different model — lightweight check (goal alignment, interpretation challenger)
- **L3**: Different model — strong critique (post-exec research review, dual review on hard gates)
- **L4**: Human (irreversible decisions, visual inspection gate)
