from __future__ import annotations

from resorch.cli import build_parser


def test_project_new_accepts_idea_id_argument() -> None:
    args = build_parser().parse_args(
        ["project", "new", "--title", "Demo", "--idea-id", "idea-42"]
    )
    assert args.idea_id == "idea-42"
