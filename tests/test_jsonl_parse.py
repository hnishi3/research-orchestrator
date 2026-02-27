from __future__ import annotations

from resorch.codex_runner import parse_jsonl_line


def test_parse_jsonl_line() -> None:
    assert parse_jsonl_line(" \n") is None
    assert parse_jsonl_line("{\"a\": 1}\n") == {"a": 1}

    bad = parse_jsonl_line("{not json}\n")
    assert bad is not None
    assert bad.get("_parse_error") is True
    assert "raw" in bad

