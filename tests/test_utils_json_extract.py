from __future__ import annotations

import pytest

from resorch.utils import extract_json_object


def test_extract_json_object_parses_full_json_object() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_parses_markdown_fence() -> None:
    text = "```json\n{\"a\": 1}\n```"
    assert extract_json_object(text) == {"a": 1}


def test_extract_json_object_parses_surrounded_json() -> None:
    text = "Here is the JSON:\n{\"a\": 1}\nThanks!"
    assert extract_json_object(text) == {"a": 1}


def test_extract_json_object_ignores_trailing_brace_noise() -> None:
    text = "{\"a\": 1}\nNote: {notjson}"
    assert extract_json_object(text) == {"a": 1}


def test_extract_json_object_skips_non_json_braces_before_object() -> None:
    text = "prefix {notjson}\n{\"a\": \"{stillstring}\", \"b\": 2}\nsuffix"
    assert extract_json_object(text) == {"a": "{stillstring}", "b": 2}


def test_extract_json_object_requires_json_object() -> None:
    with pytest.raises(ValueError):
        extract_json_object("[1, 2, 3]")

