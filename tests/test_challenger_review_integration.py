"""Tests for Challenger → Post-Exec Review integration.

Verifies that:
1. _build_challenger_section() in jobs.py correctly generates/omits the section
2. challenger_flags flow from iter_out through agent_loop into review job spec
"""
from __future__ import annotations

import pytest

from resorch.jobs import _build_challenger_section


class TestBuildChallengerSection:
    def test_flags_present(self) -> None:
        spec = {"challenger_flags": ["Causal claim not supported", "Multiple comparison issue"]}
        section = _build_challenger_section(spec)
        assert "INTERPRETATION CHALLENGER FLAGS" in section
        assert "MUST address" in section
        assert "Causal claim not supported" in section
        assert "Multiple comparison issue" in section

    def test_single_flag(self) -> None:
        spec = {"challenger_flags": ["Effect size too small"]}
        section = _build_challenger_section(spec)
        assert "Effect size too small" in section
        assert "END CHALLENGER FLAGS" in section

    def test_no_flags_key(self) -> None:
        spec = {"stage": "analysis"}
        section = _build_challenger_section(spec)
        assert section == ""

    def test_empty_list(self) -> None:
        spec = {"challenger_flags": []}
        section = _build_challenger_section(spec)
        assert section == ""

    def test_none_value(self) -> None:
        spec = {"challenger_flags": None}
        section = _build_challenger_section(spec)
        assert section == ""

    def test_not_a_list(self) -> None:
        spec = {"challenger_flags": "some string"}
        section = _build_challenger_section(spec)
        assert section == ""

    def test_list_with_empty_strings(self) -> None:
        spec = {"challenger_flags": ["", "", ""]}
        section = _build_challenger_section(spec)
        assert section == ""

    def test_mixed_valid_and_empty(self) -> None:
        spec = {"challenger_flags": ["Valid flag", "", "Another flag"]}
        section = _build_challenger_section(spec)
        assert "Valid flag" in section
        assert "Another flag" in section


class TestChallengerFlagsExtraction:
    """Test the challenger_flags extraction logic used in agent_loop.py."""

    @staticmethod
    def _extract_flags(iter_out: dict) -> list:
        """Mirror the extraction logic from agent_loop.py."""
        challenger_result = iter_out.get("interpretation_challenger") or {}
        challenger_flags_list: list = []
        if (
            challenger_result.get("enabled")
            and str(challenger_result.get("overall_concern_level") or "").strip().lower()
            in {"medium", "high"}
        ):
            _cflags = challenger_result.get("flags")
            if isinstance(_cflags, list) and _cflags:
                challenger_flags_list = [str(f) for f in _cflags[:10] if f]
        return challenger_flags_list

    def test_high_concern_with_flags(self) -> None:
        iter_out = {
            "interpretation_challenger": {
                "enabled": True,
                "overall_concern_level": "high",
                "flags": ["flag1", "flag2"],
            }
        }
        assert self._extract_flags(iter_out) == ["flag1", "flag2"]

    def test_medium_concern_with_flags(self) -> None:
        iter_out = {
            "interpretation_challenger": {
                "enabled": True,
                "overall_concern_level": "medium",
                "flags": ["flag1"],
            }
        }
        assert self._extract_flags(iter_out) == ["flag1"]

    def test_low_concern_excluded(self) -> None:
        iter_out = {
            "interpretation_challenger": {
                "enabled": True,
                "overall_concern_level": "low",
                "flags": ["flag1"],
            }
        }
        assert self._extract_flags(iter_out) == []

    def test_disabled_excluded(self) -> None:
        iter_out = {
            "interpretation_challenger": {
                "enabled": False,
                "overall_concern_level": "high",
                "flags": ["flag1"],
            }
        }
        assert self._extract_flags(iter_out) == []

    def test_no_challenger_result(self) -> None:
        assert self._extract_flags({}) == []

    def test_flags_truncated_to_10(self) -> None:
        iter_out = {
            "interpretation_challenger": {
                "enabled": True,
                "overall_concern_level": "high",
                "flags": [f"flag{i}" for i in range(15)],
            }
        }
        result = self._extract_flags(iter_out)
        assert len(result) == 10

    def test_empty_flags_list(self) -> None:
        iter_out = {
            "interpretation_challenger": {
                "enabled": True,
                "overall_concern_level": "high",
                "flags": [],
            }
        }
        assert self._extract_flags(iter_out) == []
