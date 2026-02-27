from __future__ import annotations

import json
import math
import re
import shutil
import textwrap
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

from resorch.manuscript_checker import CheckResult, ConsistencyReport, check_manuscript_consistency
from resorch.verification_checklist import generate_verification_checklist


CHECK_IDS: Tuple[str, ...] = (
    "fig_ref_exists",
    "fig_file_referenced",
    "fig_numbering_sequential",
    "table_ref_exists",
    "table_numbering_sequential",
    "claims_have_evidence",
    "evidence_urls_valid",
    "stats_have_effect_sizes",
    "stats_p_value_format",
    "refs_have_dois",
    "refs_no_placeholder",
    "scoreboard_exists",
    "scoreboard_has_primary_metric",
    "method_describes_metrics",
    "code_in_workspace",
    "text_numbers_match_scoreboard",
    "refs_citations_exist",
    "abstract_body_consistency",
    "stats_significance_claim_valid",
)


ALL_ERROR_TYPES: Tuple[str, ...] = (
    "plant_missing_figure_file",
    "plant_unreferenced_figure",
    "plant_nonsequential_figure_numbers",
    "plant_missing_table",
    "plant_nonsequential_table_numbers",
    "plant_claim_without_evidence",
    "plant_invalid_evidence_url",
    "plant_p_value_without_effect_size",
    "plant_bad_p_value_format",
    "plant_no_dois_in_refs",
    "plant_placeholder_refs",
    "plant_missing_scoreboard",
    "plant_null_primary_metric",
    "plant_missing_method_metrics",
    "plant_no_code",
    "plant_mismatched_numbers",
)


SPOT_ERROR_TYPES: Tuple[str, ...] = (
    "plant_spot_data_inconsistency_table_text",
    "plant_spot_data_inconsistency_abstract_body",
    "plant_spot_statistical_misuse",
    "plant_spot_missing_error_bars",
    "plant_spot_orphaned_citation",
)


EXPECTED_CHECK_MAP: Dict[str, str] = {
    "plant_missing_figure_file": "fig_ref_exists",
    "plant_unreferenced_figure": "fig_file_referenced",
    "plant_nonsequential_figure_numbers": "fig_numbering_sequential",
    "plant_missing_table": "table_ref_exists",
    "plant_nonsequential_table_numbers": "table_numbering_sequential",
    "plant_claim_without_evidence": "claims_have_evidence",
    "plant_invalid_evidence_url": "evidence_urls_valid",
    "plant_p_value_without_effect_size": "stats_have_effect_sizes",
    "plant_bad_p_value_format": "stats_p_value_format",
    "plant_no_dois_in_refs": "refs_have_dois",
    "plant_placeholder_refs": "refs_no_placeholder",
    "plant_missing_scoreboard": "scoreboard_exists",
    "plant_null_primary_metric": "scoreboard_has_primary_metric",
    "plant_missing_method_metrics": "method_describes_metrics",
    "plant_no_code": "code_in_workspace",
    "plant_mismatched_numbers": "text_numbers_match_scoreboard",
}

SPOT_EXPECTED_CHECK_MAP: Dict[str, str | None] = {
    "plant_spot_data_inconsistency_table_text": None,
    "plant_spot_data_inconsistency_abstract_body": "abstract_body_consistency",
    "plant_spot_statistical_misuse": "stats_significance_claim_valid",
    "plant_spot_missing_error_bars": None,
    "plant_spot_orphaned_citation": "refs_citations_exist",
}

NEW_SPOT_DETECTED_ERROR_TYPES: Tuple[str, ...] = (
    "plant_spot_orphaned_citation",
    "plant_spot_data_inconsistency_abstract_body",
    "plant_spot_statistical_misuse",
)


# These are coupled checks by design, not false positives.
DEPENDENT_FAILURES: Dict[str, Set[str]] = {
    "plant_missing_scoreboard": {"scoreboard_has_primary_metric", "text_numbers_match_scoreboard"},
}


def _checks_by_id(report: ConsistencyReport) -> Dict[str, CheckResult]:
    return {check.check_id: check for check in report.checks}


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Return a two-sided Wilson confidence interval for a Bernoulli proportion."""
    if trials <= 0:
        return (0.0, 0.0)
    p_hat = successes / trials
    z2_over_n = (z * z) / trials
    denom = 1.0 + z2_over_n
    center = p_hat + (z2_over_n / 2.0)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) / trials) + (z2_over_n / (4.0 * trials)))
    return ((center - margin) / denom, (center + margin) / denom)


class PlantedErrorWorkspace:
    """Helper that creates a synthetic workspace with specific planted errors."""

    _CLAIM_EVIDENCE_ID = "a" * 32

    def __init__(self, tmp_path: Path):
        self.root = tmp_path
        self._planted_errors: List[str] = []
        self._setup_clean_baseline()

    @property
    def manuscript_path(self) -> Path:
        return self.root / "paper" / "manuscript.md"

    @property
    def scoreboard_path(self) -> Path:
        return self.root / "results" / "scoreboard.json"

    @property
    def method_path(self) -> Path:
        return self.root / "notes" / "method.md"

    @property
    def claims_path(self) -> Path:
        return self.root / "claims" / "claim-1.md"

    @property
    def evidence_path(self) -> Path:
        return self.root / "evidence" / f"{self._CLAIM_EVIDENCE_ID}.json"

    def _write_text(self, relative_path: str, content: str) -> None:
        out_path = self.root / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")

    def _write_json(self, relative_path: str, payload: Dict[str, object]) -> None:
        self._write_text(relative_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _write_bytes(self, relative_path: str, content: bytes) -> None:
        out_path = self.root / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(content)

    def _read_manuscript(self) -> str:
        return self.manuscript_path.read_text(encoding="utf-8")

    def _write_manuscript(self, text: str) -> None:
        self.manuscript_path.parent.mkdir(parents=True, exist_ok=True)
        self.manuscript_path.write_text(text, encoding="utf-8")

    def _append_before_references(self, snippet: str) -> None:
        manuscript = self._read_manuscript()
        marker = "## References\n"
        if marker not in manuscript:
            self._write_manuscript(manuscript.rstrip() + "\n\n" + snippet.strip() + "\n")
            return
        prefix, suffix = manuscript.split(marker, 1)
        updated = f"{prefix.rstrip()}\n\n{snippet.strip()}\n\n{marker}{suffix.lstrip()}"
        self._write_manuscript(updated)

    def _replace_references(self, entries: List[str]) -> None:
        manuscript = self._read_manuscript()
        heading = "## References"
        if heading in manuscript:
            prefix = manuscript.split(heading, 1)[0].rstrip()
        else:
            prefix = manuscript.rstrip()
        rendered = "\n".join(f"{idx}. {entry}" for idx, entry in enumerate(entries, start=1))
        updated = f"{prefix}\n\n{heading}\n{rendered}\n"
        self._write_manuscript(updated)

    def _setup_clean_baseline(self) -> None:
        """Create a workspace where all consistency checks should pass."""
        self.root.mkdir(parents=True, exist_ok=True)

        manuscript = textwrap.dedent(
            """
            # Synthetic Manuscript

            ## Results
            Figure 1 shows the validation workflow.
            Figure 2 summarizes the final evaluation.
            Table 1 reports held-out metrics.

            ### Table 1
            | Metric | Value |
            | --- | --- |
            | Accuracy | 0.85 |

            The model achieved accuracy of 0.85 on the held-out set.
            The improvement was statistically significant (p = 0.032, Cohen's d = 0.41).

            ## References
            1. Alpha et al. Reliable benchmarking for reproducible AI. DOI:10.1234/alpha.2024.001
            2. Beta et al. Methods for controlled error injection. https://doi.org/10.2345/beta.2023.010
            """
        ).strip() + "\n"
        self._write_text("paper/manuscript.md", manuscript)

        self._write_json(
            "results/scoreboard.json",
            {
                "primary_metric": {
                    "name": "accuracy",
                    "direction": "maximize",
                    "current": 0.85,
                    "baseline": 0.80,
                    "best": 0.85,
                    "delta_vs_baseline": 0.05,
                },
                "metrics": {
                    "accuracy": 0.85,
                },
            },
        )

        method = textwrap.dedent(
            """
            # Method
            ## Metric Definitions
            - Primary metric: accuracy
            - Definition: fraction of correct predictions on held-out data.
            - Direction: maximize.
            - Baseline: 0.80.
            """
        ).strip() + "\n"
        self._write_text("notes/method.md", method)
        self._write_text("src/model.py", "def predict(x: float) -> float:\n    return x\n")

        self._write_bytes("results/fig/fig1.png", b"fake-png-1")
        self._write_bytes("results/fig/fig2.png", b"fake-png-2")

        claim = textwrap.dedent(
            f"""
            # Claim 1
            - claim: The approach improves held-out performance.
            - evidence_ids:
              - {self._CLAIM_EVIDENCE_ID}
            """
        ).strip() + "\n"
        self._write_text("claims/claim-1.md", claim)

        self._write_json(
            f"evidence/{self._CLAIM_EVIDENCE_ID}.json",
            {
                "id": self._CLAIM_EVIDENCE_ID,
                "url": "https://example.com/evidence/source",
                "title": "Evidence Source",
            },
        )

    def plant_error(self, error_type: str) -> "PlantedErrorWorkspace":
        """Plant a specific type of error into the workspace."""
        method = getattr(self, error_type, None)
        if method is None or not callable(method):
            raise ValueError(f"Unknown error type: {error_type}")
        method()
        self._planted_errors.append(error_type)
        return self

    def plant_missing_figure_file(self) -> None:
        self._append_before_references("Figure 3 reports the ablation study.")

    def plant_unreferenced_figure(self) -> None:
        self._write_bytes("results/fig/fig3.png", b"fake-png-3")

    def plant_nonsequential_figure_numbers(self) -> None:
        manuscript = self._read_manuscript().replace(
            "Figure 2 summarizes the final evaluation.",
            "Figure 3 summarizes the final evaluation.",
        )
        self._write_manuscript(manuscript)
        (self.root / "results" / "fig" / "fig2.png").unlink(missing_ok=True)
        self._write_bytes("results/fig/fig3.png", b"fake-png-3")

    def plant_missing_table(self) -> None:
        self._append_before_references("Table 2 reports ablation metrics.")

    def plant_nonsequential_table_numbers(self) -> None:
        self._append_before_references(
            textwrap.dedent(
                """
                Table 3 reports robustness diagnostics.

                ### Table 3
                | Metric | Value |
                | --- | --- |
                | Robustness | 0.77 |
                """
            ).strip()
        )

    def plant_claim_without_evidence(self) -> None:
        claim_without_evidence = textwrap.dedent(
            """
            # Claim 1
            - claim: The approach improves held-out performance.
            - evidence_ids:
              - (none)
            """
        ).strip() + "\n"
        self._write_text("claims/claim-1.md", claim_without_evidence)

    def plant_invalid_evidence_url(self) -> None:
        self._write_json(
            f"evidence/{self._CLAIM_EVIDENCE_ID}.json",
            {
                "id": self._CLAIM_EVIDENCE_ID,
                "url": "ftp://invalid.example.com/not-http",
                "title": "Invalid URL",
            },
        )

    def plant_p_value_without_effect_size(self) -> None:
        self._append_before_references("A secondary analysis was significant (p < 0.001) in held-out data.")

    def plant_bad_p_value_format(self) -> None:
        self._append_before_references(
            "A robustness check was significant (p = .000, d = 0.80) with strong practical impact."
        )

    def plant_no_dois_in_refs(self) -> None:
        self._replace_references(
            [
                "Alpha et al. Reliable benchmarking for reproducible AI.",
                "Beta et al. Methods for controlled error injection.",
            ]
        )

    def plant_placeholder_refs(self) -> None:
        self._replace_references(
            [
                "Alpha et al. Reliable benchmarking for reproducible AI. DOI:10.1234/alpha.2024.001",
                "Beta et al. Methods for controlled error injection. https://doi.org/10.2345/beta.2023.010",
                "TODO complete this citation. DOI:10.9999/placeholder.2026.001",
                "[?] pending source details. DOI:10.8888/placeholder.2026.002",
            ]
        )

    def plant_missing_scoreboard(self) -> None:
        self.scoreboard_path.unlink(missing_ok=True)

    def plant_null_primary_metric(self) -> None:
        payload = json.loads(self.scoreboard_path.read_text(encoding="utf-8"))
        payload.setdefault("primary_metric", {})["current"] = None
        self.scoreboard_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def plant_missing_method_metrics(self) -> None:
        self.method_path.unlink(missing_ok=True)

    def plant_no_code(self) -> None:
        shutil.rmtree(self.root / "src", ignore_errors=True)

    def plant_mismatched_numbers(self) -> None:
        manuscript = self._read_manuscript().replace(
            "accuracy of 0.85",
            "accuracy of 0.95",
        )
        self._write_manuscript(manuscript)

    def plant_spot_data_inconsistency_table_text(self) -> None:
        manuscript = self._read_manuscript()
        manuscript = manuscript.replace("| Accuracy | 0.85 |", "| Accuracy | 0.87 |")
        manuscript = manuscript.replace(
            "The model achieved accuracy of 0.85 on the held-out set.",
            "We achieved 95% accuracy on the held-out set.",
        )
        self._write_manuscript(manuscript)

    def plant_spot_data_inconsistency_abstract_body(self) -> None:
        manuscript = self._read_manuscript()
        manuscript = manuscript.replace(
            "## Results\n",
            "## Abstract\nWe improved performance by 40% over baseline.\n\n## Results\n",
            1,
        )
        manuscript = manuscript.replace(
            "The model achieved accuracy of 0.85 on the held-out set.",
            "The model achieved accuracy of 0.85 on the held-out set.\n"
            "In the main body, performance improved by 25% over baseline.",
        )
        self._write_manuscript(manuscript)

    def plant_spot_statistical_misuse(self) -> None:
        self._append_before_references(
            "A follow-up analysis was statistically significant (p = 0.07, Cohen's d = 0.12)."
        )

    def plant_spot_missing_error_bars(self) -> None:
        self._append_before_references(
            textwrap.dedent(
                """
                Table 2 summarizes mean outcomes over n=30 experiments.

                ### Table 2
                | Metric | Mean |
                | --- | --- |
                | Accuracy | 0.88 |
                | F1 | 0.81 |
                """
            ).strip()
        )

    def plant_spot_orphaned_citation(self) -> None:
        self._append_before_references("Prior work supports this setup [9].")


def _figure_file_numbers(workspace: Path) -> Set[int]:
    fig_dir = workspace / "results" / "fig"
    numbers: Set[int] = set()
    if not fig_dir.exists():
        return numbers
    for path in fig_dir.glob("*"):
        if not path.is_file():
            continue
        match = re.search(r"(\d+)", path.stem)
        if match:
            numbers.add(int(match.group(1)))
    return numbers


def _table_caption_numbers(text: str) -> Set[int]:
    return {int(m) for m in re.findall(r"^\s*(?:#{2,6}\s+)?Table\s+(\d+)\b", text, flags=re.IGNORECASE | re.MULTILINE)}


def _extract_references_section(text: str) -> str:
    marker = "## References"
    if marker not in text:
        return ""
    return text.split(marker, 1)[1]


def _simple_grep_baseline_detects(workspace: Path, error_type: str) -> bool:
    manuscript_path = workspace / "paper" / "manuscript.md"
    manuscript = manuscript_path.read_text(encoding="utf-8") if manuscript_path.exists() else ""
    lower = manuscript.lower()
    refs = _extract_references_section(manuscript).lower()
    fig_refs = {int(m) for m in re.findall(r"\b(?:figure|fig\.?)\s*(\d+)\b", manuscript, flags=re.IGNORECASE)}
    table_refs = {int(m) for m in re.findall(r"\btable\s+(\d+)\b", manuscript, flags=re.IGNORECASE)}
    fig_nums = _figure_file_numbers(workspace)
    table_caps = _table_caption_numbers(manuscript)

    if error_type == "plant_missing_figure_file":
        return bool(fig_refs and fig_nums and max(fig_refs) > max(fig_nums))
    if error_type == "plant_unreferenced_figure":
        return any(n not in fig_refs for n in fig_nums)
    if error_type == "plant_nonsequential_figure_numbers":
        return ("figure 1" in lower and "figure 3" in lower and "figure 2" not in lower)
    if error_type == "plant_missing_table":
        return bool(2 in table_refs and 2 not in table_caps)
    if error_type == "plant_nonsequential_table_numbers":
        return ("table 1" in lower and "table 3" in lower and "table 2" not in lower)
    if error_type == "plant_claim_without_evidence":
        claim_text = (workspace / "claims" / "claim-1.md").read_text(encoding="utf-8")
        return "evidence_ids" in claim_text and "(none)" in claim_text.lower()
    if error_type == "plant_invalid_evidence_url":
        evidence_text = (workspace / "evidence" / f"{PlantedErrorWorkspace._CLAIM_EVIDENCE_ID}.json").read_text(
            encoding="utf-8"
        )
        return '"url": "ftp://' in evidence_text
    if error_type == "plant_p_value_without_effect_size":
        return "p < 0.001" in lower and "cohen" not in lower and "effect size" not in lower
    if error_type == "plant_bad_p_value_format":
        return "p = .000" in lower
    if error_type == "plant_no_dois_in_refs":
        return bool(refs) and ("doi:" not in refs and "doi.org/" not in refs)
    if error_type == "plant_placeholder_refs":
        return "[?]" in refs or "todo" in refs
    if error_type == "plant_missing_scoreboard":
        return not (workspace / "results" / "scoreboard.json").exists()
    if error_type == "plant_null_primary_metric":
        score_path = workspace / "results" / "scoreboard.json"
        return '"current": null' in score_path.read_text(encoding="utf-8")
    if error_type == "plant_missing_method_metrics":
        return not (workspace / "notes" / "method.md").exists()
    if error_type == "plant_no_code":
        src_dir = workspace / "src"
        return (not src_dir.exists()) or (not list(src_dir.rglob("*.py")))
    if error_type == "plant_mismatched_numbers":
        score = json.loads((workspace / "results" / "scoreboard.json").read_text(encoding="utf-8"))
        expected = float(score["metrics"]["accuracy"])
        match = re.search(r"accuracy\s+of\s+([0-9]*\.[0-9]+|[0-9]+)", manuscript, flags=re.IGNORECASE)
        if not match:
            return False
        found = float(match.group(1))
        return abs(found - expected) > 0.01
    raise ValueError(f"Unhandled error type for baseline: {error_type}")


@pytest.mark.parametrize("error_type", ALL_ERROR_TYPES)
def test_single_planted_error_detection_and_isolation(tmp_path: Path, error_type: str) -> None:
    ws = PlantedErrorWorkspace(tmp_path / error_type)
    ws.plant_error(error_type)
    report = check_manuscript_consistency(ws.root)
    checks = _checks_by_id(report)

    expected_check = EXPECTED_CHECK_MAP[error_type]
    expected_failures = {expected_check} | DEPENDENT_FAILURES.get(error_type, set())

    assert expected_check in checks
    assert checks[expected_check].passed is False, f"Expected failure not detected for {error_type}: {expected_check}"

    missing_expected = [check_id for check_id in expected_failures if checks[check_id].passed]
    assert not missing_expected, f"Expected dependent failures not triggered for {error_type}: {missing_expected}"

    unexpected_failures = [check_id for check_id, check in checks.items() if (not check.passed) and check_id not in expected_failures]
    assert not unexpected_failures, f"Unexpected failures for {error_type}: {unexpected_failures}"


def test_planted_error_detection_rate(tmp_path: Path) -> None:
    """Measure overall true positive rate across all 16 planted errors."""
    results: List[Tuple[str, bool]] = []
    unexpected_false_positives = 0
    total_negative_checks = 0
    per_check_hits: Dict[str, int] = {check_id: 0 for check_id in CHECK_IDS}
    per_check_trials: Dict[str, int] = {check_id: 0 for check_id in CHECK_IDS}

    for error_type in ALL_ERROR_TYPES:
        ws = PlantedErrorWorkspace(tmp_path / error_type)
        ws.plant_error(error_type)
        report = check_manuscript_consistency(ws.root)
        failed_ids = {check.check_id for check in report.checks if not check.passed}

        expected = EXPECTED_CHECK_MAP[error_type]
        expected_failures = {expected} | DEPENDENT_FAILURES.get(error_type, set())
        detected = expected in failed_ids
        results.append((error_type, detected))

        per_check_trials[expected] += 1
        if detected:
            per_check_hits[expected] += 1

        for check_id in CHECK_IDS:
            if check_id in expected_failures:
                continue
            total_negative_checks += 1
            if check_id in failed_ids:
                unexpected_false_positives += 1

    true_positive_rate = sum(1 for _, detected in results if detected) / len(results)
    false_positive_rate = (
        unexpected_false_positives / total_negative_checks if total_negative_checks else 0.0
    )
    tpr_lower, tpr_upper = _wilson_interval(sum(1 for _, detected in results if detected), len(results))
    fpr_lower, fpr_upper = _wilson_interval(unexpected_false_positives, total_negative_checks)
    per_check_accuracy = {
        check_id: (per_check_hits[check_id] / per_check_trials[check_id]) if per_check_trials[check_id] else 0.0
        for check_id in CHECK_IDS
    }
    missed_errors = [error for error, detected in results if not detected]
    missed_checks = [check_id for check_id, acc in per_check_accuracy.items() if per_check_trials[check_id] and acc < 1.0]

    assert true_positive_rate >= 0.75, f"TPR {true_positive_rate:.2f} < 0.75; misses={missed_errors}"
    assert tpr_lower >= 0.75, (
        f"TPR Wilson 95% CI lower bound {tpr_lower:.3f} < 0.75; "
        f"point_estimate={true_positive_rate:.3f}; ci=({tpr_lower:.3f}, {tpr_upper:.3f}); misses={missed_errors}"
    )
    assert false_positive_rate <= 0.05, (
        f"FPR {false_positive_rate:.3f} > 0.05; "
        f"unexpected_false_positives={unexpected_false_positives}/{total_negative_checks}"
    )
    assert fpr_upper <= 0.05, (
        f"FPR Wilson 95% CI upper bound {fpr_upper:.3f} > 0.05; "
        f"point_estimate={false_positive_rate:.3f}; ci=({fpr_lower:.3f}, {fpr_upper:.3f}); "
        f"unexpected_false_positives={unexpected_false_positives}/{total_negative_checks}"
    )
    assert not missed_checks, f"Per-check detection misses: {missed_checks}; accuracies={per_check_accuracy}"


def test_clean_baseline_no_false_positives(tmp_path: Path) -> None:
    """A clean workspace with no errors should pass all checks."""
    ws = PlantedErrorWorkspace(tmp_path)
    report = check_manuscript_consistency(ws.root)
    false_positives = [check for check in report.checks if not check.passed]
    assert len(false_positives) == 0, f"False positives: {[check.check_id for check in false_positives]}"
    assert report.consistency_score == 1.0


def test_multiple_simultaneous_errors(tmp_path: Path) -> None:
    """Plant 5 errors simultaneously, verify all 5 are detected."""
    ws = PlantedErrorWorkspace(tmp_path)
    ws.plant_error("plant_missing_figure_file")
    ws.plant_error("plant_placeholder_refs")
    ws.plant_error("plant_p_value_without_effect_size")
    ws.plant_error("plant_missing_scoreboard")
    ws.plant_error("plant_no_code")
    report = check_manuscript_consistency(ws.root)
    failed_ids = {check.check_id for check in report.checks if not check.passed}
    expected_ids = {
        "fig_ref_exists",
        "refs_no_placeholder",
        "stats_have_effect_sizes",
        "scoreboard_exists",
        "code_in_workspace",
    }
    assert expected_ids.issubset(failed_ids), f"Missing detections: {expected_ids - failed_ids}"


def test_compare_to_grep_baseline(tmp_path: Path) -> None:
    """Compare checker to a simple grep baseline on same planted errors."""
    checker_hits = 0
    baseline_hits = 0
    baseline_results: List[Tuple[str, bool, bool]] = []

    for error_type in ALL_ERROR_TYPES:
        ws = PlantedErrorWorkspace(tmp_path / error_type)
        ws.plant_error(error_type)

        report = check_manuscript_consistency(ws.root)
        expected = EXPECTED_CHECK_MAP[error_type]
        checker_detected = any((not check.passed) and check.check_id == expected for check in report.checks)
        baseline_detected = _simple_grep_baseline_detects(ws.root, error_type)

        checker_hits += int(checker_detected)
        baseline_hits += int(baseline_detected)
        baseline_results.append((error_type, checker_detected, baseline_detected))

    misses_vs_baseline = [error for error, checker, baseline in baseline_results if baseline and not checker]
    assert baseline_hits > 0, "Simple grep baseline detected zero planted errors; baseline is not informative."
    assert checker_hits >= baseline_hits, (
        f"Checker ({checker_hits}) underperformed grep baseline ({baseline_hits}); "
        f"misses_vs_baseline={misses_vs_baseline}"
    )


def test_new_spot_checks_detect_planted_errors(tmp_path: Path) -> None:
    for error_type in NEW_SPOT_DETECTED_ERROR_TYPES:
        expected_check = SPOT_EXPECTED_CHECK_MAP[error_type]
        assert expected_check is not None

        ws = PlantedErrorWorkspace(tmp_path / error_type)
        ws.plant_error(error_type)
        report = check_manuscript_consistency(ws.root)
        checks = _checks_by_id(report)

        expected_failures = {expected_check} | DEPENDENT_FAILURES.get(error_type, set())
        assert checks[expected_check].passed is False, f"Expected SPOT detection missing: {error_type}->{expected_check}"

        missing_expected = [check_id for check_id in expected_failures if checks[check_id].passed]
        assert not missing_expected, f"Expected dependent failures not triggered for {error_type}: {missing_expected}"

        unexpected_failures = [
            check_id for check_id, check in checks.items() if (not check.passed) and check_id not in expected_failures
        ]
        assert not unexpected_failures, f"Unexpected failures for {error_type}: {unexpected_failures}"


def test_spot_inspired_error_detection(tmp_path: Path) -> None:
    """Measure whether existing checks surface SPOT-inspired errors not designed to match current rules."""
    assert set(SPOT_EXPECTED_CHECK_MAP.keys()) == set(SPOT_ERROR_TYPES)

    results: List[Tuple[str, bool, List[str]]] = []
    for error_type in SPOT_ERROR_TYPES:
        ws = PlantedErrorWorkspace(tmp_path / error_type)
        ws.plant_error(error_type)

        report = check_manuscript_consistency(ws.root)
        failed_ids = sorted(check.check_id for check in report.checks if not check.passed)
        expected_check = SPOT_EXPECTED_CHECK_MAP[error_type]
        detected = (expected_check in failed_ids) if expected_check else False
        results.append((error_type, detected, failed_ids))

    caught = [error_type for error_type, detected, _ in results if detected]
    missed = [error_type for error_type, detected, _ in results if not detected]
    detection_rate = (len(caught) / len(results)) if results else 1.0
    details = ", ".join(
        f"{error_type}:{'caught' if detected else 'missed'}(fails={failed_ids})"
        for error_type, detected, failed_ids in results
    )
    warnings.warn(
        (
            f"SPOT-inspired checker detection rate={detection_rate:.3f} ({len(caught)}/{len(results)}); "
            f"caught={caught}; missed={missed}; details={details}"
        ),
        UserWarning,
    )
    expected_caught = {error_type for error_type, check_id in SPOT_EXPECTED_CHECK_MAP.items() if check_id is not None}
    expected_missed = {error_type for error_type, check_id in SPOT_EXPECTED_CHECK_MAP.items() if check_id is None}
    assert set(caught) == expected_caught, f"Expected caught={sorted(expected_caught)}, got={caught}"
    assert set(missed) == expected_missed, f"Expected missed={sorted(expected_missed)}, got={missed}"
    assert detection_rate >= 0.5, f"SPOT detection rate too low: {detection_rate:.3f} ({len(caught)}/{len(results)})"


def test_verification_surface_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Measure verification surface coverage:
    fraction of planted errors surfaced by consistency report OR verification checklist.
    """
    monkeypatch.setattr(
        "resorch.verification_checklist._check_url",
        lambda _url, timeout=4.0: {"ok": True, "status_code": 200, "timeout": timeout},
    )

    all_errors = ALL_ERROR_TYPES + SPOT_ERROR_TYPES
    assert len(all_errors) == 21

    expected_check_map: Dict[str, str | None] = {}
    expected_check_map.update(EXPECTED_CHECK_MAP)
    expected_check_map.update(SPOT_EXPECTED_CHECK_MAP)

    baseline_ws = PlantedErrorWorkspace(tmp_path / "baseline")
    baseline_report = check_manuscript_consistency(baseline_ws.root)
    assert baseline_report.total_checks == len(CHECK_IDS) == 19
    baseline_failed_checks = {check.check_id for check in baseline_report.checks if not check.passed}
    baseline_checklist = generate_verification_checklist(baseline_ws.root, include_manuscript_checks=True)
    baseline_items = {item.id: (item.auto_status, item.auto_evidence) for item in baseline_checklist.items}

    surfaced_rows: List[Tuple[str, bool, bool, bool, List[str], List[str]]] = []
    for error_type in all_errors:
        ws = PlantedErrorWorkspace(tmp_path / error_type)
        ws.plant_error(error_type)

        report = check_manuscript_consistency(ws.root)
        failed_ids = sorted(check.check_id for check in report.checks if not check.passed)

        expected_check = expected_check_map.get(error_type)
        if expected_check is None:
            consistency_surfaced = bool(set(failed_ids) - baseline_failed_checks)
        else:
            consistency_surfaced = expected_check in failed_ids

        checklist = generate_verification_checklist(ws.root, include_manuscript_checks=True)
        changed_negative_items: List[str] = []
        for item in checklist.items:
            base_status, base_evidence = baseline_items.get(item.id, ("not_applicable", ""))
            changed = (item.auto_status != base_status) or (item.auto_evidence != base_evidence)
            if changed and item.auto_status in {"fail", "needs_human"}:
                changed_negative_items.append(item.id)
        checklist_surfaced = bool(changed_negative_items)

        combined_surfaced = consistency_surfaced or checklist_surfaced
        surfaced_rows.append(
            (
                error_type,
                consistency_surfaced,
                checklist_surfaced,
                combined_surfaced,
                failed_ids,
                sorted(changed_negative_items),
            )
        )

    consistency_hits = sum(1 for _, consistency_surfaced, _, _, _, _ in surfaced_rows if consistency_surfaced)
    checklist_hits = sum(1 for _, _, checklist_surfaced, _, _, _ in surfaced_rows if checklist_surfaced)
    combined_hits = sum(1 for _, _, _, combined_surfaced, _, _ in surfaced_rows if combined_surfaced)
    coverage = (combined_hits / len(all_errors)) if all_errors else 1.0
    missed = [error_type for error_type, _, _, combined_surfaced, _, _ in surfaced_rows if not combined_surfaced]

    surface_details = ", ".join(
        (
            f"{error_type}:consistency={consistency_surfaced},"
            f"checklist={checklist_surfaced},combined={combined_surfaced},"
            f"failed_checks={failed_ids},checklist_items={changed_items}"
        )
        for (
            error_type,
            consistency_surfaced,
            checklist_surfaced,
            combined_surfaced,
            failed_ids,
            changed_items,
        ) in surfaced_rows
    )
    warnings.warn(
        (
            f"verification_surface_coverage={coverage:.3f} ({combined_hits}/{len(all_errors)}); "
            f"consistency_hits={consistency_hits}; checklist_hits={checklist_hits}; missed={missed}; "
            f"details={surface_details}"
        ),
        UserWarning,
    )
    spot_consistency_hits = sum(
        1 for error_type, consistency_surfaced, _, _, _, _ in surfaced_rows if error_type in SPOT_ERROR_TYPES and consistency_surfaced
    )
    assert spot_consistency_hits >= 3, f"Expected at least 3/5 SPOT consistency detections, got {spot_consistency_hits}/5"
    assert consistency_hits >= 19, f"Expected at least 19 consistency detections, got {consistency_hits}/21"
    assert combined_hits >= 19, f"Expected at least 19 combined detections, got {combined_hits}/21"
