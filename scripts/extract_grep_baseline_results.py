from __future__ import annotations

import json
import sys
import tempfile
import importlib.util
from pathlib import Path

sys.path.insert(0, ".")

TEST_MODULE_PATH = Path(__file__).resolve().parents[1] / "tests" / "test_checker_validation.py"
_spec = importlib.util.spec_from_file_location("test_checker_validation", TEST_MODULE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load {TEST_MODULE_PATH}")
_test_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_test_module)

ALL_ERROR_TYPES = _test_module.ALL_ERROR_TYPES
EXPECTED_CHECK_MAP = _test_module.EXPECTED_CHECK_MAP
PlantedErrorWorkspace = _test_module.PlantedErrorWorkspace
_simple_grep_baseline_detects = _test_module._simple_grep_baseline_detects
check_manuscript_consistency = _test_module.check_manuscript_consistency


def main() -> int:
    results = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for error_type in ALL_ERROR_TYPES:
            ws = PlantedErrorWorkspace(tmp_path / error_type)
            ws.plant_error(error_type)
            report = check_manuscript_consistency(ws.root)
            expected = EXPECTED_CHECK_MAP[error_type]
            checker_detected = any((not c.passed) and c.check_id == expected for c in report.checks)
            baseline_detected = _simple_grep_baseline_detects(ws.root, error_type)
            row = {
                "error": error_type,
                "checker": checker_detected,
                "grep_baseline": baseline_detected,
            }
            results.append(row)
            print(f"{error_type}: checker={checker_detected}, grep={baseline_detected}")

    checker_hits = sum(1 for r in results if r["checker"])
    grep_hits = sum(1 for r in results if r["grep_baseline"])
    total = len(results)
    checker_tpr = checker_hits / total if total else 0.0
    grep_tpr = grep_hits / total if total else 0.0

    print(f"\nChecker TPR: {checker_tpr:.2f} ({checker_hits}/{total})")
    print(f"Grep TPR: {grep_tpr:.2f} ({grep_hits}/{total})")
    print(f"Checker advantage: {checker_tpr - grep_tpr:+.2f}")

    output = {
        "per_error_results": results,
        "summary": {
            "checker_tpr": checker_tpr,
            "checker_hits": checker_hits,
            "grep_tpr": grep_tpr,
            "grep_hits": grep_hits,
            "total_errors": total,
            "checker_advantage": checker_tpr - grep_tpr,
        },
    }

    out_path = Path("../results/grep_baseline_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
