from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

# Ensure the repo root (which contains the `resorch/` package) is importable under
# pytest's import modes.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resorch.ledger import Ledger  # noqa: E402
from resorch.paths import RepoPaths  # noqa: E402


def make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / f"repo_{uuid4().hex}"
    repo_root.mkdir(parents=True, exist_ok=False)
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


try:
    from hypothesis import settings

    settings.register_profile("ci", max_examples=50, derandomize=True)
    settings.register_profile("dev", max_examples=10)
    if os.environ.get("CI"):
        settings.load_profile("ci")
except ImportError:
    pass
