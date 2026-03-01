"""``orchestrator doctor`` — environment and configuration diagnostics."""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List


def run_doctor(*, repo_root: Path) -> Dict[str, Any]:
    """Run all diagnostic checks and return a structured report."""
    checks: List[Dict[str, str]] = []

    # 1. Python version
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        checks.append({"name": "python_version", "status": "ok", "detail": f"{ver} (>= 3.10)"})
    else:
        checks.append({"name": "python_version", "status": "error", "detail": f"{ver} (requires >= 3.10)"})

    # 2-3. Required packages
    for pkg_import, pkg_name in [("yaml", "PyYAML"), ("jsonschema", "jsonschema")]:
        if importlib.util.find_spec(pkg_import) is not None:
            checks.append({"name": pkg_name, "status": "ok", "detail": "available"})
        else:
            checks.append({"name": pkg_name, "status": "error", "detail": "not installed (pip install research-orchestrator)"})

    # 4-5. External CLI tools
    for exe in ["codex", "claude"]:
        path = shutil.which(exe)
        if path:
            checks.append({"name": f"{exe}_cli", "status": "ok", "detail": path})
        else:
            checks.append({"name": f"{exe}_cli", "status": "warn", "detail": "not found in PATH"})

    # 6-7. API keys (report presence, never the value)
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        val = os.environ.get(key, "")
        if val:
            checks.append({"name": key, "status": "ok", "detail": "set"})
        else:
            checks.append({"name": key, "status": "info", "detail": "not set"})

    # 8. Ledger initialization
    db_path = repo_root / ".orchestrator" / "ledger.db"
    if db_path.exists():
        checks.append({"name": "ledger", "status": "ok", "detail": str(db_path)})
    else:
        checks.append({"name": "ledger", "status": "warn", "detail": "not initialized (run: orchestrator init)"})

    # 9-11. Config files
    for cfg_name in ["agent_loop.yaml", "review_policy.yaml", "pivot_policy.yaml"]:
        cfg_path = repo_root / "configs" / cfg_name
        if not cfg_path.exists():
            severity = "info" if cfg_name == "pivot_policy.yaml" else "warn"
            checks.append({"name": cfg_name, "status": severity, "detail": "not found (defaults will be used)"})
        else:
            try:
                import yaml  # noqa: F811

                raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                if raw is None or isinstance(raw, dict):
                    checks.append({"name": cfg_name, "status": "ok", "detail": "valid"})
                else:
                    checks.append({"name": cfg_name, "status": "warn", "detail": "expected YAML mapping, got other type"})
            except Exception as exc:  # noqa: BLE001
                checks.append({"name": cfg_name, "status": "warn", "detail": f"parse error: {exc}"})

    # Compute overall status
    severity_order = {"error": 3, "warn": 2, "info": 1, "ok": 0}
    worst = max(severity_order.get(c["status"], 0) for c in checks)
    overall = {3: "error", 2: "warn", 1: "ok", 0: "ok"}[worst]

    return {
        "status": overall,
        "python_version": ver,
        "platform": platform.platform(),
        "repo_root": str(repo_root),
        "checks": checks,
    }


def print_doctor_summary(result: Dict[str, Any]) -> None:
    """Print a human-readable summary to stderr."""
    status_labels = {"ok": "[ok]  ", "warn": "[warn]", "info": "[info]", "error": "[ERR] "}
    for c in result["checks"]:
        label = status_labels.get(c["status"], "[??]  ")
        print(f"  {label} {c['name']}: {c['detail']}", file=sys.stderr)

    counts = {}
    for c in result["checks"]:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    parts = []
    for s in ["ok", "info", "warn", "error"]:
        if counts.get(s, 0) > 0:
            parts.append(f"{counts[s]} {s}")
    print(f"\n  Status: {result['status']} ({', '.join(parts)})", file=sys.stderr)
