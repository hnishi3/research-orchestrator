"""Tests for the three quick-fix items:

1. autopilot_pivot: assert → if-guard (safe under python -O)
2. codex_runner: assert → RuntimeError on missing pipes
3. artifacts: atomic overwrite via tempfile+replace
4. http_json: backoff capped at 300s
"""
from __future__ import annotations

import io
import json
import subprocess
from email.message import Message
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError

import pytest

from resorch.autopilot_pivot import _pivot_no_improvement_trigger
from resorch.codex_runner import run_codex_exec_jsonl
from resorch.artifacts import put_artifact
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.providers.http_json import HttpJsonError, request_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> Ledger:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo))
    ledger.init()
    return ledger


def _http_error(*, status: int, reason: str, body_text: str, headers: Optional[Dict[str, str]] = None) -> HTTPError:
    hdrs = Message()
    for k, v in (headers or {}).items():
        hdrs[k] = v
    fp = io.BytesIO(body_text.encode("utf-8"))
    return HTTPError(url="http://example.test", code=status, msg=reason, hdrs=hdrs, fp=fp)


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


# ---------------------------------------------------------------------------
# 1. pivot: assert replaced with if-guard
# ---------------------------------------------------------------------------

def test_pivot_ci_overlap_tolerates_none_ci_gracefully(tmp_path: Path) -> None:
    """Ensure the CI-overlap loop doesn't crash if a CI entry is somehow None.

    Before the fix this relied on `assert ci is not None` which would vanish
    under `python -O`.  The new code uses `if ... is None: continue`.

    We verify the *existing* behaviour (all-not-None) still triggers a pivot.
    """
    ledger = _make_repo(tmp_path)
    (ledger.paths.root / "configs").mkdir(exist_ok=True)
    (ledger.paths.root / "configs" / "pivot_policy.yaml").write_text(
        "policy_version: 1\n"
        "no_improvement:\n"
        "  enabled: true\n"
        "  metric_path: primary_metric.current.mean\n"
        "  direction: maximize\n"
        "  min_delta: 0.0\n"
        "  window_runs: 3\n"
        "  review_level: soft\n"
        "  use_ci_overlap: true\n",
        encoding="utf-8",
    )
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="analysis", git_init=False)
    ws = Path(project["repo_path"])

    # Overlapping CIs across 3 runs → should pivot.
    (ws / "results" / "scoreboard.json").write_text(
        json.dumps({
            "primary_metric": {
                "name": "acc",
                "direction": "maximize",
                "current": {"mean": 0.5, "ci_95": [0.48, 0.52]},
            },
            "runs": [
                {"primary_metric": {"current": {"mean": 0.5, "ci_95": [0.48, 0.52]}}},
                {"primary_metric": {"current": {"mean": 0.5, "ci_95": [0.48, 0.52]}}},
            ],
        }) + "\n",
        encoding="utf-8",
    )

    result = _pivot_no_improvement_trigger(repo_root=ledger.paths.root, workspace=ws)
    assert result is not None
    assert result[0] == "soft"
    assert "ci_overlap" in result[1]


# ---------------------------------------------------------------------------
# 2. codex_runner: assert → RuntimeError on missing stdin/stdout
# ---------------------------------------------------------------------------

def test_codex_runner_raises_on_missing_pipes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If Popen returns None for stdin/stdout, should raise RuntimeError, not AssertionError."""
    class FakePopen:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.stdin = None
            self.stdout = None
            self.returncode = -1

        def kill(self) -> None:
            pass

    monkeypatch.setattr("resorch.codex_runner.subprocess.Popen", FakePopen)

    jsonl = tmp_path / "out.jsonl"
    last_msg = tmp_path / "last_msg.txt"
    stderr = tmp_path / "stderr.txt"

    with pytest.raises(RuntimeError, match="stdin/stdout"):
        run_codex_exec_jsonl(
            prompt="test",
            cd=tmp_path,
            sandbox="read-only",
            model=None,
            config_overrides=[],
            jsonl_path=jsonl,
            last_message_path=last_msg,
            stderr_path=stderr,
            on_event=None,
        )


# ---------------------------------------------------------------------------
# 3. artifacts: atomic overwrite (tempfile + replace)
# ---------------------------------------------------------------------------

def test_put_artifact_overwrite_is_atomic(tmp_path: Path) -> None:
    """Overwrite mode should use tempfile+replace so the file is never partial."""
    ledger = _make_repo(tmp_path)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="intake", git_init=False)
    ws = Path(project["repo_path"])

    # First write.
    put_artifact(ledger=ledger, project=project, relative_path="data/test.txt", content="hello", mode="overwrite", kind="test")
    assert (ws / "data" / "test.txt").read_text(encoding="utf-8") == "hello"

    # Overwrite — should atomically replace.
    put_artifact(ledger=ledger, project=project, relative_path="data/test.txt", content="world", mode="overwrite", kind="test")
    assert (ws / "data" / "test.txt").read_text(encoding="utf-8") == "world"

    # No leftover .tmp files.
    tmp_files = list((ws / "data").glob("*.tmp"))
    assert tmp_files == [], f"Leftover temp files: {tmp_files}"


def test_put_artifact_append_still_works(tmp_path: Path) -> None:
    """Append mode should NOT use atomic replace (append is inherently non-atomic)."""
    ledger = _make_repo(tmp_path)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="intake", git_init=False)
    ws = Path(project["repo_path"])

    put_artifact(ledger=ledger, project=project, relative_path="log.txt", content="line1\n", mode="append", kind="log")
    put_artifact(ledger=ledger, project=project, relative_path="log.txt", content="line2\n", mode="append", kind="log")
    assert (ws / "log.txt").read_text(encoding="utf-8") == "line1\nline2\n"


def test_put_artifact_overwrite_cleans_tmp_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If write fails mid-way, temp file should be cleaned up."""
    ledger = _make_repo(tmp_path)
    project = create_project(ledger=ledger, project_id="p1", title="P1", domain="", stage="intake", git_init=False)
    ws = Path(project["repo_path"])

    # Pre-create existing file.
    (ws / "notes").mkdir(parents=True, exist_ok=True)
    (ws / "notes" / "keep.txt").write_text("original", encoding="utf-8")

    # Monkey-patch Path.replace to simulate a crash after write.
    original_replace = Path.replace

    def boom(self: Path, target: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", boom)

    with pytest.raises(OSError, match="disk full"):
        put_artifact(ledger=ledger, project=project, relative_path="notes/keep.txt", content="CORRUPTED", mode="overwrite", kind="test")

    # Original file should be untouched.
    assert (ws / "notes" / "keep.txt").read_text(encoding="utf-8") == "original"

    # No leftover .tmp files.
    tmp_files = list((ws / "notes").glob("*.tmp"))
    assert tmp_files == [], f"Leftover temp files: {tmp_files}"


# ---------------------------------------------------------------------------
# 4. http_json: backoff capped at 300s
# ---------------------------------------------------------------------------

def test_http_backoff_capped_at_300_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rate-limit backoff with no Retry-After header should never sleep > 300s."""
    monkeypatch.setenv("RESORCH_HTTP_RATE_LIMIT_RETRIES", "20")
    monkeypatch.setenv("RESORCH_HTTP_RETRY_BACKOFF_SEC", "1.0")

    sleeps: List[float] = []

    def fake_sleep(sec: float) -> None:
        sleeps.append(sec)

    call_count = {"n": 0}

    def fake_urlopen(_req: Any, timeout: int = 0) -> _FakeResp:
        call_count["n"] += 1
        if call_count["n"] <= 15:
            raise _http_error(status=429, reason="Too Many Requests", body_text="rl")
        return _FakeResp(b'{"ok": true}')

    monkeypatch.setattr("resorch.providers.http_json.time.sleep", fake_sleep)
    monkeypatch.setattr("resorch.providers.http_json.urllib.request.urlopen", fake_urlopen)

    result = request_json(method="GET", url="http://example.test")
    assert result == {"ok": True}

    # Key assertion: no sleep exceeds 300s.
    assert all(s <= 300.0 for s in sleeps), f"Sleep exceeded 300s cap: {sleeps}"
    # Without the cap, attempt 9 would be 1.0 * 2^9 = 512s.
    assert len(sleeps) >= 10, f"Expected at least 10 sleeps, got {len(sleeps)}"


def test_http_general_backoff_capped_at_300_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """General retry backoff (non-429) should also be capped at 300s."""
    monkeypatch.setenv("RESORCH_HTTP_RETRIES", "12")
    monkeypatch.setenv("RESORCH_HTTP_RETRY_BACKOFF_SEC", "1.0")

    sleeps: List[float] = []

    def fake_sleep(sec: float) -> None:
        sleeps.append(sec)

    call_count = {"n": 0}

    def fake_urlopen(_req: Any, timeout: int = 0) -> _FakeResp:
        call_count["n"] += 1
        if call_count["n"] <= 11:
            raise _http_error(status=500, reason="Server Error", body_text="err")
        return _FakeResp(b'{"ok": true}')

    monkeypatch.setattr("resorch.providers.http_json.time.sleep", fake_sleep)
    monkeypatch.setattr("resorch.providers.http_json.urllib.request.urlopen", fake_urlopen)

    result = request_json(method="GET", url="http://example.test")
    assert result == {"ok": True}
    assert all(s <= 300.0 for s in sleeps), f"Sleep exceeded 300s cap: {sleeps}"
