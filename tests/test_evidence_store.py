from __future__ import annotations

from pathlib import Path
from urllib.error import URLError

import pytest

from resorch.artifacts import list_artifacts
from resorch import evidence_store
from resorch.evidence_store import add_evidence, get_evidence, list_evidence, validate_evidence_url
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_add_list_get_evidence(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )

    out = add_evidence(
        ledger=ledger,
        project_id=project["id"],
        kind="paper",
        title="Test Paper",
        url="https://example.com/paper",
        summary="Summary here.",
        relevance=0.9,
    )
    evid = out["evidence"]
    assert evid["project_id"] == "p1"
    assert evid["kind"] == "paper"
    assert evid["url"] == "https://example.com/paper"

    stored = Path(out["stored_path"])
    assert stored.exists()
    assert stored.parent.name == "evidence"

    arts = list_artifacts(ledger, project_id=project["id"], prefix="evidence/", limit=50)
    assert any(a["kind"] == "evidence_json" for a in arts)

    rows = list_evidence(ledger=ledger, project_id=project["id"], limit=10)
    assert len(rows) == 1
    got = get_evidence(ledger=ledger, evidence_id=evid["id"])
    assert got["title"] == "Test Paper"


def test_validate_evidence_url() -> None:
    ok = validate_evidence_url("https://example.com/paper")
    assert ok["valid"] is True
    assert ok["netloc"] == "example.com"

    bad = validate_evidence_url("example.com/paper")
    assert bad["valid"] is False
    assert "Invalid evidence URL" in bad["error"]


def test_add_evidence_rejects_invalid_url(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )

    with pytest.raises(SystemExit, match="Invalid evidence URL"):
        add_evidence(
            ledger=ledger,
            project_id=project["id"],
            kind="paper",
            title="Test Paper",
            url="not-a-valid-url",
            summary="Summary here.",
        )


def test_add_evidence_url_reachability_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyResponse:
        status = 204

        def __enter__(self) -> "_DummyResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def _ok_urlopen(req, timeout: float = 10.0):  # noqa: ANN001
        assert req.get_method() == "HEAD"
        assert timeout == 10.0
        return _DummyResponse()

    monkeypatch.setattr(evidence_store, "urlopen", _ok_urlopen)

    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )
    out = add_evidence(
        ledger=ledger,
        project_id=project["id"],
        kind="paper",
        title="Test Paper",
        url="https://example.com/paper",
        summary="Summary here.",
        validate_url_reachable=True,
    )
    assert out["evidence"]["meta"]["url_validation"] == {"reachable": True, "status_code": 204}


def test_add_evidence_url_reachability_failure_is_non_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail_urlopen(req, timeout: float = 10.0):  # noqa: ANN001
        assert req.get_method() == "HEAD"
        assert timeout == 10.0
        raise URLError("network down")

    monkeypatch.setattr(evidence_store, "urlopen", _fail_urlopen)

    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )
    out = add_evidence(
        ledger=ledger,
        project_id=project["id"],
        kind="paper",
        title="Test Paper",
        url="https://example.com/paper",
        summary="Summary here.",
        validate_url_reachable=True,
    )
    url_meta = out["evidence"]["meta"]["url_validation"]
    assert url_meta["reachable"] is False
    assert "network down" in url_meta["error"]
