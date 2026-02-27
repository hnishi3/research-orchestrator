from __future__ import annotations

from pathlib import Path

import pytest

from resorch import evidence_store
from resorch.evidence_store import add_evidence, validate_evidence_url
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project


def _make_ledger(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def _make_project(ledger: Ledger) -> dict:
    return create_project(
        ledger=ledger,
        project_id="p1",
        title="Evidence Project",
        domain="",
        stage="intake",
        git_init=False,
    )


def test_add_evidence_rejects_url_without_scheme(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    project = _make_project(ledger)

    with pytest.raises(SystemExit, match="Invalid evidence URL"):
        add_evidence(
            ledger=ledger,
            project_id=project["id"],
            kind="paper",
            title="Paper",
            url="not-a-url",
            summary="Summary",
        )


def test_add_evidence_rejects_url_without_netloc(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    project = _make_project(ledger)

    with pytest.raises(SystemExit, match="Invalid evidence URL"):
        add_evidence(
            ledger=ledger,
            project_id=project["id"],
            kind="paper",
            title="Paper",
            url="https:///missing-hostname",
            summary="Summary",
        )


@pytest.mark.parametrize("url", ["http://example.com/paper", "https://example.com/paper"])
def test_add_evidence_accepts_valid_http_https_urls(tmp_path: Path, url: str) -> None:
    ledger = _make_ledger(tmp_path)
    project = _make_project(ledger)

    out = add_evidence(
        ledger=ledger,
        project_id=project["id"],
        kind="paper",
        title="Paper",
        url=url,
        summary="Summary",
    )

    assert out["evidence"]["url"] == url


def test_validate_evidence_url_valid_format() -> None:
    out = validate_evidence_url("https://example.org/resource")
    assert out["valid"] is True
    assert out["scheme"] == "https"
    assert out["netloc"] == "example.org"


def test_validate_evidence_url_invalid_format() -> None:
    out = validate_evidence_url("example.org/resource")
    assert out["valid"] is False
    assert "Invalid evidence URL" in str(out.get("error"))


def test_add_evidence_reachability_validation_uses_mocked_urlopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _DummyResponse:
        status = 200

        def __enter__(self) -> "_DummyResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def _mock_urlopen(req, timeout: float = 10.0):  # noqa: ANN001
        assert req.get_method() == "HEAD"
        assert timeout == 10.0
        return _DummyResponse()

    monkeypatch.setattr(evidence_store, "urlopen", _mock_urlopen)

    ledger = _make_ledger(tmp_path)
    project = _make_project(ledger)
    out = add_evidence(
        ledger=ledger,
        project_id=project["id"],
        kind="paper",
        title="Reachable URL",
        url="https://example.com",
        summary="Summary",
        validate_url_reachable=True,
    )

    assert out["evidence"]["meta"]["url_validation"] == {"reachable": True, "status_code": 200}

