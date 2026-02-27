from __future__ import annotations

import json
from pathlib import Path

import pytest

import resorch.jobs as jobs_mod
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.artifacts import list_artifacts


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_openai_job_run_and_poll(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )

    class DummyOpenAI:
        def __init__(self) -> None:
            self.polled = False

        def responses_create(self, payload: dict) -> dict:
            assert payload["model"] == "gpt-5.2"
            return {"id": "resp_123", "status": "in_progress"}

        def responses_get(self, response_id: str) -> dict:
            assert response_id == "resp_123"
            self.polled = True
            return {"id": "resp_123", "status": "completed", "output_text": "done"}

    dummy = DummyOpenAI()

    def fake_from_env(cls):  # noqa: ANN001
        return dummy

    monkeypatch.setattr(jobs_mod.OpenAIClient, "from_env", classmethod(fake_from_env))

    job = jobs_mod.create_job(
        ledger=ledger,
        project_id=project["id"],
        provider="openai",
        kind="response",
        spec={
            "payload": {"model": "gpt-5.2", "input": "hi", "background": True},
            "artifact_path": "notes/openai_response.json",
        },
    )

    run1 = jobs_mod.run_job(ledger=ledger, job_id=job["id"])
    assert run1["remote_id"] == "resp_123"
    assert run1["status"] in {"submitted", "running"}

    ws = Path(project["repo_path"])
    saved = ws / "notes" / "openai_response.json"
    assert saved.exists()

    polled = jobs_mod.poll_job(ledger=ledger, job_id=job["id"])
    assert dummy.polled is True
    assert polled["status"] == "succeeded"

    arts = list_artifacts(ledger, project_id=project["id"], prefix="notes/", limit=200)
    assert any(a["path"] == "notes/openai_response.json" for a in arts)


def test_openai_job_run_non_background_nonterminal_marks_submitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="pnb",
        title="PNB",
        domain="",
        stage="intake",
        git_init=False,
    )

    class DummyOpenAI:
        def responses_create(self, payload: dict) -> dict:
            assert payload["background"] is False
            return {"id": "resp_nb", "status": "in_progress"}

        def responses_get(self, response_id: str) -> dict:
            assert response_id == "resp_nb"
            return {"id": "resp_nb", "status": "completed", "output_text": "done"}

    def fake_from_env(cls):  # noqa: ANN001
        return DummyOpenAI()

    monkeypatch.setattr(jobs_mod.OpenAIClient, "from_env", classmethod(fake_from_env))

    job = jobs_mod.create_job(
        ledger=ledger,
        project_id=project["id"],
        provider="openai",
        kind="response",
        spec={"payload": {"model": "gpt-5.2", "input": "hi", "background": False}},
    )
    out = jobs_mod.run_job(ledger=ledger, job_id=job["id"])
    assert out["status"] == "submitted"

    polled = jobs_mod.poll_job(ledger=ledger, job_id=job["id"])
    assert polled["status"] == "succeeded"


def test_openai_deep_research_job_saves_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="pdr",
        title="PDR",
        domain="",
        stage="intake",
        git_init=False,
    )

    class DummyOpenAI:
        def responses_create(self, payload: dict) -> dict:
            assert payload["model"] == "o3-deep-research"
            assert payload["input"] == "Find 1 paper about X."
            assert payload["background"] is True
            assert payload["tools"][0]["type"] == "web_search_preview"
            return {"id": "resp_dr", "status": "completed", "output_text": "report\n"}

    def fake_from_env(cls):  # noqa: ANN001
        return DummyOpenAI()

    monkeypatch.setattr(jobs_mod.OpenAIClient, "from_env", classmethod(fake_from_env))

    job = jobs_mod.create_job(
        ledger=ledger,
        project_id=project["id"],
        provider="openai",
        kind="deep_research",
        spec={
            "query": "Find 1 paper about X.",
            "artifact_path": "notes/deep_research.md",
            "artifact_format": "text",
        },
    )
    out = jobs_mod.run_job(ledger=ledger, job_id=job["id"])
    assert out["status"] == "succeeded"

    ws = Path(project["repo_path"])
    saved = ws / "notes" / "deep_research.md"
    assert saved.read_text(encoding="utf-8") == "report\n"


def test_anthropic_review_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p2",
        title="P2",
        domain="",
        stage="intake",
        git_init=False,
    )

    class DummyAnthropic:
        def messages_create(self, **kwargs):  # noqa: ANN003
            # Return a JSON review result in a text block.
            review = {
                "recommendation": "minor",
                "reviewer": "claude",
                "findings": [
                    {
                        "severity": "minor",
                        "category": "writing",
                        "message": "Tighten the research question.",
                        "target_paths": ["notes/problem.md"],
                    }
                ],
            }
            return {"content": [{"type": "text", "text": json.dumps(review)}]}

    def fake_from_env(cls):  # noqa: ANN001
        return DummyAnthropic()

    monkeypatch.setattr(jobs_mod.AnthropicClient, "from_env", classmethod(fake_from_env))

    job = jobs_mod.create_job(
        ledger=ledger,
        project_id=project["id"],
        provider="anthropic",
        kind="review",
        spec={
            "stage": "intake",
            "targets": ["notes/problem.md"],
            "questions": ["Is the question clear?"],
            "include_target_contents": True,
        },
    )

    out = jobs_mod.run_job(ledger=ledger, job_id=job["id"])
    assert out["status"] == "succeeded"
    # kind="review" (research review) → no review_fix tasks created.
    assert out["result"]["ingested"]["tasks_created"] == 0

    tasks = ledger.list_tasks(project_id=project["id"])
    assert len(tasks) == 0


def test_openai_review_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p3",
        title="P3",
        domain="",
        stage="intake",
        git_init=False,
    )

    class DummyOpenAI:
        def responses_create(self, payload: dict) -> dict:
            # Ensure function tools are requested.
            assert payload["model"] == "gpt-5.2-pro"
            assert payload["tools"][0]["type"] == "function"
            return {
                "id": "resp_review",
                "status": "completed",
                "output": [
                    {
                        "type": "tool_call",
                        "name": "submit_review",
                        "arguments": {
                            "recommendation": "minor",
                            "findings": [
                                {
                                    "severity": "minor",
                                    "category": "writing",
                                    "message": "Tighten the research question.",
                                    "target_paths": ["notes/problem.md"],
                                }
                            ],
                        },
                    }
                ],
            }

        def responses_get(self, response_id: str) -> dict:
            raise AssertionError("responses_get should not be called for completed responses")

    def fake_from_env(cls):  # noqa: ANN001
        return DummyOpenAI()

    monkeypatch.setattr(jobs_mod.OpenAIClient, "from_env", classmethod(fake_from_env))

    job = jobs_mod.create_job(
        ledger=ledger,
        project_id=project["id"],
        provider="openai",
        kind="review",
        spec={
            "stage": "intake",
            "targets": ["notes/problem.md"],
            "questions": ["Is the question clear?"],
            "model": "gpt-5.2-pro",
            "background": False,
            "reviewer": "openai",
        },
    )
    out = jobs_mod.run_job(ledger=ledger, job_id=job["id"])
    assert out["status"] == "succeeded"
    # kind="review" (research review) → no review_fix tasks created.
    assert out["result"]["ingested"]["tasks_created"] == 0

    tasks = ledger.list_tasks(project_id=project["id"])
    assert len(tasks) == 0


@pytest.mark.parametrize("provider", ["anthropic", "claude_code_cli", "codex_cli"])
def test_poll_sync_provider_returns_early(tmp_path: Path, provider: str) -> None:
    """Synchronous providers should return the job as-is without raising."""
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="psync",
        title="PSY",
        domain="",
        stage="intake",
        git_init=False,
    )
    job = jobs_mod.create_job(
        ledger=ledger,
        project_id=project["id"],
        provider=provider,
        kind="review",
        spec={"stage": "intake", "targets": []},
    )
    # Force status to submitted so poll_job doesn't short-circuit.
    with ledger.transaction():
        ledger.update_job(job_id=job["id"], status="submitted")

    result = jobs_mod.poll_job(ledger=ledger, job_id=job["id"])
    assert result["provider"] == provider
