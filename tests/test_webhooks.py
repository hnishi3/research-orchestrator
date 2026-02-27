from __future__ import annotations

import base64
import hashlib
import hmac
import time
from pathlib import Path

from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import create_project
from resorch.jobs import create_job, get_job
from resorch.webhooks import handle_openai_webhook
from resorch.webhooks import verify_standard_webhook_signature, WebhookSignatureError


def _make_tmp_repo(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


def test_handle_openai_webhook_updates_job(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )
    job = create_job(
        ledger=ledger,
        project_id=project["id"],
        provider="openai",
        kind="response",
        spec={"payload": {"model": "gpt-5.2", "input": "hi", "background": True}},
    )
    ledger.update_job(job_id=job["id"], status="submitted", remote_id="resp_999")

    updated = handle_openai_webhook(ledger=ledger, payload={"id": "resp_999", "status": "completed", "output_text": "ok"})
    assert updated is not None

    after = get_job(ledger, job["id"])
    assert after["status"] == "succeeded"


def test_handle_openai_webhook_event_wrapper_updates_job(tmp_path: Path) -> None:
    ledger = _make_tmp_repo(tmp_path)
    project = create_project(
        ledger=ledger,
        project_id="p1",
        title="P1",
        domain="",
        stage="intake",
        git_init=False,
    )
    job = create_job(
        ledger=ledger,
        project_id=project["id"],
        provider="openai",
        kind="response",
        spec={"payload": {"model": "gpt-5.2", "input": "hi", "background": True}},
    )
    ledger.update_job(job_id=job["id"], status="submitted", remote_id="resp_999")

    updated = handle_openai_webhook(
        ledger=ledger,
        payload={"object": "event", "type": "response.completed", "data": {"id": "resp_999"}},
    )
    assert updated is not None

    after = get_job(ledger, job["id"])
    assert after["status"] == "succeeded"


def test_verify_standard_webhook_signature_roundtrip() -> None:
    secret_bytes = b"supersecret"
    secret_b64 = base64.b64encode(secret_bytes).decode("utf-8")
    body = b'{"object":"event","type":"response.completed","data":{"id":"resp_123"}}'
    msg_id = "msg_123"
    ts = time.time()
    ts_floor = str(int(ts))
    to_sign = f"{msg_id}.{ts_floor}.{body.decode('utf-8')}".encode("utf-8")
    sig = hmac.new(secret_bytes, to_sign, hashlib.sha256).digest()
    sig_b64 = base64.b64encode(sig).decode("utf-8")
    headers = {
        "webhook-id": msg_id,
        "webhook-timestamp": str(ts),
        "webhook-signature": f"v1,{sig_b64}",
    }
    verify_standard_webhook_signature(secret=f"whsec_{secret_b64}", body=body, headers=headers, tolerance_sec=300)

    bad_headers = dict(headers)
    bad_headers["webhook-signature"] = "v1,ZmFrZQ=="  # base64("fake")
    try:
        verify_standard_webhook_signature(secret=secret_b64, body=body, headers=bad_headers, tolerance_sec=300)
        raise AssertionError("expected WebhookSignatureError")
    except WebhookSignatureError:
        pass
