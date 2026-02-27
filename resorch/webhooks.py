from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from math import floor
from typing import Any, Dict, Optional, Tuple

from resorch.ledger import Ledger
from resorch.paths import resolve_repo_paths
from resorch.providers.openai import is_response_done


def _get_token_from_request(path: str, headers: Dict[str, str], *, allow_query_token: bool) -> Optional[str]:
    # Prefer header token.
    for k in ("X-Resorch-Token", "X-Webhook-Token", "X-Webhook-Secret"):
        if k in headers:
            return headers[k]
    if not allow_query_token or "?" not in path:
        return None
    query = path.split("?", 1)[1]
    for part in query.split("&"):
        if not part:
            continue
        if part.startswith("token="):
            return part.split("=", 1)[1]
    return None


class WebhookSignatureError(RuntimeError):
    pass


def _decode_standard_webhook_secret(secret: str) -> bytes:
    s = str(secret or "").strip()
    if not s:
        raise WebhookSignatureError("Empty webhook secret")
    if s.startswith("whsec_"):
        s = s[len("whsec_") :]
    try:
        return base64.b64decode(s)
    except Exception as e:  # noqa: BLE001
        raise WebhookSignatureError("Invalid webhook secret (expected base64 or whsec_... base64)") from e


def verify_standard_webhook_signature(
    *,
    secret: str,
    body: bytes,
    headers: Dict[str, str],
    tolerance_sec: int = 300,
) -> None:
    """Verify Standard Webhooks signature headers (webhook-id/timestamp/signature).

    Spec reference (community): https://github.com/standard-webhooks/standard-webhooks

    Headers:
      - webhook-id
      - webhook-timestamp (unix seconds, float/int)
      - webhook-signature ("v1,<base64>" possibly multiple separated by spaces)
    """

    hdrs = {str(k).lower(): str(v) for (k, v) in (headers or {}).items()}
    msg_id = hdrs.get("webhook-id")
    msg_timestamp = hdrs.get("webhook-timestamp")
    msg_signature = hdrs.get("webhook-signature")
    if not (msg_id and msg_timestamp and msg_signature):
        raise WebhookSignatureError("Missing required webhook signature headers")

    try:
        ts = float(msg_timestamp)
    except ValueError as e:
        raise WebhookSignatureError("Invalid webhook-timestamp") from e

    now = time.time()
    tol = max(0, int(tolerance_sec))
    if ts < (now - tol):
        raise WebhookSignatureError("Message timestamp too old")
    if ts > (now + tol):
        raise WebhookSignatureError("Message timestamp too new")

    secret_bytes = _decode_standard_webhook_secret(secret)
    try:
        body_text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as e:
        raise WebhookSignatureError("Webhook body is not valid UTF-8") from e
    ts_str = str(floor(ts))
    to_sign = f"{msg_id}.{ts_str}.{body_text}".encode("utf-8")
    expected_sig = hmac.new(secret_bytes, to_sign, hashlib.sha256).digest()

    for versioned_sig in str(msg_signature).split():
        if "," not in versioned_sig:
            continue
        version, sig_b64 = versioned_sig.split(",", 1)
        if version != "v1":
            continue
        try:
            sig_bytes = base64.b64decode(sig_b64)
        except Exception:  # noqa: BLE001
            continue
        if hmac.compare_digest(expected_sig, sig_bytes):
            return

    raise WebhookSignatureError("No matching signature found")


def handle_openai_webhook(*, ledger: Ledger, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle an incoming OpenAI webhook payload.

    OpenAI webhooks are delivered as Standard Webhooks "event" objects.
    In the simplest case, the payload looks like:

      {"object":"event","type":"response.completed","data":{"id":"resp_..."}}

    However, during local testing it can also be convenient to POST a raw
    Responses object (or a wrapper containing `response`). This handler accepts
    both shapes.
    """

    remote_id: Optional[str] = None
    resp_obj: Dict[str, Any]

    # 1) Standard event wrapper (recommended)
    if payload.get("object") == "event" and isinstance(payload.get("data"), dict):
        data = payload["data"]
        if isinstance(data.get("id"), str):
            remote_id = data["id"]
            ev_type = str(payload.get("type") or "").strip()

            # Derive a coarse response status from the event type.
            # (If you want full outputs, fetch the response by id.)
            derived_status: Optional[str] = None
            if ev_type == "response.completed":
                derived_status = "completed"
            elif ev_type in {"response.failed", "response.incomplete"}:
                derived_status = "failed"
            elif ev_type in {"response.cancelled", "response.canceled"}:
                derived_status = "cancelled"

            resp_obj = {"id": remote_id, "status": derived_status, "event": payload}
        else:
            return None

    # 2) Direct Responses object
    elif "id" in payload and isinstance(payload.get("id"), str):
        remote_id = payload["id"]
        resp_obj = payload

    # 3) Wrapper with `response: {...}`
    elif isinstance(payload.get("response"), dict) and isinstance(payload["response"].get("id"), str):
        remote_id = payload["response"]["id"]
        resp_obj = payload["response"]

    else:
        return None

    matches = ledger.find_jobs_by_remote_id(remote_id=remote_id, provider="openai")
    if not matches:
        return None
    job_id = matches[0]["id"]

    done = is_response_done(resp_obj)
    if done is True:
        status = "succeeded"
        finished = True
    elif done is False:
        status = "failed"
        finished = True
    else:
        status = "running"
        finished = False

    with ledger.transaction():
        ledger.update_job(job_id=job_id, status=status, result=resp_obj, finished=finished, commit=False)
        ledger.insert_job_event(
            job_id=job_id,
            event_type="webhook.openai",
            data={
                "remote_id": remote_id,
                "status": resp_obj.get("status"),
                "event_type": (resp_obj.get("event") or {}).get("type") if isinstance(resp_obj.get("event"), dict) else None,
            },
        )
    return ledger.get_job(job_id)


class WebhookHandler(BaseHTTPRequestHandler):
    server_version = "resorch-webhooks/0.1"

    def _send_json(self, status: int, obj: Dict[str, Any]) -> None:
        body = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        # Token auth (optional).
        expected = os.environ.get("RESORCH_WEBHOOK_TOKEN")
        hdrs = {k: v for k, v in self.headers.items()}
        if expected:
            allow_query = bool(os.environ.get("RESORCH_WEBHOOK_ALLOW_QUERY_TOKEN"))
            got = _get_token_from_request(self.path, hdrs, allow_query_token=allow_query)
            if (not isinstance(got, str)) or (not hmac.compare_digest(got, expected)):
                self._send_json(401, {"error": "unauthorized"})
                return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""

        # Optional: verify Standard Webhooks signatures.
        # When OPENAI_WEBHOOK_SECRET is set, require signature verification on /openai.
        if self.path.split("?", 1)[0] == "/openai":
            secret = os.environ.get("OPENAI_WEBHOOK_SECRET")
            if secret:
                try:
                    tol = int(os.environ.get("RESORCH_WEBHOOK_TOLERANCE_SEC") or "300")
                except ValueError:
                    tol = 300
                try:
                    verify_standard_webhook_signature(secret=secret, body=raw, headers=hdrs, tolerance_sec=tol)
                except WebhookSignatureError as e:
                    self._send_json(401, {"error": f"invalid_signature: {e}"})
                    return

        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"invalid_json: {e}"})
            return

        ledger: Ledger = self.server.ledger  # type: ignore[attr-defined]

        if self.path.split("?", 1)[0] == "/openai":
            job = handle_openai_webhook(ledger=ledger, payload=payload)
            if job is None:
                self._send_json(404, {"error": "job_not_found_or_invalid_payload"})
                return
            self._send_json(200, {"ok": True, "job_id": job["id"]})
            return

        self._send_json(404, {"error": "unknown_path"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Quiet by default; set RESORCH_WEBHOOK_LOG=1 to enable.
        if os.environ.get("RESORCH_WEBHOOK_LOG"):
            super().log_message(format, *args)


def run_server(*, ledger: Ledger, host: str, port: int) -> None:
    httpd = HTTPServer((host, port), WebhookHandler)
    httpd.ledger = ledger  # type: ignore[attr-defined]
    httpd.serve_forever()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Resorch webhook receiver (HTTP)")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    paths = resolve_repo_paths(args.repo_root)
    ledger = Ledger(paths)
    ledger.init()
    run_server(ledger=ledger, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
