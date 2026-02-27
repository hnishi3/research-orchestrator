from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class HttpJsonError(Exception):
    status: int
    message: str
    body_text: str

    def __str__(self) -> str:  # pragma: no cover
        return f"HTTP {self.status}: {self.message}"

    @property
    def is_retriable(self) -> bool:
        return self.status in {408, 429, 500, 502, 503, 504}


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    v = str(value).strip()
    if not v:
        return None

    # Retry-After can be either a delay-seconds or an HTTP-date.
    if v.isdigit():
        try:
            return max(0.0, float(int(v)))
        except ValueError:
            return None

    try:
        dt = parsedate_to_datetime(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (dt - now).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def request_json(
    *,
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    timeout_sec: int = 120,
) -> Dict[str, Any]:
    method_u = method.upper()
    data = None
    req_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(url=url, method=method_u, data=data, headers=req_headers)

    default_retries = 2
    try:
        max_retries = int(os.environ.get("RESORCH_HTTP_RETRIES", str(default_retries)))
    except ValueError:
        max_retries = default_retries
    default_rate_limit_retries = 5
    try:
        rate_limit_retries = int(os.environ.get("RESORCH_HTTP_RATE_LIMIT_RETRIES", str(default_rate_limit_retries)))
    except ValueError:
        rate_limit_retries = default_rate_limit_retries
    try:
        backoff_sec = float(os.environ.get("RESORCH_HTTP_RETRY_BACKOFF_SEC", "0.5"))
    except ValueError:
        backoff_sec = 0.5

    attempt_general = 0
    attempt_rate_limit = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                body = resp.read()
                text = body.decode("utf-8", errors="replace")
                if not text.strip():
                    return {}
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"_raw": text}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            status = int(e.code)
            message = str(e.reason)
            if status == 400 and "context_length_exceeded" in (body or "").lower():
                message = f"{message} (context_length_exceeded)"

            retriable = status in {408, 429, 500, 502, 503, 504}
            if retriable:
                if status == 429:
                    if attempt_rate_limit < rate_limit_retries:
                        retry_after = _parse_retry_after(e.headers.get("Retry-After") if e.headers else None)
                        if retry_after is None:
                            retry_after = max(0.0, backoff_sec) * (2**attempt_rate_limit)
                        time.sleep(min(float(retry_after), 300.0))
                        attempt_rate_limit += 1
                        continue
                elif attempt_general < max_retries:
                    time.sleep(min(max(0.0, backoff_sec) * (2**attempt_general), 300.0))
                    attempt_general += 1
                    continue

            raise HttpJsonError(status=status, message=message, body_text=body)
        except urllib.error.URLError as e:
            if attempt_general < max_retries:
                time.sleep(min(max(0.0, backoff_sec) * (2**attempt_general), 300.0))
                attempt_general += 1
                continue
            raise HttpJsonError(status=0, message=str(e.reason), body_text="")
        except (TimeoutError, OSError) as e:
            if attempt_general < max_retries:
                time.sleep(min(max(0.0, backoff_sec) * (2**attempt_general), 300.0))
                attempt_general += 1
                continue
            raise HttpJsonError(status=0, message=f"timeout: {e}", body_text="")
