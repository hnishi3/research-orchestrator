from __future__ import annotations

import io
from email.message import Message
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError

import pytest

from resorch.providers.http_json import HttpJsonError, request_json


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


def _http_error(*, status: int, reason: str, body_text: str, headers: Optional[Dict[str, str]] = None) -> HTTPError:
    hdrs = Message()
    for k, v in (headers or {}).items():
        hdrs[k] = v
    fp = io.BytesIO(body_text.encode("utf-8"))
    return HTTPError(url="http://example.test", code=status, msg=reason, hdrs=hdrs, fp=fp)


def test_request_json_200_returns_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(_req, timeout: int = 0):  # noqa: ANN001
        return _FakeResp(b'{"ok": true}')

    monkeypatch.setattr("resorch.providers.http_json.urllib.request.urlopen", fake_urlopen)
    out = request_json(method="GET", url="http://example.test")
    assert out == {"ok": True}


def test_request_json_400_context_length_exceeded_sets_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(_req, timeout: int = 0):  # noqa: ANN001
        raise _http_error(status=400, reason="Bad Request", body_text='{"error":{"code":"context_length_exceeded"}}')

    monkeypatch.setattr("resorch.providers.http_json.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(HttpJsonError) as e:
        request_json(method="POST", url="http://example.test", payload={"x": "y"})
    assert "context_length" in e.value.message


def test_request_json_500_retries_twice_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESORCH_HTTP_RETRIES", "2")
    monkeypatch.setenv("RESORCH_HTTP_RETRY_BACKOFF_SEC", "0.1")

    sleeps: List[float] = []

    def fake_sleep(sec: float) -> None:
        sleeps.append(float(sec))

    calls = {"n": 0}

    def fake_urlopen(_req, timeout: int = 0):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _http_error(status=500, reason="Server Error", body_text="oops")
        return _FakeResp(b'{"ok": 1}')

    monkeypatch.setattr("resorch.providers.http_json.time.sleep", fake_sleep)
    monkeypatch.setattr("resorch.providers.http_json.urllib.request.urlopen", fake_urlopen)

    out = request_json(method="GET", url="http://example.test")
    assert out == {"ok": 1}
    assert sleeps == [0.1, 0.2]


def test_request_json_429_respects_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESORCH_HTTP_RATE_LIMIT_RETRIES", "5")
    monkeypatch.setenv("RESORCH_HTTP_RETRY_BACKOFF_SEC", "0.1")

    sleeps: List[float] = []

    def fake_sleep(sec: float) -> None:
        sleeps.append(float(sec))

    calls = {"n": 0}

    def fake_urlopen(_req, timeout: int = 0):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(status=429, reason="Too Many Requests", body_text="rl", headers={"Retry-After": "2"})
        return _FakeResp(b'{"ok": 1}')

    monkeypatch.setattr("resorch.providers.http_json.time.sleep", fake_sleep)
    monkeypatch.setattr("resorch.providers.http_json.urllib.request.urlopen", fake_urlopen)

    out = request_json(method="GET", url="http://example.test")
    assert out == {"ok": 1}
    assert sleeps == [2.0]


def test_request_json_429_exhausts_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default 429 retry budget is 5 retries (6 total attempts).
    for k in ["RESORCH_HTTP_RATE_LIMIT_RETRIES", "RESORCH_HTTP_RETRIES"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RESORCH_HTTP_RETRY_BACKOFF_SEC", "0.0")

    sleeps: List[float] = []

    def fake_sleep(sec: float) -> None:
        sleeps.append(float(sec))

    def fake_urlopen(_req, timeout: int = 0):  # noqa: ANN001
        raise _http_error(status=429, reason="Too Many Requests", body_text="rl")

    monkeypatch.setattr("resorch.providers.http_json.time.sleep", fake_sleep)
    monkeypatch.setattr("resorch.providers.http_json.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(HttpJsonError) as e:
        request_json(method="GET", url="http://example.test")
    assert e.value.status == 429
    assert len(sleeps) == 5
    assert e.value.is_retriable is True
