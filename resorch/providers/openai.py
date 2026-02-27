from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from resorch.providers.http_json import HttpJsonError, request_json


@dataclass(frozen=True)
class OpenAIClient:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout_sec: int = 3600

    @classmethod
    def from_env(cls) -> "OpenAIClient":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Missing OPENAI_API_KEY.")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return cls(api_key=key, base_url=base_url)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def responses_create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return request_json(
            method="POST",
            url=f"{self.base_url}/responses",
            headers=self._headers(),
            payload=payload,
            timeout_sec=self.timeout_sec,
        )

    def responses_get(self, response_id: str) -> Dict[str, Any]:
        return request_json(
            method="GET",
            url=f"{self.base_url}/responses/{response_id}",
            headers=self._headers(),
            payload=None,
            timeout_sec=self.timeout_sec,
        )


def is_response_done(resp: Dict[str, Any]) -> Optional[bool]:
    """Return whether a Responses API response is in a terminal state.

    Returns:
      - True: completed successfully
      - False: terminal failure or incomplete
      - None: still running / not yet terminal / unknown

    Note: The Responses API includes an "incomplete" status for cases where the
    response stops early; treat that as terminal failure for orchestration.
    """

    status = str(resp.get("status") or "").lower().strip()
    if not status:
        return None

    if status == "completed":
        return True

    # Terminal non-success states
    if status in {"failed", "cancelled", "canceled", "incomplete", "error"}:
        return False

    # Known non-terminal states
    if status in {"queued", "in_progress"}:
        return None

    # Unknown status – be conservative and keep polling.
    return None
