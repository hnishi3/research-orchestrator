from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from resorch.providers.http_json import request_json


@dataclass(frozen=True)
class AnthropicClient:
    api_key: str
    base_url: str = "https://api.anthropic.com"
    version: str = "2023-06-01"
    timeout_sec: int = 120

    @classmethod
    def from_env(cls) -> "AnthropicClient":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("Missing ANTHROPIC_API_KEY.")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        version = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")
        return cls(api_key=key, base_url=base_url, version=version)

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
        }

    def messages_create(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": int(max_tokens),
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        return request_json(
            method="POST",
            url=f"{self.base_url}/v1/messages",
            headers=self._headers(),
            payload=payload,
            timeout_sec=self.timeout_sec,
        )


def extract_text(resp: Dict[str, Any]) -> str:
    content = resp.get("content")
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts).strip()
    if isinstance(content, str):
        return content.strip()
    # Fallback: try common fields.
    for k in ("output_text", "text"):
        v = resp.get(k)
        if isinstance(v, str):
            return v.strip()
    return ""

