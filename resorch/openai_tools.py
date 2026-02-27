from __future__ import annotations

import json
import logging
import time
import warnings
from typing import Any, Dict, Iterable, Optional

from resorch.providers.openai import OpenAIClient, is_response_done
from resorch.providers.http_json import HttpJsonError
from resorch.utils import extract_json_object

log = logging.getLogger(__name__)


def run_response_to_completion(
    *,
    client: OpenAIClient,
    payload: Dict[str, Any],
    timeout_sec: int = 3600,
    poll_initial_interval_sec: float = 1.0,
    poll_max_interval_sec: float = 10.0,
) -> Dict[str, Any]:
    """Create an OpenAI Responses API response and poll until terminal.

    This helper intentionally polls even if `background` is omitted/false, since
    some environments still return non-terminal statuses from `responses.create`.
    """

    resp = client.responses_create(payload)
    response_id = resp.get("id")

    status = str(resp.get("status") or "").lower().strip()
    # NOTE: OpenAI Responses API status is expected to be one of:
    # completed/failed/in_progress/cancelled/queued/incomplete.
    # If we see "requires_action", it's likely a mismatched endpoint/SDK shape.
    if status == "requires_action":
        warnings.warn(
            "OpenAI response status 'requires_action' is unexpected for the Responses API. "
            "Check that you're calling /v1/responses (not Assistants). Returning early.",
            RuntimeWarning,
        )
        return resp

    done = is_response_done(resp)
    if done is not None:
        return resp
    if not isinstance(response_id, str) or not response_id:
        return resp

    start = time.time()
    interval = poll_initial_interval_sec
    while True:
        if (time.time() - start) > timeout_sec:
            raise TimeoutError(f"OpenAI response did not finish within {timeout_sec}s. id={response_id}")
        time.sleep(interval)
        resp = client.responses_get(response_id)
        status = str(resp.get("status") or "").lower().strip()
        if status == "requires_action":
            warnings.warn(
                "OpenAI response status 'requires_action' is unexpected for the Responses API. "
                "Check that you're calling /v1/responses (not Assistants). Returning early.",
                RuntimeWarning,
            )
            return resp
        done = is_response_done(resp)
        if done is not None:
            return resp
        interval = min(poll_max_interval_sec, interval * 1.5)


def run_response_to_completion_with_fallback(
    *,
    client: OpenAIClient,
    payload_variants: Iterable[Dict[str, Any]],
    timeout_sec: int = 3600,
) -> Dict[str, Any]:
    """Try multiple payload shapes (API-version differences) until one works."""

    last_err: Optional[Exception] = None
    for payload in payload_variants:
        try:
            return run_response_to_completion(client=client, payload=payload, timeout_sec=timeout_sec)
        except HttpJsonError as e:
            last_err = e
            # Payload-shape errors are typically 400; try the next variant.
            if int(e.status) == 400:
                body_preview = (e.body_text or "")[:500]
                log.warning("OpenAI returned HTTP 400: %s", body_preview)
                continue
            raise

    if last_err is not None:
        raise last_err
    raise ValueError("No payload variants provided.")


def _iter_tool_call_objects(resp: Any) -> Iterable[Dict[str, Any]]:
    if not isinstance(resp, dict):
        return

    # Responses API (common): {"output": [ ... ]}
    output = resp.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                yield item
                # Some wrappers nest tool calls.
                tool_calls = item.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            yield tc
                # Message content items can include tool calls.
                content = item.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            yield c

    # Top-level tool_calls
    tool_calls = resp.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict):
                yield tc

    # Chat Completions-style: {"choices":[{"message":{"tool_calls":[...]}}]}
    choices = resp.get("choices")
    if isinstance(choices, list):
        for ch in choices:
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message")
            if not isinstance(msg, dict):
                continue
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list):
                for tc in tcs:
                    if isinstance(tc, dict):
                        yield tc


def extract_function_call_arguments(resp: Dict[str, Any], *, function_name: str) -> Optional[Dict[str, Any]]:
    """Extract function-call arguments for a named tool from an OpenAI response.

    Supports several response shapes (Responses API, Chat Completions legacy).
    Returns None if not found or not parseable as a JSON object.
    """

    for tc in _iter_tool_call_objects(resp):
        name: Optional[str] = None
        args: Any = None

        # Common patterns
        if isinstance(tc.get("name"), str):
            name = tc["name"]
            args = tc.get("arguments")
        elif isinstance(tc.get("function"), dict):
            fn = tc["function"]
            if isinstance(fn.get("name"), str):
                name = fn["name"]
                args = fn.get("arguments")

        if name != function_name:
            continue

        if isinstance(args, dict):
            return args
        if isinstance(args, str) and args.strip():
            try:
                parsed = json.loads(args)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None

    return None


def extract_json_object_from_response(resp: Dict[str, Any], *, function_name: Optional[str] = None) -> Dict[str, Any]:
    """Extract a JSON object either from a tool call or from output_text."""

    if function_name:
        args = extract_function_call_arguments(resp, function_name=function_name)
        if args is not None:
            return args

    out_text = resp.get("output_text")
    if isinstance(out_text, str) and out_text.strip():
        return extract_json_object(out_text)

    raise ValueError("Could not extract JSON object from response.")
