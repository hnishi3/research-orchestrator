from __future__ import annotations

import json

from resorch.openai_tools import extract_function_call_arguments


def test_extract_function_call_arguments_from_responses_output() -> None:
    resp = {
        "id": "resp_1",
        "status": "completed",
        "output": [
            {
                "type": "tool_call",
                "name": "submit_plan",
                "arguments": {"plan_id": "p1"},
            }
        ],
    }
    out = extract_function_call_arguments(resp, function_name="submit_plan")
    assert out == {"plan_id": "p1"}


def test_extract_function_call_arguments_from_chat_completions_tool_calls() -> None:
    resp = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "submit_plan", "arguments": json.dumps({"plan_id": "p2"})},
                        }
                    ]
                }
            }
        ]
    }
    out = extract_function_call_arguments(resp, function_name="submit_plan")
    assert out == {"plan_id": "p2"}

