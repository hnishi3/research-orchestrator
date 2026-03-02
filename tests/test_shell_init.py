from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict

import yaml

from resorch.agent_loop import AgentLoopConfig
from resorch.autopilot_action import _inject_shell_init, _inject_shell_init_into_codex


CONDA_ACTIVATE = "source /data/anaconda3/etc/profile.d/conda.sh && conda activate simple-neutralnet"


# ---------------------------------------------------------------------------
# AgentLoopConfig parsing
# ---------------------------------------------------------------------------


def test_agent_loop_config_shell_init_from_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "agent_loop.yaml"
    cfg_path.write_text(
        textwrap.dedent(f"""\
            planner:
              provider: claude_code_cli
              model: opus
            executor:
              shell_init: "{CONDA_ACTIVATE}"
        """),
        encoding="utf-8",
    )
    cfg = AgentLoopConfig.from_yaml(cfg_path)
    assert cfg.shell_init == CONDA_ACTIVATE


def test_agent_loop_config_shell_init_default_none(tmp_path: Path) -> None:
    cfg_path = tmp_path / "agent_loop.yaml"
    cfg_path.write_text(
        textwrap.dedent("""\
            planner:
              provider: claude_code_cli
              model: opus
        """),
        encoding="utf-8",
    )
    cfg = AgentLoopConfig.from_yaml(cfg_path)
    assert cfg.shell_init is None


def test_agent_loop_config_shell_init_empty_string(tmp_path: Path) -> None:
    """Empty string should be treated as None (falsy)."""
    cfg_path = tmp_path / "agent_loop.yaml"
    cfg_path.write_text(
        textwrap.dedent("""\
            planner:
              provider: claude_code_cli
            executor:
              shell_init: ""
        """),
        encoding="utf-8",
    )
    cfg = AgentLoopConfig.from_yaml(cfg_path)
    assert cfg.shell_init is None


# ---------------------------------------------------------------------------
# _inject_shell_init (shell_exec)
# ---------------------------------------------------------------------------


def test_inject_shell_init_prepends_and_forces_shell() -> None:
    spec: Dict[str, Any] = {"command": "python train.py", "shell": False}
    result = _inject_shell_init(spec, CONDA_ACTIVATE)
    assert result["command"] == f"{CONDA_ACTIVATE} && python train.py"
    assert result["shell"] is True
    # Original unchanged
    assert spec["shell"] is False


def test_inject_shell_init_list_command() -> None:
    spec: Dict[str, Any] = {"command": ["python", "train.py"]}
    result = _inject_shell_init(spec, CONDA_ACTIVATE)
    assert result["command"] == f"{CONDA_ACTIVATE} && python train.py"
    assert result["shell"] is True


def test_inject_shell_init_preserves_other_keys() -> None:
    spec: Dict[str, Any] = {"command": "echo hello", "timeout_sec": 300, "cd": "src"}
    result = _inject_shell_init(spec, CONDA_ACTIVATE)
    assert result["timeout_sec"] == 300
    assert result["cd"] == "src"


# ---------------------------------------------------------------------------
# _inject_shell_init_into_codex (codex_exec)
# ---------------------------------------------------------------------------


def test_inject_shell_init_into_codex_appends_prompt() -> None:
    spec: Dict[str, Any] = {"prompt": "Run the analysis pipeline."}
    result = _inject_shell_init_into_codex(spec, CONDA_ACTIVATE)
    assert "--- SHELL ENVIRONMENT ---" in result["prompt"]
    assert CONDA_ACTIVATE in result["prompt"]
    assert "--- END SHELL ENVIRONMENT ---" in result["prompt"]
    assert result["prompt"].startswith("Run the analysis pipeline.")
    # Original unchanged
    assert "SHELL ENVIRONMENT" not in spec["prompt"]


def test_inject_shell_init_into_codex_empty_prompt() -> None:
    spec: Dict[str, Any] = {"prompt": ""}
    result = _inject_shell_init_into_codex(spec, CONDA_ACTIVATE)
    assert CONDA_ACTIVATE in result["prompt"]


# ---------------------------------------------------------------------------
# No injection when shell_init is None
# ---------------------------------------------------------------------------


def test_shell_init_not_injected_when_none() -> None:
    """Simulate what autopilot.py does: only inject if shell_init is truthy."""
    spec: Dict[str, Any] = {"command": "echo hello"}
    shell_init = None
    if shell_init:
        spec = _inject_shell_init(spec, shell_init)
    assert spec["command"] == "echo hello"
    assert "shell" not in spec
