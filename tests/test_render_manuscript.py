from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_render_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "render_manuscript.py"
    spec = importlib.util.spec_from_file_location("render_manuscript", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_render_manuscript_replaces_placeholders(tmp_path: Path) -> None:
    module = _load_render_module()
    workspace = tmp_path / "ws"

    _write(
        workspace / "paper" / "manuscript.template.md",
        "\n".join(
            [
                "# {{primary_metric_name}} Report",
                "Current: {{primary_metric_current}}",
                "Baseline: {{primary_metric_baseline}}",
                "Direction: {{primary_metric_direction}}",
                "Delta: {{delta_vs_baseline}}",
                "Pass: {{test_pass_count}}",
            ]
        )
        + "\n",
    )
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps(
            {
                "primary_metric": {
                    "name": "accuracy",
                    "direction": "maximize",
                    "current": 0.82,
                    "baseline": 0.8,
                },
                "metrics": {"test_pass_count": 281},
            }
        )
        + "\n",
    )

    rendered = module.render_manuscript(workspace)

    assert "accuracy Report" in rendered
    assert "Current: 0.82" in rendered
    assert "Baseline: 0.8" in rendered
    assert "Direction: maximize" in rendered
    assert "Delta: 0.02" in rendered
    assert "Pass: 281" in rendered
    assert (workspace / "paper" / "manuscript.md").exists()


def test_render_manuscript_unknown_key_raises_value_error(tmp_path: Path) -> None:
    module = _load_render_module()
    workspace = tmp_path / "ws"

    _write(workspace / "paper" / "manuscript.template.md", "Unknown: {{does_not_exist}}\n")
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps({"primary_metric": {"name": "m", "direction": "maximize", "current": 1.0, "baseline": 0.5}})
        + "\n",
    )

    with pytest.raises(ValueError, match="Unknown template key"):
        module.render_manuscript(workspace)


def test_render_manuscript_invalid_placeholder_format_raises_value_error(tmp_path: Path) -> None:
    module = _load_render_module()
    workspace = tmp_path / "ws"

    _write(
        workspace / "paper" / "manuscript.template.md",
        "Metric: {{primary_metric_name}}\nBroken: {{bad-key}}\n",
    )
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps({"primary_metric": {"name": "f1", "direction": "maximize", "current": 0.9, "baseline": 0.8}})
        + "\n",
    )

    with pytest.raises(ValueError, match="Invalid placeholder format"):
        module.render_manuscript(workspace)


def test_render_manuscript_missing_template_file_is_graceful(tmp_path: Path) -> None:
    module = _load_render_module()
    workspace = tmp_path / "ws"
    (workspace / "results").mkdir(parents=True, exist_ok=True)
    _write(
        workspace / "results" / "scoreboard.json",
        json.dumps({"primary_metric": {"name": "acc", "direction": "maximize", "current": 0.9, "baseline": 0.8}})
        + "\n",
    )

    with pytest.raises(ValueError, match="Missing manuscript template"):
        module.render_manuscript(workspace)
