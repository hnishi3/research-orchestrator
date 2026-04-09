from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from resorch.submission_verifier import _run_compile_check


def test_run_compile_check_retries_with_no_pdf_when_latex_engine_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "ws"
    (workspace / "paper" / "output").mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, capture_output: bool, text: bool, timeout: int) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert timeout == 120
        calls.append(cmd)
        if "--no-pdf" in cmd:
            tex_path = workspace / "paper" / "output" / "manuscript.tex"
            tex_path.write_text("\\documentclass{article}\n", encoding="utf-8")
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=f"LaTeX source generated: {tex_path}\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="Parsing manuscript...\nConverting figures...\nBuilding LaTeX document...\nConverting tables...\n",
            stderr=(
                "ERROR: No LaTeX engine found. Install TeX Live:\n"
                "  Ubuntu/Debian: sudo apt-get install texlive-xetex texlive-latex-extra\n"
                "  macOS: brew install --cask mactex\n"
                "  Or use --no-pdf to generate .tex only.\n"
            ),
        )

    monkeypatch.setattr("resorch.submission_verifier.subprocess.run", fake_run)

    status, detail, log_path, log_text = _run_compile_check(workspace, repo_root)

    assert status == "needs_human"
    assert "generated manuscript.tex with --no-pdf" in detail
    assert calls == [
        [sys.executable, str(repo_root / "scripts" / "compile_paper.py"), str(workspace)],
        [sys.executable, str(repo_root / "scripts" / "compile_paper.py"), str(workspace), "--no-pdf"],
    ]
    assert log_path is not None
    assert log_path.read_text(encoding="utf-8") == log_text
    assert "attempt: pdf_compile" in log_text
    assert "attempt: tex_only_retry" in log_text
    assert "--no-pdf" in log_text


def test_run_compile_check_keeps_real_compile_failures_as_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, capture_output: bool, text: bool, timeout: int) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert timeout == 120
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="Parsing manuscript...\n",
            stderr="ERROR: manuscript not found: /tmp/ws/paper/manuscript.md\n",
        )

    monkeypatch.setattr("resorch.submission_verifier.subprocess.run", fake_run)

    status, detail, log_path, log_text = _run_compile_check(workspace, repo_root)

    assert status == "fail"
    assert detail == "compile_paper.py failed with exit code 1"
    assert calls == [[sys.executable, str(repo_root / "scripts" / "compile_paper.py"), str(workspace)]]
    assert log_path is not None
    assert log_path.read_text(encoding="utf-8") == log_text
    assert "--no-pdf" not in log_text
