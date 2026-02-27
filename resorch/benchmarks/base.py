from __future__ import annotations

import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


_VALID_STATUSES = {"success", "failed", "skipped", "not_available", "ready"}


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    """Read flat `key: value` YAML pairs.

    Limitations: multi-line values and nested YAML structures are not supported.
    """

    data: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    title: str
    description: str
    paper_ref: Optional[str] = None
    rubric_path: Optional[Path] = None


@dataclass
class BenchmarkResult:
    task_id: str
    status: str
    score: Optional[float] = None
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid benchmark status '{self.status}'. "
                f"Expected one of {sorted(_VALID_STATUSES)}."
            )


class BenchmarkSuite(ABC):
    name: str
    description: str
    external_path: Path

    def __init__(self, *, name: str, description: str, external_path: Path) -> None:
        self.name = str(name)
        self.description = str(description)
        self.external_path = Path(external_path).expanduser()

    def _require_external_path(self, *, hint: str) -> Path:
        path = self.external_path.resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"{self.name} benchmark path not found: {path}. {hint}"
            )
        return path

    @abstractmethod
    def list_tasks(self) -> list[BenchmarkTask]:
        raise NotImplementedError

    def get_task(self, task_id: str) -> BenchmarkTask:
        task_id_norm = str(task_id).strip()
        for task in self.list_tasks():
            if task.task_id == task_id_norm:
                return task
        raise KeyError(f"Task not found in suite '{self.name}': {task_id_norm}")

    def setup_workspace_for_task(
        self, task: BenchmarkTask, workspace: Path, ledger: Any = None
    ) -> dict[str, Any]:
        """Copy benchmark task inputs into workspace and optionally create a project."""

        run_dir = Path(workspace).resolve() / "benchmarks" / self.name / task.task_id
        problem_md = run_dir / "notes" / "problem.md"
        setup_info: dict[str, Any] = {
            "run_dir": str(run_dir),
            "problem_md": str(problem_md),
            "project_id": None,
        }
        try:
            run_dir.mkdir(parents=True, exist_ok=True)

            if task.rubric_path and task.rubric_path.exists():
                import shutil

                shutil.copy2(task.rubric_path, run_dir / task.rubric_path.name)

            notes_dir = run_dir / "notes"
            notes_dir.mkdir(exist_ok=True)
            problem_md.write_text(
                f"# Benchmark Task: {task.title}\n\n"
                f"Suite: {self.name}\n"
                f"Task ID: {task.task_id}\n"
                f"Description: {task.description}\n"
                f'{"Paper: " + task.paper_ref if task.paper_ref else ""}\n',
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            setup_info["error"] = f"workspace setup failed: {exc}"
            return setup_info

        if ledger is not None:
            try:
                from resorch.projects import create_project

                project = create_project(
                    ledger=ledger,
                    project_id=None,
                    title=f"[bench:{self.name}] {task.title}",
                    domain=f"benchmark/{self.name}",
                    stage="intake",
                    git_init=False,
                )
                setup_info["project_id"] = project.get("id")
            except Exception as exc:  # noqa: BLE001
                setup_info["project_error"] = f"project creation failed: {exc}"

        return setup_info

    @abstractmethod
    def run_task(
        self,
        task: str | BenchmarkTask,
        workspace: Path,
        ledger: Any = None,
        dry_run: bool = False,
        max_steps: int = 20,
    ) -> BenchmarkResult:
        raise NotImplementedError


def _run_agent_subprocess(
    *,
    project_id: str,
    objective: str,
    max_steps: int = 20,
    repo_root: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Launch the resorch agent loop as a subprocess and return the result."""
    cmd = [
        sys.executable,
        "-m",
        "resorch",
        "agent",
        "run",
        "--project",
        str(project_id),
        "--objective",
        objective,
        "--max-steps",
        str(int(max_steps)),
    ]
    if repo_root is not None:
        cmd.extend(["--repo-root", str(repo_root)])
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
