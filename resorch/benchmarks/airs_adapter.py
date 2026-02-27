from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import subprocess
import sys

from resorch.benchmarks.base import (
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkTask,
    _read_simple_yaml,
    _run_agent_subprocess,
)


class AIRSBenchSuite(BenchmarkSuite):
    def __init__(self, external_path: Path | None = None) -> None:
        super().__init__(
            name="airs",
            description=(
                "AIRS-Bench (Meta, arXiv:2602.06855): 20 ML tasks with "
                "metadata + prepare/evaluate scripts."
            ),
            external_path=external_path or Path("external/airs-bench"),
        )

    def _tasks_root(self) -> Path:
        root = self._require_external_path(
            hint="Clone facebookresearch/airs-bench under external/airs-bench or pass --external-path."
        )
        candidates = [root / "airsbench" / "tasks", root / "tasks"]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return root / "airsbench" / "tasks"

    def list_tasks(self) -> list[BenchmarkTask]:
        tasks_root = self._tasks_root()
        if not tasks_root.exists():
            return []

        tasks: list[BenchmarkTask] = []
        for task_dir in sorted([p for p in tasks_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            metadata_path = task_dir / "metadata.yaml"
            if not metadata_path.exists():
                continue
            metadata = _read_simple_yaml(metadata_path)
            task = BenchmarkTask(
                task_id=str(metadata.get("task_id") or task_dir.name),
                title=str(metadata.get("title") or task_dir.name),
                description=str(metadata.get("description") or f"AIRS task at {task_dir.name}"),
                paper_ref=str(metadata.get("paper_ref")) if metadata.get("paper_ref") else None,
                rubric_path=metadata_path,
            )
            tasks.append(task)
        return tasks

    def _task_dir_for_id(self, task_id: str) -> Path | None:
        tasks_root = self._tasks_root()
        if not tasks_root.exists():
            return None
        for task_dir in sorted([p for p in tasks_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            metadata_path = task_dir / "metadata.yaml"
            if not metadata_path.exists():
                continue
            metadata = _read_simple_yaml(metadata_path)
            candidate_id = str(metadata.get("task_id") or task_dir.name)
            if candidate_id == task_id:
                return task_dir
        return None

    def run_task(
        self,
        task: str | BenchmarkTask,
        workspace: Path,
        ledger: Any = None,
        dry_run: bool = False,
        max_steps: int = 20,
    ) -> BenchmarkResult:
        task_obj = task if isinstance(task, BenchmarkTask) else None
        task_id = task_obj.task_id if task_obj is not None else str(task)
        if task_obj is None:
            try:
                task_obj = self.get_task(task_id)
            except (FileNotFoundError, KeyError) as exc:
                return BenchmarkResult(
                    task_id=task_id,
                    status="not_available",
                    score=None,
                    details={
                        "suite": self.name,
                        "error": str(exc),
                    },
                )

        if dry_run:
            return BenchmarkResult(
                task_id=task_obj.task_id,
                status="skipped",
                score=None,
                details={
                    "suite": self.name,
                    "mode": "dry_run",
                    "note": "Dry run — agent subprocess would be launched here.",
                },
            )

        setup = self.setup_workspace_for_task(task_obj, workspace=Path(workspace), ledger=ledger)
        if setup.get("error"):
            return BenchmarkResult(
                task_id=task_obj.task_id,
                status="not_available",
                score=None,
                details={
                    "suite": self.name,
                    "setup": setup,
                },
            )

        run_dir = Path(str(setup["run_dir"]))
        inputs_dir = run_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        try:
            task_dir = self._task_dir_for_id(task_obj.task_id)
            if task_dir is None:
                raise FileNotFoundError(f"Task directory not found for id={task_obj.task_id}")

            copied_items: list[str] = []
            for entry in sorted(task_dir.iterdir(), key=lambda p: p.name):
                if entry.name.startswith("."):
                    continue
                dst = inputs_dir / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(entry, dst)
                copied_items.append(entry.name)
        except Exception as exc:  # noqa: BLE001
            return BenchmarkResult(
                task_id=task_obj.task_id,
                status="not_available",
                score=None,
                details={
                    "suite": self.name,
                    "setup": setup,
                    "error": str(exc),
                },
            )

        project_id = setup.get("project_id")
        if not project_id:
            return BenchmarkResult(
                task_id=task_obj.task_id,
                status="not_available",
                score=None,
                details={
                    "suite": self.name,
                    "run_dir": str(run_dir),
                    "error": "No project_id created (ledger not provided?)",
                },
            )

        # Run prepare.py if present in the task directory
        prepare_script = inputs_dir / "prepare.py"
        if prepare_script.exists():
            subprocess.run(
                [sys.executable, str(prepare_script)],
                cwd=str(run_dir),
                capture_output=True,
                text=True,
                check=False,
            )

        objective = (
            f"AIRS benchmark task: {task_obj.title}.\n"
            f"Description: {task_obj.description}"
        )
        proc = _run_agent_subprocess(
            project_id=project_id,
            objective=objective,
            max_steps=max_steps,
            repo_root=run_dir,
        )
        status = "success" if proc.returncode == 0 else "failed"

        # Run evaluate.py if present and agent succeeded
        score = None
        evaluate_script = inputs_dir / "evaluate.py"
        if evaluate_script.exists() and proc.returncode == 0:
            try:
                eval_proc = subprocess.run(
                    [sys.executable, str(evaluate_script)],
                    cwd=str(run_dir),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if eval_proc.returncode == 0 and eval_proc.stdout.strip():
                    eval_result = json.loads(eval_proc.stdout.strip())
                    if isinstance(eval_result, dict) and "score" in eval_result:
                        score = float(eval_result["score"])
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return BenchmarkResult(
            task_id=task_obj.task_id,
            status=status,
            score=score,
            details={
                "suite": self.name,
                "run_dir": str(run_dir),
                "project_id": project_id,
                "problem_md": setup.get("problem_md"),
                "inputs_dir": str(inputs_dir),
                "copied_items": copied_items,
                "returncode": proc.returncode,
            },
        )
