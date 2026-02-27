from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from resorch.benchmarks.base import (
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkTask,
    _read_simple_yaml,
    _run_agent_subprocess,
)


def _load_metadata(task_dir: Path) -> dict[str, Any]:
    for name in ("metadata.json", "task.json"):
        fp = task_dir / name
        if fp.exists():
            try:
                payload = json.loads(fp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                return payload
    for name in ("metadata.yaml", "metadata.yml", "task.yaml", "task.yml"):
        fp = task_dir / name
        if fp.exists():
            return _read_simple_yaml(fp)
    return {}


class ReplicatorBenchSuite(BenchmarkSuite):
    def __init__(self, external_path: Path | None = None) -> None:
        super().__init__(
            name="replicatorbench",
            description=(
                "ReplicatorBench (arXiv:2602.11354): social/behavioral science "
                "replication tasks across multi-stage workflows."
            ),
            external_path=external_path or Path("external/replicatorbench"),
        )

    def _task_dirs(self) -> list[Path]:
        root = self._require_external_path(
            hint="Clone ReplicatorBench under external/replicatorbench or pass --external-path."
        )

        task_dirs: list[Path] = []
        for rel in ("tasks", "benchmarks", "stages"):
            base = root / rel
            if not (base.exists() and base.is_dir()):
                continue
            for entry in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name):
                if any((entry / name).exists() for name in ("metadata.json", "metadata.yaml", "metadata.yml", "task.json", "task.yaml", "task.yml")):
                    task_dirs.append(entry)
                else:
                    nested = [p for p in entry.iterdir() if p.is_dir()]
                    for child in sorted(nested, key=lambda p: p.name):
                        if any((child / name).exists() for name in ("metadata.json", "metadata.yaml", "metadata.yml", "task.json", "task.yaml", "task.yml")):
                            task_dirs.append(child)
        if task_dirs:
            return task_dirs

        # Fallback for unknown layouts: any directory with metadata file.
        for candidate in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: str(p)):
            if any((candidate / name).exists() for name in ("metadata.json", "metadata.yaml", "metadata.yml", "task.json", "task.yaml", "task.yml")):
                task_dirs.append(candidate)
        return task_dirs

    def list_tasks(self) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        for task_dir in self._task_dirs():
            metadata = _load_metadata(task_dir)
            stage = metadata.get("stage") or metadata.get("phase")
            description = str(metadata.get("description") or "ReplicatorBench task")
            if stage:
                description = f"[{stage}] {description}"
            tasks.append(
                BenchmarkTask(
                    task_id=str(metadata.get("task_id") or task_dir.name),
                    title=str(metadata.get("title") or task_dir.name),
                    description=description,
                    paper_ref=str(metadata.get("paper_ref")) if metadata.get("paper_ref") else None,
                    rubric_path=(task_dir / "rubric.md") if (task_dir / "rubric.md").exists() else None,
                )
            )
        deduped: dict[str, BenchmarkTask] = {task.task_id: task for task in tasks}
        return [deduped[key] for key in sorted(deduped)]

    def _task_dir_for_id(self, task_id: str) -> Path | None:
        for task_dir in self._task_dirs():
            metadata = _load_metadata(task_dir)
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

        objective = (
            f"Replicate study: {task_obj.title}.\n"
            f"Description: {task_obj.description}"
        )
        proc = _run_agent_subprocess(
            project_id=project_id,
            objective=objective,
            max_steps=max_steps,
            repo_root=run_dir,
        )
        status = "success" if proc.returncode == 0 else "failed"
        return BenchmarkResult(
            task_id=task_obj.task_id,
            status=status,
            score=None,
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
