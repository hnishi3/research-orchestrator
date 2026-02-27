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


def _read_metadata(task_dir: Path) -> dict[str, Any]:
    for name in ("metadata.json", "paper.json", "task.json"):
        fp = task_dir / name
        if fp.exists():
            try:
                payload = json.loads(fp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                return payload
    for name in ("metadata.yaml", "metadata.yml"):
        fp = task_dir / name
        if fp.exists():
            return _read_simple_yaml(fp)
    return {}


def _find_rubric(task_dir: Path) -> Path | None:
    for name in ("rubric.md", "rubric.yaml", "rubric.yml", "rubric.json"):
        fp = task_dir / name
        if fp.exists():
            return fp
    return None


class PaperBenchSuite(BenchmarkSuite):
    def __init__(self, external_path: Path | None = None) -> None:
        super().__init__(
            name="paperbench",
            description=(
                "PaperBench (Staab et al., arXiv:2504.01848): ICML 2024 "
                "paper reproduction benchmark."
            ),
            external_path=external_path or Path("external/paperbench"),
        )

    def _task_dirs(self) -> list[Path]:
        root = self._require_external_path(
            hint="Clone/download PaperBench under external/paperbench or pass --external-path."
        )
        candidates: list[Path] = []
        for rel in ("papers", "tasks"):
            base = root / rel
            if base.exists() and base.is_dir():
                candidates.extend(sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name))
        if candidates:
            return candidates
        return sorted(
            [
                p
                for p in root.iterdir()
                if p.is_dir() and not p.name.startswith(".") and p.name not in {".git", "docs", "scripts"}
            ],
            key=lambda p: p.name,
        )

    def list_tasks(self) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        for task_dir in self._task_dirs():
            metadata = _read_metadata(task_dir)
            paper_ref = metadata.get("paper_ref") or metadata.get("arxiv") or metadata.get("paper")
            task = BenchmarkTask(
                task_id=str(metadata.get("task_id") or task_dir.name),
                title=str(metadata.get("title") or metadata.get("paper_title") or task_dir.name),
                description=str(
                    metadata.get("description")
                    or metadata.get("summary")
                    or f"PaperBench task at {task_dir.name}"
                ),
                paper_ref=str(paper_ref) if paper_ref else None,
                rubric_path=_find_rubric(task_dir),
            )
            tasks.append(task)
        return tasks

    def _task_dir_for_id(self, task_id: str) -> Path | None:
        for task_dir in self._task_dirs():
            metadata = _read_metadata(task_dir)
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

            copied_files: list[str] = []
            for filename in (
                "metadata.json",
                "paper.json",
                "task.json",
                "metadata.yaml",
                "metadata.yml",
                "rubric.md",
                "rubric.yaml",
                "rubric.yml",
                "rubric.json",
            ):
                src = task_dir / filename
                if src.exists() and src.is_file():
                    shutil.copy2(src, inputs_dir / src.name)
                    copied_files.append(src.name)

            for pdf in sorted(task_dir.glob("*.pdf"), key=lambda p: p.name):
                shutil.copy2(pdf, inputs_dir / pdf.name)
                copied_files.append(pdf.name)
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
            f"Reproduce the paper: {task_obj.title}.\n"
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
                "copied_files": copied_files,
                "returncode": proc.returncode,
            },
        )
