from __future__ import annotations

import json
from pathlib import Path

import pytest

from resorch.agent_loop import run_agent_loop
from resorch.benchmarks.airs_adapter import AIRSBenchSuite
from resorch.benchmarks.base import BenchmarkResult, BenchmarkSuite, BenchmarkTask
from resorch.benchmarks.paperbench_adapter import PaperBenchSuite
from resorch.benchmarks.replicatorbench_adapter import ReplicatorBenchSuite
from resorch.cli import build_parser
from resorch.ledger import Ledger
from resorch.paths import RepoPaths
from resorch.projects import list_projects


def _make_tmp_repo_ledger(tmp_path: Path) -> Ledger:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    ledger = Ledger(RepoPaths(root=repo_root))
    ledger.init()
    return ledger


class _DummySuite(BenchmarkSuite):
    def __init__(self, external_path: Path) -> None:
        super().__init__(name="dummy", description="dummy suite", external_path=external_path)

    def list_tasks(self) -> list[BenchmarkTask]:
        return [
            BenchmarkTask(
                task_id="dummy-1",
                title="Dummy Task",
                description="A dummy benchmark task",
                paper_ref=None,
                rubric_path=None,
            )
        ]

    def run_task(
        self,
        task: str | BenchmarkTask,
        workspace: Path,
        ledger: Ledger | None = None,
        dry_run: bool = False,
        max_steps: int = 20,
    ) -> BenchmarkResult:
        task_id = task.task_id if isinstance(task, BenchmarkTask) else str(task)
        return BenchmarkResult(task_id=task_id, status="skipped" if dry_run else "not_available", details={})


def test_benchmark_base_classes_can_be_instantiated(tmp_path: Path) -> None:
    task = BenchmarkTask(task_id="t1", title="Task 1", description="desc", paper_ref="arXiv:1234")
    result = BenchmarkResult(task_id="t1", status="success", score=0.75, details={"ok": True})
    suite = _DummySuite(external_path=tmp_path)

    fetched = suite.get_task("dummy-1")

    assert task.task_id == "t1"
    assert result.status == "success"
    assert fetched.task_id == "dummy-1"


def test_paperbench_suite_raises_when_external_path_missing(tmp_path: Path) -> None:
    suite = PaperBenchSuite(external_path=tmp_path / "missing-paperbench")
    with pytest.raises(FileNotFoundError):
        suite.list_tasks()


def test_airs_suite_raises_when_external_path_missing(tmp_path: Path) -> None:
    suite = AIRSBenchSuite(external_path=tmp_path / "missing-airs")
    with pytest.raises(FileNotFoundError):
        suite.list_tasks()


def test_replicatorbench_suite_raises_when_external_path_missing(tmp_path: Path) -> None:
    suite = ReplicatorBenchSuite(external_path=tmp_path / "missing-replicator")
    with pytest.raises(FileNotFoundError):
        suite.list_tasks()


def test_paperbench_lists_fake_paper_directory_with_metadata(tmp_path: Path) -> None:
    root = tmp_path / "paperbench"
    paper = root / "papers" / "paper-001"
    paper.mkdir(parents=True)
    (paper / "metadata.json").write_text(
        json.dumps(
            {
                "task_id": "pb-001",
                "title": "PaperBench Fake Task",
                "description": "minimal metadata",
                "paper_ref": "arXiv:2504.01848",
            }
        ),
        encoding="utf-8",
    )
    (paper / "rubric.md").write_text("# rubric\n", encoding="utf-8")

    suite = PaperBenchSuite(external_path=root)
    tasks = suite.list_tasks()

    assert len(tasks) == 1
    assert tasks[0].task_id == "pb-001"
    assert tasks[0].rubric_path is not None


def test_paperbench_handles_minimal_directory_without_metadata(tmp_path: Path) -> None:
    root = tmp_path / "paperbench"
    paper = root / "tasks" / "paper-alpha"
    paper.mkdir(parents=True)

    suite = PaperBenchSuite(external_path=root)
    task = suite.get_task("paper-alpha")

    assert task.task_id == "paper-alpha"
    assert task.title == "paper-alpha"


def test_airs_suite_lists_tasks_from_metadata_yaml(tmp_path: Path) -> None:
    root = tmp_path / "airs-bench"
    task_dir = root / "airsbench" / "tasks" / "airs_task_1"
    task_dir.mkdir(parents=True)
    (task_dir / "metadata.yaml").write_text(
        "\n".join(
            [
                "task_id: airs-001",
                "title: AIRS Fake Task",
                "description: Evaluate model behavior",
                "paper_ref: arXiv:2602.06855",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    suite = AIRSBenchSuite(external_path=root)
    tasks = suite.list_tasks()

    assert len(tasks) == 1
    assert tasks[0].task_id == "airs-001"
    assert tasks[0].title == "AIRS Fake Task"


def test_replicatorbench_suite_lists_task_definitions(tmp_path: Path) -> None:
    root = tmp_path / "replicatorbench"
    task_dir = root / "tasks" / "stage1_task_a"
    task_dir.mkdir(parents=True)
    (task_dir / "metadata.yaml").write_text(
        "\n".join(
            [
                "task_id: rep-001",
                "title: Replicator Stage 1",
                "description: Reproduce original setup",
                "stage: stage1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    suite = ReplicatorBenchSuite(external_path=root)
    tasks = suite.list_tasks()

    assert len(tasks) == 1
    assert tasks[0].task_id == "rep-001"
    assert tasks[0].description.startswith("[stage1]")


def test_setup_workspace_for_task_creates_expected_layout(tmp_path: Path) -> None:
    external_root = tmp_path / "external"
    external_root.mkdir()
    rubric_path = external_root / "rubric.md"
    rubric_path.write_text("# Rubric\n", encoding="utf-8")

    suite = _DummySuite(external_path=external_root)
    task = BenchmarkTask(
        task_id="dummy-1",
        title="Dummy Task",
        description="A dummy benchmark task",
        paper_ref="arXiv:1234.56789",
        rubric_path=rubric_path,
    )
    setup = suite.setup_workspace_for_task(task=task, workspace=tmp_path / "workspace")

    run_dir = Path(setup["run_dir"])
    assert run_dir.exists()
    assert (run_dir / "notes" / "problem.md").exists()
    assert (run_dir / "rubric.md").exists()
    assert setup["project_id"] is None


def test_paperbench_run_task_returns_success_with_mock_agent(tmp_path: Path, monkeypatch) -> None:
    ledger = _make_tmp_repo_ledger(tmp_path)
    root = tmp_path / "paperbench"
    task_dir = root / "papers" / "paper-001"
    task_dir.mkdir(parents=True)
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task_id": "pb-001",
                "title": "PaperBench Ready Task",
                "description": "minimal metadata",
                "paper_ref": "arXiv:2504.01848",
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "rubric.md").write_text("# rubric\n", encoding="utf-8")
    (task_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n")

    import subprocess
    import resorch.benchmarks.paperbench_adapter as pb_mod
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(pb_mod, "_run_agent_subprocess", lambda **kw: fake_proc)

    suite = PaperBenchSuite(external_path=root)
    result = suite.run_task("pb-001", workspace=tmp_path / "workspace", ledger=ledger)

    assert result.status == "success"
    run_dir = Path(result.details["run_dir"])
    assert (run_dir / "notes" / "problem.md").exists()
    assert (run_dir / "inputs" / "paper.pdf").exists()
    assert result.details.get("project_id") is not None


def test_bench_dry_run_produces_valid_workspace(tmp_path: Path, monkeypatch) -> None:
    ledger = _make_tmp_repo_ledger(tmp_path)
    workspace = tmp_path / "workspace"

    external_root = tmp_path / "external" / "paperbench"
    task_dir = external_root / "papers" / "paper-001"
    task_dir.mkdir(parents=True)
    (task_dir / "metadata.yaml").write_text(
        "\n".join(
            [
                "task_id: pb-001",
                "title: PaperBench Integration Task",
                "description: Validate benchmark adapter pipeline.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (task_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n")

    import subprocess
    import resorch.benchmarks.paperbench_adapter as pb_mod
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(pb_mod, "_run_agent_subprocess", lambda **kw: fake_proc)

    suite = PaperBenchSuite(external_path=external_root)
    task = suite.get_task("pb-001")
    result = suite.run_task(task, workspace=workspace, ledger=ledger, dry_run=False)

    assert result.status == "success"
    problem_md = workspace / "benchmarks" / "paperbench" / "pb-001" / "notes" / "problem.md"
    assert problem_md.exists()
    contents = problem_md.read_text(encoding="utf-8")
    assert "PaperBench Integration Task" in contents
    assert "Validate benchmark adapter pipeline." in contents

    project_id = result.details.get("project_id")
    assert project_id is not None
    assert str(project_id) in {p["id"] for p in list_projects(ledger)}


def test_bench_workspace_is_agent_loop_compatible(tmp_path: Path) -> None:
    ledger = _make_tmp_repo_ledger(tmp_path)
    workspace = tmp_path / "workspace"

    external_root = tmp_path / "external" / "paperbench"
    task_dir = external_root / "papers" / "paper-agent-loop"
    task_dir.mkdir(parents=True)
    (task_dir / "metadata.yaml").write_text(
        "\n".join(
            [
                "task_id: pb-agent",
                "title: Agent Loop Compatible Task",
                "description: Ensure benchmark workspace can feed into agent loop.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    suite = PaperBenchSuite(external_path=external_root)
    task = suite.get_task("pb-agent")
    setup = suite.setup_workspace_for_task(task=task, workspace=workspace, ledger=ledger)

    run_dir = Path(str(setup["run_dir"]))
    problem_md = run_dir / "notes" / "problem.md"
    assert problem_md.exists()
    assert problem_md.read_text(encoding="utf-8").strip() != ""

    # Validate this run directory can be used as a project workspace by agent_loop.
    ledger.insert_project(
        project_id="bench-agent-loop",
        title="Bench Agent Loop",
        domain="benchmark/paperbench",
        stage="intake",
        repo_path=str(run_dir),
        meta={},
    )
    out = run_agent_loop(
        ledger=ledger,
        project_id="bench-agent-loop",
        objective="compatibility check",
        max_steps=0,
        dry_run=True,
        config_path=None,
    )
    assert out["project_id"] == "bench-agent-loop"
    assert out["steps"] == []


def test_bench_handles_missing_external_gracefully(tmp_path: Path) -> None:
    suite = PaperBenchSuite(external_path=tmp_path / "missing-paperbench")

    try:
        tasks = suite.list_tasks()
        assert tasks == []
    except FileNotFoundError as exc:
        assert "benchmark path not found" in str(exc)

    fabricated = BenchmarkTask(
        task_id="pb-missing-001",
        title="Fabricated Missing Task",
        description="Missing external benchmark path",
    )
    result = suite.run_task(fabricated, workspace=tmp_path / "workspace", dry_run=False)
    assert result.status == "not_available"
    assert result.details.get("suite") == "paperbench"


def test_airs_run_task_returns_success_with_mock_agent(tmp_path: Path, monkeypatch) -> None:
    ledger = _make_tmp_repo_ledger(tmp_path)
    root = tmp_path / "airs-bench"
    task_dir = root / "airsbench" / "tasks" / "airs_task_1"
    task_dir.mkdir(parents=True)
    (task_dir / "metadata.yaml").write_text(
        "\n".join(
            [
                "task_id: airs-001",
                "title: AIRS Ready Task",
                "description: Evaluate model behavior",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (task_dir / "prepare.py").write_text("print('prepare')\n", encoding="utf-8")
    (task_dir / "evaluate.py").write_text("print('evaluate')\n", encoding="utf-8")

    import subprocess
    import resorch.benchmarks.airs_adapter as airs_mod
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(airs_mod, "_run_agent_subprocess", lambda **kw: fake_proc)

    suite = AIRSBenchSuite(external_path=root)
    result = suite.run_task("airs-001", workspace=tmp_path / "workspace", ledger=ledger)

    assert result.status == "success"
    run_dir = Path(result.details["run_dir"])
    assert (run_dir / "notes" / "problem.md").exists()
    assert (run_dir / "inputs" / "metadata.yaml").exists()
    assert result.details.get("project_id") is not None


def test_replicatorbench_run_task_returns_success_with_mock_agent(tmp_path: Path, monkeypatch) -> None:
    ledger = _make_tmp_repo_ledger(tmp_path)
    root = tmp_path / "replicatorbench"
    task_dir = root / "tasks" / "stage1_task_a"
    task_dir.mkdir(parents=True)
    (task_dir / "metadata.yaml").write_text(
        "\n".join(
            [
                "task_id: rep-001",
                "title: Replicator Ready Task",
                "description: Reproduce original setup",
                "stage: stage1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (task_dir / "rubric.md").write_text("# rubric\n", encoding="utf-8")

    import subprocess
    import resorch.benchmarks.replicatorbench_adapter as rep_mod
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(rep_mod, "_run_agent_subprocess", lambda **kw: fake_proc)

    suite = ReplicatorBenchSuite(external_path=root)
    result = suite.run_task("rep-001", workspace=tmp_path / "workspace", ledger=ledger)

    assert result.status == "success"
    run_dir = Path(result.details["run_dir"])
    assert (run_dir / "notes" / "problem.md").exists()
    assert (run_dir / "inputs" / "metadata.yaml").exists()
    assert result.details.get("project_id") is not None


def test_benchmark_run_task_dry_run_still_returns_skipped(tmp_path: Path) -> None:
    root = tmp_path / "paperbench"
    paper = root / "papers" / "paper-001"
    paper.mkdir(parents=True)
    (paper / "metadata.json").write_text(
        json.dumps({"task_id": "pb-001", "title": "Dry Run Task", "description": "dry run"}),
        encoding="utf-8",
    )

    suite = PaperBenchSuite(external_path=root)
    result = suite.run_task("pb-001", workspace=tmp_path / "workspace", dry_run=True)

    assert result.status == "skipped"


def test_bench_cli_parser_list_subcommand() -> None:
    args = build_parser().parse_args(["bench", "list", "--suite", "paperbench"])
    assert args._handler == "bench_list"
    assert args.suite == "paperbench"


def test_bench_cli_parser_run_subcommand() -> None:
    args = build_parser().parse_args(["bench", "run", "--suite", "airs", "--task", "airs-001", "--dry-run"])
    assert args._handler == "bench_run"
    assert args.suite == "airs"
    assert args.task == "airs-001"
    assert args.dry_run is True
    assert args.max_steps == 20


def test_bench_cli_parser_run_max_steps() -> None:
    args = build_parser().parse_args(["bench", "run", "--suite", "paperbench", "--task", "t1", "--max-steps", "50"])
    assert args.max_steps == 50


def test_paperbench_agent_exit_nonzero_returns_failed(tmp_path: Path, monkeypatch) -> None:
    ledger = _make_tmp_repo_ledger(tmp_path)
    root = tmp_path / "paperbench"
    task_dir = root / "papers" / "paper-fail"
    task_dir.mkdir(parents=True)
    (task_dir / "metadata.json").write_text(
        json.dumps({"task_id": "pb-fail", "title": "Fail Task", "description": "will fail"}),
        encoding="utf-8",
    )

    import subprocess
    import resorch.benchmarks.paperbench_adapter as pb_mod
    fake_proc = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")
    monkeypatch.setattr(pb_mod, "_run_agent_subprocess", lambda **kw: fake_proc)

    suite = PaperBenchSuite(external_path=root)
    result = suite.run_task("pb-fail", workspace=tmp_path / "workspace", ledger=ledger)

    assert result.status == "failed"
    assert result.details["returncode"] == 1


def test_airs_agent_exit_nonzero_returns_failed(tmp_path: Path, monkeypatch) -> None:
    ledger = _make_tmp_repo_ledger(tmp_path)
    root = tmp_path / "airs-bench"
    task_dir = root / "airsbench" / "tasks" / "airs_fail"
    task_dir.mkdir(parents=True)
    (task_dir / "metadata.yaml").write_text(
        "task_id: airs-fail\ntitle: AIRS Fail\ndescription: will fail\n",
        encoding="utf-8",
    )

    import subprocess
    import resorch.benchmarks.airs_adapter as airs_mod
    fake_proc = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")
    monkeypatch.setattr(airs_mod, "_run_agent_subprocess", lambda **kw: fake_proc)

    suite = AIRSBenchSuite(external_path=root)
    result = suite.run_task("airs-fail", workspace=tmp_path / "workspace", ledger=ledger)

    assert result.status == "failed"
    assert result.details["returncode"] == 1


def test_replicatorbench_agent_exit_nonzero_returns_failed(tmp_path: Path, monkeypatch) -> None:
    ledger = _make_tmp_repo_ledger(tmp_path)
    root = tmp_path / "replicatorbench"
    task_dir = root / "tasks" / "rep_fail"
    task_dir.mkdir(parents=True)
    (task_dir / "metadata.yaml").write_text(
        "task_id: rep-fail\ntitle: Rep Fail\ndescription: will fail\n",
        encoding="utf-8",
    )

    import subprocess
    import resorch.benchmarks.replicatorbench_adapter as rep_mod
    fake_proc = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")
    monkeypatch.setattr(rep_mod, "_run_agent_subprocess", lambda **kw: fake_proc)

    suite = ReplicatorBenchSuite(external_path=root)
    result = suite.run_task("rep-fail", workspace=tmp_path / "workspace", ledger=ledger)

    assert result.status == "failed"
    assert result.details["returncode"] == 1


def test_paperbench_no_ledger_returns_not_available(tmp_path: Path) -> None:
    """Without a ledger, no project_id is created, so run_task returns not_available."""
    root = tmp_path / "paperbench"
    task_dir = root / "papers" / "paper-001"
    task_dir.mkdir(parents=True)
    (task_dir / "metadata.json").write_text(
        json.dumps({"task_id": "pb-001", "title": "No Ledger", "description": "no ledger"}),
        encoding="utf-8",
    )

    suite = PaperBenchSuite(external_path=root)
    result = suite.run_task("pb-001", workspace=tmp_path / "workspace")

    assert result.status == "not_available"
