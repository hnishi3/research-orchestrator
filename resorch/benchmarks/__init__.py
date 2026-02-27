from __future__ import annotations

from resorch.benchmarks.airs_adapter import AIRSBenchSuite
from resorch.benchmarks.base import BenchmarkResult, BenchmarkSuite, BenchmarkTask
from resorch.benchmarks.paperbench_adapter import PaperBenchSuite
from resorch.benchmarks.replicatorbench_adapter import ReplicatorBenchSuite

__all__ = [
    "AIRSBenchSuite",
    "BenchmarkResult",
    "BenchmarkSuite",
    "BenchmarkTask",
    "PaperBenchSuite",
    "ReplicatorBenchSuite",
]
