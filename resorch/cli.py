from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.ledger import Ledger
from resorch.paths import resolve_repo_paths
from resorch.projects import create_project, create_successor_project, get_project, list_projects, set_project_stage
from resorch.reviews import ingest_review_result, write_review_request
from resorch.tasks import create_task, get_task, list_tasks, run_task
from resorch.artifacts import list_artifacts as list_artifacts_fn
from resorch.artifacts import put_artifact as put_artifact_fn
from resorch.retrieval import fetch as fetch_fn
from resorch.jobs import create_job as create_job_fn
from resorch.jobs import get_job as get_job_fn
from resorch.jobs import list_jobs as list_jobs_fn
from resorch.jobs import poll_job as poll_job_fn
from resorch.jobs import run_job as run_job_fn
from resorch.webhooks import run_server as run_webhook_server
from resorch.ideas import import_ideas_jsonl as import_ideas_jsonl_fn
from resorch.ideas import get_idea as get_idea_fn
from resorch.ideas import list_ideas as list_ideas_fn
from resorch.ideas import set_idea_status as set_idea_status_fn
from resorch.ideas import score_ideas as score_ideas_fn
from resorch.ideas import dedupe_ideas_jsonl as dedupe_ideas_jsonl_fn
from resorch.idea_bank import build_idea_graph as build_idea_graph_fn
from resorch.idea_bank import format_idea_graph_dot as format_idea_graph_dot_fn
from resorch.idea_bank import link_ideas as link_ideas_fn
from resorch.idea_bank import park_idea as park_idea_fn
from resorch.idea_bank import revive_idea_to_new_project as revive_idea_to_new_project_fn
from resorch.idea_bank import spawn_idea as spawn_idea_fn
from resorch.playbook_store import get_playbook_entry as get_playbook_entry_fn
from resorch.playbook_store import list_playbook_entries as list_playbook_entries_fn
from resorch.playbook_store import put_playbook_entry as put_playbook_entry_fn
from resorch.playbook_extractor import extract_and_save as extract_playbook_entry_fn
from resorch.constraints import write_constraints_template as write_constraints_template_fn
from resorch.db_inventory import ensure_databases as ensure_databases_fn
from resorch.summary_ingest import ingest_summary as ingest_summary_fn
from resorch.stage_gates import compute_gate_env as compute_gate_env_fn
from resorch.stage_gates import evaluate_transitions as evaluate_transitions_fn
from resorch.stage_gates import load_stage_transitions as load_stage_transitions_fn
from resorch.stage_gates import Unknown as GateUnknown
from resorch.smoke_tests import ingest_smoke_test_result as ingest_smoke_test_result_fn
from resorch.smoke_tests import list_smoke_tests as list_smoke_tests_fn
from resorch.topic_brief import write_topic_brief as write_topic_brief_fn
from resorch.topic_engine_loop import run_topic_engine as run_topic_engine_fn
from resorch.idea_launcher import commit_and_launch as commit_and_launch_fn
from resorch.evidence_store import add_evidence as add_evidence_fn
from resorch.evidence_store import get_evidence as get_evidence_fn
from resorch.evidence_store import list_evidence as list_evidence_fn
from resorch.claims import create_claim as create_claim_fn
from resorch.autopilot import run_autopilot_iteration as run_autopilot_iteration_fn
from resorch.autopilot import load_review_policy as load_review_policy_fn
from resorch.agent_loop import run_agent_loop as run_agent_loop_fn
from resorch.manuscript_checker import check_manuscript_consistency as check_manuscript_consistency_fn
from resorch.manuscript_checker import write_consistency_report as write_consistency_report_fn
from resorch.verification_checklist import generate_verification_checklist as generate_verification_checklist_fn
from resorch.verification_checklist import write_checklist as write_checklist_fn
from resorch.visual_inspection import approve_visual_inspection as approve_visual_inspection_fn
from resorch.visual_inspection import get_visual_inspection_status as get_visual_inspection_status_fn
from resorch.cohort import run_cohort as run_cohort_fn
from resorch.submission_verifier import verify_submission as verify_submission_fn
from resorch.portfolio import run_portfolio_cycle as run_portfolio_cycle_fn


def _json_default(obj: Any) -> Any:
    if isinstance(obj, GateUnknown):
        return {"unknown": True, "reason": obj.reason}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default))


def _benchmark_task_to_dict(task: Any) -> Dict[str, Any]:
    rubric_path = getattr(task, "rubric_path", None)
    return {
        "task_id": str(getattr(task, "task_id", "")),
        "title": str(getattr(task, "title", "")),
        "description": str(getattr(task, "description", "")),
        "paper_ref": str(getattr(task, "paper_ref")) if getattr(task, "paper_ref", None) else None,
        "rubric_path": str(rubric_path) if rubric_path else None,
    }


def _benchmark_result_to_dict(result: Any) -> Dict[str, Any]:
    details = getattr(result, "details", {}) or {}
    if not isinstance(details, dict):
        details = {"value": str(details)}
    return {
        "task_id": str(getattr(result, "task_id", "")),
        "status": str(getattr(result, "status", "")),
        "score": getattr(result, "score", None),
        "details": {str(k): (str(v) if isinstance(v, Path) else v) for k, v in details.items()},
    }


def _build_benchmark_suite(suite_name: str, external_path: Optional[str]) -> Any:
    suite_key = str(suite_name or "").strip().lower()
    ext_path = Path(external_path).expanduser() if external_path else None

    if suite_key == "paperbench":
        from resorch.benchmarks.paperbench_adapter import PaperBenchSuite

        return PaperBenchSuite(external_path=ext_path)
    if suite_key == "airs":
        from resorch.benchmarks.airs_adapter import AIRSBenchSuite

        return AIRSBenchSuite(external_path=ext_path)
    if suite_key == "replicatorbench":
        from resorch.benchmarks.replicatorbench_adapter import ReplicatorBenchSuite

        return ReplicatorBenchSuite(external_path=ext_path)
    raise SystemExit(f"Unknown benchmark suite: {suite_name}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="orchestrator", description="Research Orchestrator (local-first)")
    ap.add_argument("--repo-root", default=None, help="Path to orchestrator repo root (defaults to auto-detect)")

    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Initialize local state (.orchestrator/ledger.db, logs, workspaces/)")
    p_init.set_defaults(_handler="init")

    p_doctor = sub.add_parser("doctor", help="Diagnose environment and configuration")
    p_doctor.set_defaults(_handler="doctor")

    p_proj = sub.add_parser("project", help="Project commands")
    proj_sub = p_proj.add_subparsers(dest="subcmd", required=True)

    p_proj_new = proj_sub.add_parser("new", help="Create a new project + workspace")
    p_proj_new.add_argument("--id", dest="project_id", default=None)
    p_proj_new.add_argument("--title", required=True)
    p_proj_new.add_argument("--domain", default="")
    p_proj_new.add_argument("--stage", default="intake")
    p_proj_new.add_argument("--idea-id", default=None)
    p_proj_new.add_argument(
        "--git-init",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Initialize a Git repo under the workspace (default: true)",
    )
    p_proj_new.set_defaults(_handler="project_new")

    p_proj_list = proj_sub.add_parser("list", help="List projects")
    p_proj_list.set_defaults(_handler="project_list")

    p_proj_open = proj_sub.add_parser("open", help="Show a project")
    p_proj_open.add_argument("project_id")
    p_proj_open.set_defaults(_handler="project_open")

    p_proj_set_stage = proj_sub.add_parser("set-stage", help="Update a project's stage")
    p_proj_set_stage.add_argument("project_id")
    p_proj_set_stage.add_argument("--stage", required=True)
    p_proj_set_stage.set_defaults(_handler="project_set_stage")

    p_proj_successor = proj_sub.add_parser(
        "create-successor",
        help="Create a successor project that inherits from an existing one",
    )
    p_proj_successor.add_argument(
        "--predecessor", required=True, help="ID of the predecessor project"
    )
    p_proj_successor.add_argument("--id", dest="project_id", default=None)
    p_proj_successor.add_argument("--title", default=None)
    p_proj_successor.add_argument("--domain", default=None)
    p_proj_successor.add_argument("--stage", default="intake")
    p_proj_successor.add_argument(
        "--inherit",
        nargs="*",
        default=None,
        help="Directories to inherit via symlink (default: data src configs)",
    )
    p_proj_successor.add_argument(
        "--git-init",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Initialize a Git repo under the workspace (default: true)",
    )
    p_proj_successor.set_defaults(_handler="project_create_successor")

    p_task = sub.add_parser("task", help="Task commands")
    task_sub = p_task.add_subparsers(dest="subcmd", required=True)

    p_task_create = task_sub.add_parser("create", help="Create a task")
    p_task_create.add_argument("--project", dest="project_id", required=True)
    p_task_create.add_argument("--type", dest="task_type", required=True)
    p_task_create.add_argument("--spec-json", default=None, help="Task spec as JSON string")
    p_task_create.add_argument("--spec-file", default=None, help="Task spec as JSON file path")
    p_task_create.set_defaults(_handler="task_create")

    p_task_run = task_sub.add_parser("run", help="Run a task")
    p_task_run.add_argument("task_id")
    p_task_run.set_defaults(_handler="task_run")

    p_task_status = task_sub.add_parser("status", help="List tasks (optionally filtered)")
    p_task_status.add_argument("--project", dest="project_id", default=None)
    p_task_status.set_defaults(_handler="task_status")

    p_review = sub.add_parser("review", help="Review commands")
    review_sub = p_review.add_subparsers(dest="subcmd", required=True)

    p_review_req = review_sub.add_parser("request", help="Generate a review request packet (Markdown + JSON)")
    p_review_req.add_argument("--project", dest="project_id", required=True)
    p_review_req.add_argument("--stage", required=True)
    p_review_req.add_argument("--mode", default="balanced", choices=["balanced", "devils_advocate", "supportive"],
                             help="Review tone (stored in packet metadata; does not change reviewer behavior yet)")
    p_review_req.add_argument("--target", dest="targets", action="append", default=[], help="Target artifact path (repeatable)")
    p_review_req.add_argument("--question", dest="questions", action="append", default=[], help="Question for reviewer (repeatable)")
    p_review_req.add_argument("--rubric", default=None, help="Path to rubric file")
    p_review_req.add_argument("--time-budget-minutes", type=int, default=None)
    p_review_req.set_defaults(_handler="review_request")

    p_review_redteam = review_sub.add_parser("redteam", help="Generate a redteam review request (Stage 5 gate)")
    p_review_redteam.add_argument("--project", dest="project_id", required=True)
    p_review_redteam.add_argument("--mode", default="devils_advocate", choices=["balanced", "devils_advocate", "supportive"],
                                 help="Review tone (stored in packet metadata; does not change reviewer behavior yet)")
    p_review_redteam.add_argument("--target", dest="targets", action="append", default=[], help="Target artifact path (repeatable)")
    p_review_redteam.add_argument("--question", dest="questions", action="append", default=[], help="Question for reviewer (repeatable)")
    p_review_redteam.add_argument(
        "--rubric",
        default="review/review_result.schema.json",
        help="Rubric/prompt file to embed in packet (default: review/review_result.schema.json)",
    )
    p_review_redteam.add_argument("--time-budget-minutes", type=int, default=30)
    p_review_redteam.set_defaults(_handler="review_redteam")

    p_review_ingest = review_sub.add_parser("ingest", help="Ingest a review result JSON into the ledger")
    p_review_ingest.add_argument("--result", required=True, help="Path to review_result.json (schema: review/review_result.schema.json)")
    p_review_ingest.set_defaults(_handler="review_ingest")

    p_artifact = sub.add_parser("artifact", help="Artifact commands")
    artifact_sub = p_artifact.add_subparsers(dest="subcmd", required=True)

    p_artifact_list = artifact_sub.add_parser("list", help="List artifacts registered in the ledger")
    p_artifact_list.add_argument("--project", dest="project_id", required=True)
    p_artifact_list.add_argument("--prefix", default=None)
    p_artifact_list.add_argument("--limit", type=int, default=200)
    p_artifact_list.set_defaults(_handler="artifact_list")

    p_artifact_get = artifact_sub.add_parser("get", help="Fetch an artifact file content")
    p_artifact_get.add_argument("--project", dest="project_id", required=True)
    p_artifact_get.add_argument("--path", required=True, help="Workspace-relative path")
    p_artifact_get.set_defaults(_handler="artifact_get")

    p_artifact_put = artifact_sub.add_parser("put", help="Write an artifact file and register it in the ledger")
    p_artifact_put.add_argument("--project", dest="project_id", required=True)
    p_artifact_put.add_argument("--path", required=True, help="Workspace-relative path")
    p_artifact_put.add_argument("--content", default=None)
    p_artifact_put.add_argument("--content-file", default=None)
    p_artifact_put.add_argument("--mode", default="overwrite", choices=["overwrite", "append"])
    p_artifact_put.add_argument("--kind", default=None)
    p_artifact_put.set_defaults(_handler="artifact_put")

    p_job = sub.add_parser("job", help="Background job commands (polling/webhooks)")
    job_sub = p_job.add_subparsers(dest="subcmd", required=True)

    p_job_create = job_sub.add_parser("create", help="Create a job")
    p_job_create.add_argument("--project", dest="project_id", default=None)
    p_job_create.add_argument("--provider", required=True, help="e.g., openai, anthropic")
    p_job_create.add_argument("--kind", required=True, help="e.g., response, review, deep_research")
    p_job_create.add_argument("--spec-json", default=None)
    p_job_create.add_argument("--spec-file", default=None)
    p_job_create.set_defaults(_handler="job_create")

    p_job_list = job_sub.add_parser("list", help="List jobs")
    p_job_list.add_argument("--project", dest="project_id", default=None)
    p_job_list.add_argument("--status", default=None)
    p_job_list.add_argument("--limit", type=int, default=200)
    p_job_list.set_defaults(_handler="job_list")

    p_job_get = job_sub.add_parser("get", help="Get a job")
    p_job_get.add_argument("job_id")
    p_job_get.set_defaults(_handler="job_get")

    p_job_run = job_sub.add_parser("run", help="Run a job now")
    p_job_run.add_argument("job_id")
    p_job_run.set_defaults(_handler="job_run")

    p_job_poll = job_sub.add_parser("poll", help="Poll a running/submitted job")
    p_job_poll.add_argument("job_id")
    p_job_poll.set_defaults(_handler="job_poll")

    p_webhook = sub.add_parser("webhook", help="Webhook receiver (HTTP)")
    webhook_sub = p_webhook.add_subparsers(dest="subcmd", required=True)

    p_webhook_serve = webhook_sub.add_parser("serve", help="Run webhook receiver HTTP server")
    p_webhook_serve.add_argument("--host", default="127.0.0.1")
    p_webhook_serve.add_argument("--port", type=int, default=8787)
    p_webhook_serve.set_defaults(_handler="webhook_serve")

    p_idea = sub.add_parser("idea", help="Topic Engine: idea records")
    idea_sub = p_idea.add_subparsers(dest="subcmd", required=True)

    p_idea_import = idea_sub.add_parser("import", help="Import idea records from a JSONL file into the ledger")
    p_idea_import.add_argument("--project", dest="project_id", required=True)
    p_idea_import.add_argument("--input", required=True, help="Workspace-relative (or absolute) JSONL path")
    p_idea_import.set_defaults(_handler="idea_import")

    p_idea_dedupe = idea_sub.add_parser("dedupe", help="Deduplicate idea records in a JSONL file (baseline lexical)")
    p_idea_dedupe.add_argument("--project", dest="project_id", required=True)
    p_idea_dedupe.add_argument("--input", required=True, help="Workspace-relative (or absolute) JSONL path")
    p_idea_dedupe.add_argument("--output", default="ideas/deduped.jsonl")
    p_idea_dedupe.add_argument("--mapping", default=None)
    p_idea_dedupe.add_argument("--threshold", type=float, default=0.9)
    p_idea_dedupe.set_defaults(_handler="idea_dedupe")

    p_idea_get = idea_sub.add_parser("get", help="Get an idea record by id")
    p_idea_get.add_argument("idea_id")
    p_idea_get.set_defaults(_handler="idea_get")

    p_idea_set_status = idea_sub.add_parser("set-status", help="Set an idea's status (updates both DB row and JSON)")
    p_idea_set_status.add_argument("idea_id")
    p_idea_set_status.add_argument(
        "--status",
        required=True,
        choices=["candidate", "active", "rejected", "smoke_passed", "selected", "in_progress", "parked", "done"],
    )
    p_idea_set_status.set_defaults(_handler="idea_set_status")

    p_idea_commit_launch = idea_sub.add_parser(
        "commit-and-launch",
        help="Commit an idea into a new project workspace and prepare it for agent run",
    )
    p_idea_commit_launch.add_argument("--idea-id", dest="idea_id", required=True)
    p_idea_commit_launch.add_argument("--title", default=None)
    p_idea_commit_launch.add_argument("--domain", default=None)
    p_idea_commit_launch.add_argument("--objective", default=None)
    p_idea_commit_launch.add_argument("--dry-run", action="store_true")
    p_idea_commit_launch.add_argument("--max-steps", type=int, default=1)
    p_idea_commit_launch.set_defaults(_handler="idea_commit_and_launch")

    p_idea_list = idea_sub.add_parser("list", help="List ideas (from the ledger)")
    p_idea_list.add_argument("--project", dest="project_id", required=True)
    p_idea_list.add_argument("--status", default=None)
    p_idea_list.add_argument("--limit", type=int, default=50)
    p_idea_list.set_defaults(_handler="idea_list")

    p_idea_score = idea_sub.add_parser("score", help="Score ideas using a rubric and write ranked JSONL")
    p_idea_score.add_argument("--project", dest="project_id", required=True)
    p_idea_score.add_argument("--rubric", default="rubrics/idea_score_rubric.yaml")
    p_idea_score.add_argument("--output", default="ideas/ranked.jsonl")
    p_idea_score.add_argument("--provider", default="arithmetic", choices=["arithmetic", "claude"], help="Scoring provider: arithmetic (default 2.5) or claude (LLM evaluation)")
    p_idea_score.set_defaults(_handler="idea_score")

    p_idea_link = idea_sub.add_parser("link", help="Create a lineage edge between two ideas")
    p_idea_link.add_argument("--src", dest="src_idea_id", required=True)
    p_idea_link.add_argument("--dst", dest="dst_idea_id", required=True)
    p_idea_link.add_argument("--relation", required=True, help="e.g. narrow|broaden|reframe|baseline_add|metric_swap|revive")
    p_idea_link.add_argument("--reason", default=None)
    p_idea_link.add_argument("--meta-json", default=None)
    p_idea_link.set_defaults(_handler="idea_link")

    p_idea_spawn = idea_sub.add_parser("spawn", help="Spawn a derived idea from a parent (mutation operator)")
    p_idea_spawn.add_argument("--parent", dest="parent_idea_id", required=True)
    p_idea_spawn.add_argument(
        "--operator",
        required=True,
        choices=["narrow", "broaden", "reframe", "baseline_add", "metric_swap"],
    )
    p_idea_spawn.add_argument("--project", dest="project_id", default=None, help="Target project_id (default: parent's project)")
    p_idea_spawn.add_argument("--new-id", dest="new_idea_id", default=None)
    p_idea_spawn.add_argument("--reason", default=None)
    p_idea_spawn.add_argument("--focus", default=None, help="For operator=narrow")
    p_idea_spawn.add_argument("--scope", default=None, help="For operator=broaden")
    p_idea_spawn.add_argument("--reframe", dest="reframe_text", default=None, help="For operator=reframe")
    p_idea_spawn.add_argument("--baseline", dest="baseline_add", default=None, help="For operator=baseline_add")
    p_idea_spawn.add_argument("--from", dest="metric_from", default=None, help="For operator=metric_swap")
    p_idea_spawn.add_argument("--to", dest="metric_to", default=None, help="For operator=metric_swap")
    p_idea_spawn.set_defaults(_handler="idea_spawn")

    p_idea_park = idea_sub.add_parser("park", help="Park an idea (and record reason / unblock conditions)")
    p_idea_park.add_argument("idea_id")
    p_idea_park.add_argument("--reason", dest="parked_reason", required=True)
    p_idea_park.add_argument("--unblock", dest="unblock_conditions", action="append", default=None)
    p_idea_park.add_argument("--next-check-date", default=None)
    p_idea_park.set_defaults(_handler="idea_park")

    p_idea_revive = idea_sub.add_parser("revive", help="Revive an idea into a new project (creates the project + child idea)")
    p_idea_revive.add_argument("idea_id")
    p_idea_revive.add_argument("--new-project", dest="new_project_id", required=True)
    p_idea_revive.add_argument("--title", dest="new_project_title", required=True)
    p_idea_revive.add_argument("--domain", default="")
    p_idea_revive.add_argument("--stage", default="intake")
    p_idea_revive.add_argument("--git-init", default=False, action=argparse.BooleanOptionalAction)
    p_idea_revive.add_argument("--new-idea-id", default=None)
    p_idea_revive.add_argument("--reason", default=None)
    p_idea_revive.set_defaults(_handler="idea_revive")

    p_idea_graph = idea_sub.add_parser("graph", help="Output the idea lineage graph")
    group_graph = p_idea_graph.add_mutually_exclusive_group(required=True)
    group_graph.add_argument("--project", dest="project_id", default=None)
    group_graph.add_argument("--root", dest="root_idea_id", default=None)
    p_idea_graph.add_argument("--format", default="json", choices=["json", "dot"])
    p_idea_graph.add_argument("--max-nodes", type=int, default=200)
    p_idea_graph.add_argument("--max-edges", type=int, default=500)
    p_idea_graph.set_defaults(_handler="idea_graph")

    p_playbook = sub.add_parser("playbook", help="Playbook (patterns/anti-patterns)")
    pb_sub = p_playbook.add_subparsers(dest="subcmd", required=True)

    p_pb_put = pb_sub.add_parser("put", help="Put (upsert) a playbook entry from YAML file")
    p_pb_put.add_argument("--file", required=True)
    p_pb_put.add_argument("--topic", default=None)
    p_pb_put.set_defaults(_handler="playbook_put")

    p_pb_get = pb_sub.add_parser("get", help="Get a playbook entry")
    p_pb_get.add_argument("entry_id")
    p_pb_get.set_defaults(_handler="playbook_get")

    p_pb_list = pb_sub.add_parser("list", help="List playbook entries")
    p_pb_list.add_argument("--topic", default=None)
    p_pb_list.add_argument("--limit", type=int, default=50)
    p_pb_list.set_defaults(_handler="playbook_list")

    p_pb_extract = pb_sub.add_parser("extract", help="Extract a playbook entry from project artifacts")
    p_pb_extract.add_argument("--project", dest="project_id", required=True)
    p_pb_extract.add_argument("--mode", default="compact", choices=["compact", "full"])
    p_pb_extract.set_defaults(_handler="playbook_extract")

    p_constraints = sub.add_parser("constraints", help="Topic Engine: constraints capture")
    con_sub = p_constraints.add_subparsers(dest="subcmd", required=True)

    p_con_init = con_sub.add_parser("init", help="Create a constraints.yaml template in the workspace")
    p_con_init.add_argument("--project", dest="project_id", required=True)
    p_con_init.add_argument("--path", default="constraints.yaml")
    p_con_init.add_argument("--overwrite", action="store_true")
    p_con_init.set_defaults(_handler="constraints_init")

    p_db = sub.add_parser("db", help="Data/DB manager: DB inventory (no auto-download)")
    db_sub = p_db.add_subparsers(dest="subcmd", required=True)

    p_db_ensure = db_sub.add_parser("ensure", help="Check databases in constraints.yaml and generate a manual provisioning script")
    p_db_ensure.add_argument("--project", dest="project_id", required=True)
    p_db_ensure.add_argument("--constraints", default="constraints.yaml")
    p_db_ensure.add_argument("--out-dir", default="runs/db")
    p_db_ensure.set_defaults(_handler="db_ensure")

    p_summary = sub.add_parser("summary", help="Ingest pipeline/experiment summary into scoreboard/digest")
    sum_sub = p_summary.add_subparsers(dest="subcmd", required=True)

    p_sum_ingest = sum_sub.add_parser("ingest", help="Ingest results/summary.json and update results/scoreboard.json + notes/analysis_digest.md")
    p_sum_ingest.add_argument("--project", dest="project_id", required=True)
    p_sum_ingest.add_argument("--path", default="results/summary.json")
    p_sum_ingest.add_argument(
        "--register-summary-artifact",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Register the summary file as an artifact (default: true)",
    )
    p_sum_ingest.set_defaults(_handler="summary_ingest")

    p_verify = sub.add_parser("verify", help="Verification checklist and manuscript consistency")
    verify_sub = p_verify.add_subparsers(dest="subcmd", required=True)

    p_verify_checklist = verify_sub.add_parser("checklist", help="Generate a PI verification checklist")
    p_verify_checklist.add_argument("--project", dest="project_id", required=True)
    p_verify_checklist.add_argument("--output", default="reviews/verification_checklist.md")
    p_verify_checklist.add_argument(
        "--include-manuscript-checks",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include manuscript-integrated checks (default: true)",
    )
    p_verify_checklist.set_defaults(_handler="verify_checklist")

    p_verify_consistency = verify_sub.add_parser("consistency", help="Run manuscript consistency checker")
    p_verify_consistency.add_argument("--project", dest="project_id", required=True)
    p_verify_consistency.add_argument("--output", default="results/manuscript_consistency_report.md")
    p_verify_consistency.set_defaults(_handler="verify_consistency")

    p_verify_submission = verify_sub.add_parser("submission", help="Run submission verifier (release gate)")
    p_verify_submission.add_argument("--project", dest="project_id", required=True)
    p_verify_submission.add_argument("--mode", default="quick", choices=["quick", "full"])
    p_verify_submission.set_defaults(_handler="verify_submission")

    p_visual = sub.add_parser("visual", help="Visual inspection gate (human approval marker)")
    vis_sub = p_visual.add_subparsers(dest="subcmd", required=True)

    p_vis_status = vis_sub.add_parser("status", help="Show pending figures that need human visual inspection")
    p_vis_status.add_argument("--project", dest="project_id", required=True)
    p_vis_status.set_defaults(_handler="visual_status")

    p_vis_approve = vis_sub.add_parser("approve", help="Write/refresh the visual inspection approval marker")
    p_vis_approve.add_argument("--project", dest="project_id", required=True)
    p_vis_approve.add_argument("--marker", default="results/fig/visual_inspection.ok")
    p_vis_approve.add_argument("--note", default="")
    p_vis_approve.set_defaults(_handler="visual_approve")

    p_stage = sub.add_parser("stage", help="Stage gates (Topic Engine)")
    stage_sub = p_stage.add_subparsers(dest="subcmd", required=True)

    p_stage_check = stage_sub.add_parser("check", help="Evaluate stage transitions against current ledger/workspace state")
    p_stage_check.add_argument("--project", dest="project_id", required=True)
    p_stage_check.add_argument("--config", default="configs/stage_transitions.yaml")
    p_stage_check.set_defaults(_handler="stage_check")

    p_smoke = sub.add_parser("smoke", help="Topic Engine: feasibility smoke tests")
    smoke_sub = p_smoke.add_subparsers(dest="subcmd", required=True)

    p_smoke_ingest = smoke_sub.add_parser("ingest", help="Ingest a smoke test result JSON into the ledger")
    p_smoke_ingest.add_argument("--project", dest="project_id", required=True)
    p_smoke_ingest.add_argument("--result", required=True, help="Workspace-relative (or absolute) JSON path")
    p_smoke_ingest.add_argument("--store-path", default=None, help="Workspace-relative path to store a normalized copy")
    p_smoke_ingest.add_argument(
        "--register-artifact",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Register the stored result as an artifact (default: true)",
    )
    p_smoke_ingest.add_argument(
        "--update-idea-status",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If verdict=pass, set idea.status=smoke_passed (default: true)",
    )
    p_smoke_ingest.set_defaults(_handler="smoke_ingest")

    p_smoke_list = smoke_sub.add_parser("list", help="List smoke test results (from the ledger)")
    p_smoke_list.add_argument("--project", dest="project_id", required=True)
    p_smoke_list.add_argument("--idea", dest="idea_id", default=None)
    p_smoke_list.add_argument("--limit", type=int, default=50)
    p_smoke_list.set_defaults(_handler="smoke_list")

    p_topic = sub.add_parser("topic", help="Topic Engine: commit artifacts")
    topic_sub = p_topic.add_subparsers(dest="subcmd", required=True)

    p_topic_brief = topic_sub.add_parser("brief", help="Generate topic_brief.md from an idea record")
    p_topic_brief.add_argument("--project", dest="project_id", required=True)
    p_topic_brief.add_argument("--idea", dest="idea_id", required=True)
    p_topic_brief.add_argument("--output", default="topic_brief.md")
    p_topic_brief.add_argument(
        "--register-artifact",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Register the generated brief as an artifact (default: true)",
    )
    p_topic_brief.add_argument(
        "--set-selected",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Also set idea.status=selected (default: false)",
    )
    p_topic_brief.set_defaults(_handler="topic_brief")

    p_topic_commit = topic_sub.add_parser("commit", help="Alias for topic brief --set-selected")
    p_topic_commit.add_argument("--project", dest="project_id", required=True)
    p_topic_commit.add_argument("--idea", dest="idea_id", required=True)
    p_topic_commit.add_argument("--output", default="topic_brief.md")
    p_topic_commit.add_argument(
        "--register-artifact",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Register the generated brief as an artifact (default: true)",
    )
    p_topic_commit.set_defaults(_handler="topic_commit")

    p_topic_engine = topic_sub.add_parser("engine", help="Run Topic Engine search loop")
    p_topic_engine.add_argument("--project", dest="project_id", required=True)
    p_topic_engine.add_argument("--rounds", type=int, default=3)
    p_topic_engine.add_argument("--dry-run", action="store_true")
    p_topic_engine.add_argument("--top-k", type=int, default=10, help="Number of top ideas to activate per cycle")
    p_topic_engine.set_defaults(_handler="topic_engine")

    p_evidence = sub.add_parser("evidence", help="Evidence store (provenance/citations)")
    ev_sub = p_evidence.add_subparsers(dest="subcmd", required=True)

    p_ev_add = ev_sub.add_parser("add", help="Add an evidence item (writes evidence/*.json and registers it)")
    p_ev_add.add_argument("--project", dest="project_id", required=True)
    p_ev_add.add_argument("--idea", dest="idea_id", default=None)
    p_ev_add.add_argument("--kind", required=True, choices=["paper", "blog", "doc", "dataset", "benchmark", "repo", "other"])
    p_ev_add.add_argument("--title", required=True)
    p_ev_add.add_argument("--url", required=True)
    p_ev_add.add_argument("--summary", required=True)
    p_ev_add.add_argument("--retrieved-at", default=None)
    p_ev_add.add_argument("--relevance", type=float, default=None)
    p_ev_add.add_argument("--meta-json", default=None)
    p_ev_add.add_argument("--output", default=None, help="Workspace-relative output path (default: evidence/<id>.json)")
    p_ev_add.add_argument(
        "--register-artifact",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Register the stored evidence JSON as an artifact (default: true)",
    )
    p_ev_add.set_defaults(_handler="evidence_add")

    p_ev_list = ev_sub.add_parser("list", help="List evidence items")
    p_ev_list.add_argument("--project", dest="project_id", required=True)
    p_ev_list.add_argument("--idea", dest="idea_id", default=None)
    p_ev_list.add_argument("--limit", type=int, default=50)
    p_ev_list.set_defaults(_handler="evidence_list")

    p_ev_get = ev_sub.add_parser("get", help="Get an evidence item by id")
    p_ev_get.add_argument("evidence_id")
    p_ev_get.set_defaults(_handler="evidence_get")

    p_claim = sub.add_parser("claim", help="Claims (claim -> evidence links)")
    claim_sub = p_claim.add_subparsers(dest="subcmd", required=True)

    p_claim_new = claim_sub.add_parser("new", help="Create a claim markdown file under claims/ and register it")
    p_claim_new.add_argument("--project", dest="project_id", required=True)
    p_claim_new.add_argument("--statement", required=True)
    p_claim_new.add_argument("--evidence", dest="evidence_ids", action="append", default=[], help="Evidence id (repeatable)")
    p_claim_new.add_argument("--path", default=None, help="Workspace-relative output path (default: claims/claim_###.md)")
    p_claim_new.add_argument("--overwrite", action="store_true")
    p_claim_new.add_argument(
        "--register-artifact",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Register the created claim as an artifact (default: true)",
    )
    p_claim_new.set_defaults(_handler="claim_new")

    p_bench = sub.add_parser("bench", help="External benchmark adapters (PaperBench/AIRS/ReplicatorBench)")
    bench_sub = p_bench.add_subparsers(dest="subcmd", required=True)

    p_bench_list = bench_sub.add_parser("list", help="List benchmark tasks available in an external suite")
    p_bench_list.add_argument("--suite", required=True, choices=["paperbench", "airs", "replicatorbench"])
    p_bench_list.add_argument("--external-path", default=None, help="Path to benchmark repo checkout")
    p_bench_list.set_defaults(_handler="bench_list")

    p_bench_run = bench_sub.add_parser("run", help="Run (or dry-run) one benchmark task via adapter")
    p_bench_run.add_argument("--suite", required=True, choices=["paperbench", "airs", "replicatorbench"])
    p_bench_run.add_argument("--task", required=True, help="Benchmark task id")
    p_bench_run.add_argument("--dry-run", action="store_true")
    p_bench_run.add_argument("--max-steps", type=int, default=20, help="Max agent loop steps (default: 20)")
    p_bench_run.add_argument("--external-path", default=None, help="Path to benchmark repo checkout")
    p_bench_run.set_defaults(_handler="bench_run")

    p_portfolio = sub.add_parser("portfolio", help="Portfolio manager (multi-project prioritization and execution)")
    portfolio_sub = p_portfolio.add_subparsers(dest="subcmd", required=True)

    p_portfolio_cycle = portfolio_sub.add_parser("cycle", help="Run one portfolio cycle")
    p_portfolio_cycle.add_argument("--max-projects", type=int, default=3)
    p_portfolio_cycle.add_argument("--steps-per-project", type=int, default=5)
    p_portfolio_cycle.add_argument("--dry-run", action="store_true")
    p_portfolio_cycle.set_defaults(_handler="portfolio_cycle")

    p_agent = sub.add_parser("agent", help="Agent loop (Planner -> execute -> review)")
    agent_sub = p_agent.add_subparsers(dest="subcmd", required=True)

    p_agent_run = agent_sub.add_parser("run", help="Run the agent loop for up to N steps")
    p_agent_run.add_argument("--project", dest="project_id", required=True)
    p_agent_run.add_argument("--objective", required=True)
    p_agent_run.add_argument("--max-steps", type=int, default=10)
    p_agent_run.add_argument("--max-actions", type=int, default=None)
    p_agent_run.add_argument("--model", default=None, help="Planner model (default: config/$OPENAI_PLANNER_MODEL)")
    p_agent_run.add_argument(
        "--background",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Use OpenAI background mode for planner call (default: from config)",
    )
    p_agent_run.add_argument("--dry-run", action="store_true")
    p_agent_run.add_argument("--config", default=None, help="Explicit agent_loop config path (auto-discovers workspace config if omitted)")
    p_agent_run.add_argument(
        "--reuse-last-plan",
        action="store_true",
        default=False,
        help="Skip the Planner on the first iteration and reuse the plan from the most recent step JSON. "
        "Useful when relaunching after reviewer/environment fixes where the plan itself was fine.",
    )
    p_agent_run.set_defaults(_handler="agent_run")

    p_cohort = sub.add_parser("cohort", help="Cohort mode (multiple student agents)")
    cohort_sub = p_cohort.add_subparsers(dest="subcmd", required=True)

    p_cohort_run = cohort_sub.add_parser("run", help="Run N agent loops and generate a lab-meeting artifact")
    p_cohort_run.add_argument("--project", dest="base_project_id", required=True)
    p_cohort_run.add_argument("--objective", required=True)
    p_cohort_run.add_argument("--n", type=int, default=3)
    p_cohort_run.add_argument("--max-steps", type=int, default=3)
    p_cohort_run.add_argument(
        "--dry-run",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run cohort agents in dry-run mode (default: true)",
    )
    p_cohort_run.add_argument("--config", default=None, help="Explicit agent_loop config path (auto-discovers workspace config if omitted)")
    p_cohort_run.add_argument("--ideas-per-agent", type=int, default=1)
    p_cohort_run.set_defaults(_handler="cohort_run")

    p_auto = sub.add_parser("autopilot", help="Autopilot loop (Planner -> Codex -> Review)")
    auto_sub = p_auto.add_subparsers(dest="subcmd", required=True)

    p_auto_run = auto_sub.add_parser("run", help="Run a single autopilot iteration (plan -> tasks)")
    p_auto_run.add_argument("--project", dest="project_id", required=True)
    p_auto_run.add_argument("--objective", required=True, help="Objective for this iteration (1-2 sentences).")
    p_auto_run.add_argument("--model", default=None, help="Planner model (default: $OPENAI_PLANNER_MODEL or gpt-5.2-pro)")
    p_auto_run.add_argument("--iteration", type=int, default=0, help="Iteration number (default: 0)")
    p_auto_run.add_argument("--max-actions", type=int, default=6)
    p_auto_run.add_argument("--dry-run", action="store_true", help="Only create plan + tasks; do not run tasks")
    p_auto_run.add_argument(
        "--background",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use OpenAI background mode for planner call (default: true)",
    )
    p_auto_run.set_defaults(_handler="autopilot_run")

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    paths = resolve_repo_paths(args.repo_root)

    ledger = Ledger(paths)
    atexit.register(ledger.close)

    if args._handler == "init":
        ledger.init()
        print(f"Initialized: {paths.db_path}")
        return 0

    if args._handler == "doctor":
        from resorch.doctor import print_doctor_summary, run_doctor

        result = run_doctor(repo_root=paths.root)
        print_doctor_summary(result)
        _print_json(result)
        return 0

    ledger.init()

    if args._handler == "project_new":
        project = create_project(
            ledger=ledger,
            project_id=args.project_id,
            title=args.title,
            domain=args.domain,
            stage=args.stage,
            git_init=args.git_init,
            idea_id=args.idea_id,
        )
        _print_json(project)
        return 0

    if args._handler == "project_list":
        _print_json(list_projects(ledger))
        return 0

    if args._handler == "project_open":
        _print_json(get_project(ledger, args.project_id))
        return 0

    if args._handler == "project_set_stage":
        _print_json(set_project_stage(ledger, args.project_id, str(args.stage)))
        return 0

    if args._handler == "project_create_successor":
        result = create_successor_project(
            ledger=ledger,
            predecessor_id=args.predecessor,
            project_id=args.project_id,
            title=args.title,
            domain=args.domain,
            stage=args.stage,
            git_init=args.git_init,
            inherit=args.inherit,
        )
        _print_json(result)
        return 0

    if args._handler == "task_create":
        spec: Dict[str, Any] = {}
        if args.spec_json and args.spec_file:
            raise SystemExit("Use only one of --spec-json or --spec-file.")
        if args.spec_json:
            spec = json.loads(args.spec_json)
        elif args.spec_file:
            spec = json.loads(Path(args.spec_file).read_text(encoding="utf-8"))
        task = create_task(ledger=ledger, project_id=args.project_id, task_type=args.task_type, spec=spec)
        _print_json(task)
        return 0

    if args._handler == "task_run":
        task = get_task(ledger, args.task_id)
        project = get_project(ledger, task["project_id"])
        result = run_task(ledger=ledger, project=project, task=task)
        _print_json(result)
        return 0

    if args._handler == "task_status":
        _print_json(list_tasks(ledger, project_id=args.project_id))
        return 0

    if args._handler == "review_request":
        if not args.targets:
            raise SystemExit("At least one --target is required.")
        if not args.questions:
            args.questions = ["Any major issues? Any missing baselines/related work?"]
        project = get_project(ledger, args.project_id)
        out = write_review_request(
            ledger=ledger,
            project=project,
            stage=args.stage,
            mode=args.mode,
            targets=args.targets,
            questions=args.questions,
            rubric=args.rubric,
            time_budget_minutes=args.time_budget_minutes,
        )
        _print_json(out)
        return 0

    if args._handler == "review_redteam":
        if not args.targets:
            args.targets = ["topic_brief.md"]
        if not args.questions:
            args.questions = [
                "List at least 10 reject reasons as findings. Use severities {blocker,major,minor,nit}.",
                "Suggest the top 3 cheapest fixes (as findings with suggested_fix).",
                "List 5 must-have experiments (as findings; include baseline names).",
            ]
        if not args.rubric:
            _redteam_prompt_path = paths.root / "prompts" / "reviewer_redteam.md"
            if _redteam_prompt_path.is_file():
                args.rubric = str(_redteam_prompt_path)
        project = get_project(ledger, args.project_id)
        out = write_review_request(
            ledger=ledger,
            project=project,
            stage="redteam",
            mode=args.mode,
            targets=args.targets,
            questions=args.questions,
            rubric=args.rubric,
            time_budget_minutes=args.time_budget_minutes,
        )
        _print_json(out)
        return 0

    if args._handler == "review_ingest":
        out = ingest_review_result(ledger=ledger, result_path=Path(args.result))
        _print_json(out)
        return 0

    if args._handler == "artifact_list":
        _print_json(list_artifacts_fn(ledger, project_id=args.project_id, prefix=args.prefix, limit=args.limit))
        return 0

    if args._handler == "artifact_get":
        _print_json(fetch_fn(ledger, id=f"artifact:{args.project_id}/{args.path}"))
        return 0

    if args._handler == "artifact_put":
        if args.content is not None and args.content_file is not None:
            raise SystemExit("Use only one of --content or --content-file.")
        if args.content is None and args.content_file is None:
            raise SystemExit("One of --content or --content-file is required.")
        content = args.content
        if args.content_file is not None:
            content = Path(args.content_file).read_text(encoding="utf-8")
        project = get_project(ledger, args.project_id)
        artifact = put_artifact_fn(
            ledger=ledger,
            project=project,
            relative_path=args.path,
            content=str(content),
            mode=args.mode,
            kind=args.kind,
        )
        _print_json(artifact)
        return 0

    if args._handler == "job_create":
        if args.spec_json and args.spec_file:
            raise SystemExit("Use only one of --spec-json or --spec-file.")
        spec: Dict[str, Any] = {}
        if args.spec_json:
            spec = json.loads(args.spec_json)
        elif args.spec_file:
            spec = json.loads(Path(args.spec_file).read_text(encoding="utf-8"))
        _print_json(
            create_job_fn(
                ledger=ledger,
                project_id=args.project_id,
                provider=args.provider,
                kind=args.kind,
                spec=spec,
            )
        )
        return 0

    if args._handler == "job_list":
        _print_json(list_jobs_fn(ledger, project_id=args.project_id, status=args.status, limit=args.limit))
        return 0

    if args._handler == "job_get":
        _print_json(get_job_fn(ledger, args.job_id))
        return 0

    if args._handler == "job_run":
        _print_json(run_job_fn(ledger=ledger, job_id=args.job_id))
        return 0

    if args._handler == "job_poll":
        _print_json(poll_job_fn(ledger=ledger, job_id=args.job_id))
        return 0

    if args._handler == "webhook_serve":
        run_webhook_server(ledger=ledger, host=str(args.host), port=int(args.port))
        return 0

    if args._handler == "idea_import":
        _print_json(import_ideas_jsonl_fn(ledger=ledger, project_id=args.project_id, input_path=args.input))
        return 0

    if args._handler == "idea_get":
        _print_json(get_idea_fn(ledger=ledger, idea_id=args.idea_id))
        return 0

    if args._handler == "idea_set_status":
        _print_json(set_idea_status_fn(ledger=ledger, idea_id=args.idea_id, status=args.status))
        return 0

    if args._handler == "idea_commit_and_launch":
        _print_json(
            commit_and_launch_fn(
                ledger=ledger,
                repo_paths=paths,
                idea_id=str(args.idea_id),
                project_title=args.title,
                domain=args.domain,
                objective=args.objective,
                dry_run=bool(args.dry_run),
                max_steps=int(args.max_steps),
            )
        )
        return 0

    if args._handler == "idea_dedupe":
        _print_json(
            dedupe_ideas_jsonl_fn(
                ledger=ledger,
                project_id=args.project_id,
                input_path=args.input,
                output_path=args.output,
                mapping_path=args.mapping,
                threshold=float(args.threshold),
            )
        )
        return 0

    if args._handler == "idea_list":
        _print_json(list_ideas_fn(ledger=ledger, project_id=args.project_id, status=args.status, limit=args.limit))
        return 0

    if args._handler == "idea_score":
        _print_json(
            score_ideas_fn(
                ledger=ledger,
                project_id=args.project_id,
                rubric_path=args.rubric,
                output_path=args.output,
                provider=args.provider,
            )
        )
        return 0

    if args._handler == "idea_link":
        meta = None
        if args.meta_json:
            try:
                meta = json.loads(str(args.meta_json))
            except json.JSONDecodeError:
                raise SystemExit("--meta-json must be valid JSON")
            if not isinstance(meta, dict):
                raise SystemExit("--meta-json must be a JSON object")
        _print_json(
            link_ideas_fn(
                ledger=ledger,
                src_idea_id=str(args.src_idea_id),
                dst_idea_id=str(args.dst_idea_id),
                relation=str(args.relation),
                reason=args.reason,
                meta=meta,
            )
        )
        return 0

    if args._handler == "idea_spawn":
        _print_json(
            spawn_idea_fn(
                ledger=ledger,
                parent_idea_id=str(args.parent_idea_id),
                operator=str(args.operator),
                project_id=args.project_id,
                new_idea_id=args.new_idea_id,
                reason=args.reason,
                narrow_focus=args.focus,
                broaden_scope=args.scope,
                reframe=args.reframe_text,
                baseline_add=args.baseline_add,
                metric_from=args.metric_from,
                metric_to=args.metric_to,
            )
        )
        return 0

    if args._handler == "idea_park":
        _print_json(
            park_idea_fn(
                ledger=ledger,
                idea_id=str(args.idea_id),
                parked_reason=str(args.parked_reason),
                unblock_conditions=args.unblock_conditions,
                next_check_date=args.next_check_date,
            )
        )
        return 0

    if args._handler == "idea_revive":
        _print_json(
            revive_idea_to_new_project_fn(
                ledger=ledger,
                idea_id=str(args.idea_id),
                new_project_id=str(args.new_project_id),
                new_project_title=str(args.new_project_title),
                domain=str(args.domain),
                stage=str(args.stage),
                git_init=bool(args.git_init),
                new_idea_id=args.new_idea_id,
                reason=args.reason,
            )
        )
        return 0

    if args._handler == "idea_graph":
        graph = build_idea_graph_fn(
            ledger=ledger,
            project_id=args.project_id,
            root_idea_id=args.root_idea_id,
            max_nodes=int(args.max_nodes),
            max_edges=int(args.max_edges),
        )
        if str(args.format) == "dot":
            print(format_idea_graph_dot_fn(graph), end="")
        else:
            _print_json(graph)
        return 0

    if args._handler == "playbook_put":
        _print_json(put_playbook_entry_fn(ledger=ledger, entry_path=args.file, topic=args.topic))
        return 0

    if args._handler == "playbook_get":
        _print_json(get_playbook_entry_fn(ledger, args.entry_id))
        return 0

    if args._handler == "playbook_list":
        _print_json(list_playbook_entries_fn(ledger, topic=args.topic, limit=args.limit))
        return 0

    if args._handler == "playbook_extract":
        _print_json(extract_playbook_entry_fn(ledger=ledger, project_id=args.project_id, mode=str(args.mode)))
        return 0

    if args._handler == "constraints_init":
        _print_json(
            write_constraints_template_fn(
                ledger=ledger,
                project_id=args.project_id,
                path=args.path,
                overwrite=bool(args.overwrite),
            )
        )
        return 0

    if args._handler == "db_ensure":
        _print_json(
            ensure_databases_fn(
                ledger=ledger,
                project_id=args.project_id,
                constraints_path=str(args.constraints),
                out_dir=str(args.out_dir),
            )
        )
        return 0

    if args._handler == "summary_ingest":
        _print_json(
            ingest_summary_fn(
                ledger=ledger,
                project_id=args.project_id,
                summary_path=str(args.path),
                register_summary_artifact=bool(args.register_summary_artifact),
            )
        )
        return 0

    if args._handler == "verify_checklist":
        project = get_project(ledger, args.project_id)
        ws = Path(project["repo_path"]).resolve()
        checklist = generate_verification_checklist_fn(
            workspace_dir=ws,
            project_id=str(args.project_id),
            include_manuscript_checks=bool(args.include_manuscript_checks),
        )
        out = write_checklist_fn(ws, checklist, output_path=Path(args.output))
        print(f"Wrote checklist: {out}")
        print(checklist.summary)
        return 0

    if args._handler == "verify_consistency":
        project = get_project(ledger, args.project_id)
        ws = Path(project["repo_path"]).resolve()
        report = check_manuscript_consistency_fn(ws)
        out = write_consistency_report_fn(ws, report, output_path=Path(args.output))
        print(f"Wrote consistency report: {out}")
        print(report.summary)
        return 0

    if args._handler == "verify_submission":
        result = verify_submission_fn(ledger=ledger, project_id=str(args.project_id), mode=str(args.mode))
        _print_json(result)
        return 0

    if args._handler == "visual_status":
        project = get_project(ledger, args.project_id)
        ws = Path(project["repo_path"]).resolve()
        policy = load_review_policy_fn(ledger.paths.root, workspace=ws)
        status = get_visual_inspection_status_fn(policy=policy, workspace=ws)
        _print_json({"enabled": status.enabled, "marker_path": status.marker_path, "pending_figures": status.pending_figures})
        return 0

    if args._handler == "visual_approve":
        _print_json(
            approve_visual_inspection_fn(
                ledger=ledger,
                project_id=args.project_id,
                marker_path=str(args.marker),
                note=str(args.note),
            )
        )
        return 0

    if args._handler == "stage_check":
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (ledger.paths.root / config_path).resolve()
        config = load_stage_transitions_fn(config_path)
        env = compute_gate_env_fn(ledger=ledger, project_id=args.project_id)
        _print_json(evaluate_transitions_fn(config=config, env=env))
        return 0

    if args._handler == "smoke_ingest":
        _print_json(
            ingest_smoke_test_result_fn(
                ledger=ledger,
                project_id=args.project_id,
                result_path=args.result,
                store_path=args.store_path,
                register_as_artifact=bool(args.register_artifact),
                update_idea_status_on_pass=bool(args.update_idea_status),
            )
        )
        return 0

    if args._handler == "smoke_list":
        _print_json(
            list_smoke_tests_fn(ledger=ledger, project_id=args.project_id, idea_id=args.idea_id, limit=args.limit)
        )
        return 0

    if args._handler == "topic_brief":
        _print_json(
            write_topic_brief_fn(
                ledger=ledger,
                project_id=args.project_id,
                idea_id=args.idea_id,
                output_path=args.output,
                register_as_artifact=bool(args.register_artifact),
                set_selected=bool(args.set_selected),
            )
        )
        return 0

    if args._handler == "topic_commit":
        _print_json(
            write_topic_brief_fn(
                ledger=ledger,
                project_id=args.project_id,
                idea_id=args.idea_id,
                output_path=args.output,
                register_as_artifact=bool(args.register_artifact),
                set_selected=True,
            )
        )
        return 0

    if args._handler == "topic_engine":
        _print_json(
            run_topic_engine_fn(
                ledger=ledger,
                project_id=args.project_id,
                rounds=int(args.rounds),
                dry_run=bool(args.dry_run),
                top_k=int(args.top_k),
            )
        )
        return 0

    if args._handler == "evidence_add":
        meta: Dict[str, Any] = {}
        if args.meta_json:
            meta = json.loads(args.meta_json)
            if not isinstance(meta, dict):
                raise SystemExit("--meta-json must be a JSON object")
        _print_json(
            add_evidence_fn(
                ledger=ledger,
                project_id=args.project_id,
                idea_id=args.idea_id,
                kind=args.kind,
                title=args.title,
                url=args.url,
                summary=args.summary,
                retrieved_at=args.retrieved_at,
                relevance=args.relevance,
                meta=meta,
                output_path=args.output,
                register_as_artifact=bool(args.register_artifact),
            )
        )
        return 0

    if args._handler == "evidence_list":
        _print_json(list_evidence_fn(ledger=ledger, project_id=args.project_id, idea_id=args.idea_id, limit=args.limit))
        return 0

    if args._handler == "evidence_get":
        _print_json(get_evidence_fn(ledger=ledger, evidence_id=args.evidence_id))
        return 0

    if args._handler == "claim_new":
        _print_json(
            create_claim_fn(
                ledger=ledger,
                project_id=args.project_id,
                statement=args.statement,
                evidence_ids=list(args.evidence_ids or []),
                path=args.path,
                overwrite=bool(args.overwrite),
                register_as_artifact=bool(args.register_artifact),
            )
        )
        return 0

    if args._handler == "bench_list":
        try:
            suite = _build_benchmark_suite(args.suite, args.external_path)
            tasks = suite.list_tasks()
        except ImportError as exc:
            print(f"Benchmark adapter import failed for suite '{args.suite}': {exc}", file=sys.stderr)
            return 2
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        _print_json(
            {
                "suite": suite.name,
                "description": suite.description,
                "external_path": str(suite.external_path),
                "tasks": [_benchmark_task_to_dict(task) for task in tasks],
            }
        )
        return 0

    if args._handler == "bench_run":
        try:
            suite = _build_benchmark_suite(args.suite, args.external_path)
            result = suite.run_task(
                str(args.task),
                workspace=paths.root,
                ledger=ledger,
                dry_run=bool(args.dry_run),
                max_steps=int(args.max_steps),
            )
        except ImportError as exc:
            print(f"Benchmark adapter import failed for suite '{args.suite}': {exc}", file=sys.stderr)
            return 2
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        payload = _benchmark_result_to_dict(result)
        results_path = paths.root / "results" / "bench" / str(suite.name) / f"{payload['task_id']}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        payload["result_path"] = str(results_path)
        _print_json(payload)
        return 0

    if args._handler == "portfolio_cycle":
        _print_json(
            run_portfolio_cycle_fn(
                ledger=ledger,
                max_projects=int(args.max_projects),
                steps_per_project=int(args.steps_per_project),
                dry_run=bool(args.dry_run),
            )
        )
        return 0

    if args._handler == "autopilot_run":
        model = args.model or os.environ.get("OPENAI_PLANNER_MODEL") or "gpt-5.2-pro"
        _print_json(
            run_autopilot_iteration_fn(
                ledger=ledger,
                project_id=args.project_id,
                objective=args.objective,
                model=model,
                iteration=int(args.iteration),
                dry_run=bool(args.dry_run),
                max_actions=int(args.max_actions),
                background=bool(args.background),
            )
        )
        return 0

    if args._handler == "agent_run":
        _print_json(
            run_agent_loop_fn(
                ledger=ledger,
                project_id=args.project_id,
                objective=args.objective,
                max_steps=int(args.max_steps),
                dry_run=bool(args.dry_run),
                config_path=args.config,
                model=args.model,
                background=args.background,
                max_actions=args.max_actions,
                reuse_last_plan=bool(args.reuse_last_plan),
            )
        )
        return 0

    if args._handler == "cohort_run":
        _print_json(
            run_cohort_fn(
                ledger=ledger,
                base_project_id=str(args.base_project_id),
                objective=str(args.objective),
                n=int(args.n),
                max_steps=int(args.max_steps),
                dry_run=bool(args.dry_run),
                config_path=str(args.config) if args.config else None,
                ideas_per_agent=int(args.ideas_per_agent),
            )
        )
        return 0

    ap.print_help(sys.stderr)
    return 2
