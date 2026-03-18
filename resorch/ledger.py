from __future__ import annotations

import logging
import time
from contextlib import contextmanager
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from resorch.paths import RepoPaths
from resorch.utils import utc_now_iso

log = logging.getLogger(__name__)

_DB_LOCK_MAX_RETRIES = 5
_DB_LOCK_BACKOFF_BASE = 2.0  # seconds


def _dict_row_factory(cursor: sqlite3.Cursor, row: Tuple[Any, ...]) -> Dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


@dataclass
class Ledger:
    paths: RepoPaths
    _conn: Optional[sqlite3.Connection] = None
    _txn_depth: int = 0

    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            db = self.paths.db_path
            db.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db), timeout=120)
            self._conn.row_factory = _dict_row_factory
            self._conn.execute("PRAGMA foreign_keys = ON;")
            # Safer defaults for concurrent readers/writers across CLI + webhook processes.
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.execute("PRAGMA busy_timeout = 120000;")
            self._conn.isolation_level = "DEFERRED"
        return self._conn

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute SQL with retry on 'database is locked'.

        Retries up to _DB_LOCK_MAX_RETRIES times with exponential backoff
        on top of the 120s busy_timeout already configured at connection level.
        Total worst-case wait: ~120s (busy_timeout) × 5 (retries) + backoff ≈ 12 min.
        """
        for attempt in range(_DB_LOCK_MAX_RETRIES):
            try:
                return self.conn().execute(sql, params)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < _DB_LOCK_MAX_RETRIES - 1:
                    wait = _DB_LOCK_BACKOFF_BASE * (2 ** attempt)
                    log.warning("DB locked on execute (attempt %d/%d), retrying in %.1fs: %s",
                                attempt + 1, _DB_LOCK_MAX_RETRIES, wait, sql[:80])
                    time.sleep(wait)
                else:
                    raise
        raise sqlite3.OperationalError("database is locked (exhausted retries)")

    def _maybe_commit(self) -> None:
        if self._txn_depth <= 0:
            for attempt in range(_DB_LOCK_MAX_RETRIES):
                try:
                    self.conn().commit()
                    return
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and attempt < _DB_LOCK_MAX_RETRIES - 1:
                        wait = _DB_LOCK_BACKOFF_BASE * (2 ** attempt)
                        log.warning("DB locked on commit (attempt %d/%d), retrying in %.1fs",
                                    attempt + 1, _DB_LOCK_MAX_RETRIES, wait)
                        time.sleep(wait)
                    else:
                        raise

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group multiple Ledger writes into a single transaction.

        Ledger methods are written to auto-commit by default. When executed
        inside this context manager, internal commits are suppressed and the
        outermost transaction commits once at the end (or rolls back on error).
        """

        conn = self.conn()
        outermost = self._txn_depth == 0
        self._txn_depth += 1
        if outermost:
            if not conn.in_transaction:
                conn.execute("BEGIN")
        try:
            yield
        except Exception:  # noqa: BLE001
            self._txn_depth -= 1
            if outermost:
                conn.rollback()
            raise
        else:
            self._txn_depth -= 1
            if outermost:
                conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def init(self) -> None:
        self.paths.state_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.workspaces_dir.mkdir(parents=True, exist_ok=True)

        conn = self.conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            """
        )
        conn.commit()

        schema_version = self.get_meta("schema_version")
        if schema_version is None:
            self._create_schema_v1()
            self.set_meta("schema_version", "1")
            conn.commit()
            schema_version = "1"

        _migrations = [
            ("1", "2", self._migrate_1_to_2),
            ("2", "3", self._migrate_2_to_3),
            ("3", "4", self._migrate_3_to_4),
            ("4", "5", self._migrate_4_to_5),
            ("5", "6", self._migrate_5_to_6),
            ("6", "7", self._migrate_6_to_7),
        ]
        for from_ver, to_ver, migrate_fn in _migrations:
            if schema_version == from_ver:
                with self.transaction():
                    migrate_fn()
                    self.set_meta("schema_version", to_ver)
                schema_version = to_ver

        if schema_version != "7":
            raise SystemExit(f"Unsupported ledger schema_version={schema_version}.")

    def get_meta(self, key: str) -> Optional[str]:
        row = self._exec("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        self._exec(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._maybe_commit()

    def _create_schema_v1(self) -> None:
        conn = self.conn()

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              domain TEXT NOT NULL,
              stage TEXT NOT NULL,
              repo_path TEXT NOT NULL,
              meta_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_stage ON projects(stage);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL,
              type TEXT NOT NULL,
              status TEXT NOT NULL,
              spec_json TEXT NOT NULL,
              deps_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON tasks(project_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_runs (
              id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              status TEXT NOT NULL,
              exit_code INTEGER,
              jsonl_path TEXT,
              last_message_path TEXT,
              meta_json TEXT NOT NULL,
              FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_task_runs_task_id ON task_runs(task_id);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              task_run_id TEXT NOT NULL,
              ts TEXT NOT NULL,
              event_type TEXT NOT NULL,
              data_json TEXT NOT NULL,
              FOREIGN KEY(task_run_id) REFERENCES task_runs(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_task_events_run_id ON task_events(task_run_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_task_events_type ON task_events(event_type);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL,
              kind TEXT NOT NULL,
              path TEXT NOT NULL,
              sha256 TEXT,
              meta_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_project_id ON artifacts(project_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL,
              stage TEXT NOT NULL,
              reviewer TEXT NOT NULL,
              rubric_json TEXT NOT NULL,
              findings_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_project_id ON reviews(project_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_stage ON reviews(stage);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS playbook (
              id TEXT PRIMARY KEY,
              topic TEXT NOT NULL,
              rule_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_playbook_topic ON playbook(topic);")

    def _migrate_1_to_2(self) -> None:
        conn = self.conn()

        # Background jobs for long-running tasks (polling/webhooks).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              project_id TEXT,
              provider TEXT NOT NULL,
              kind TEXT NOT NULL,
              status TEXT NOT NULL,
              spec_json TEXT NOT NULL,
              remote_id TEXT,
              result_json TEXT,
              error TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              started_at TEXT,
              finished_at TEXT,
              FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_project_id ON jobs(project_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_provider ON jobs(provider);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              job_id TEXT NOT NULL,
              ts TEXT NOT NULL,
              event_type TEXT NOT NULL,
              data_json TEXT NOT NULL,
              FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_job_events_type ON job_events(event_type);")

    def _migrate_2_to_3(self) -> None:
        conn = self.conn()

        # Topic engine: idea records (JSON) per project.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ideas (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL,
              status TEXT NOT NULL,
              score_total REAL,
              data_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ideas_project_id ON ideas(project_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas(status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ideas_score_total ON ideas(score_total);")

    def _migrate_3_to_4(self) -> None:
        conn = self.conn()

        # Topic engine: smoke test results per idea (may have multiple runs).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS smoke_tests (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              idea_id TEXT NOT NULL,
              project_id TEXT NOT NULL,
              verdict TEXT NOT NULL,
              started_at TEXT NOT NULL,
              completed_at TEXT,
              result_json TEXT NOT NULL,
              artifact_path TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
              FOREIGN KEY(idea_id) REFERENCES ideas(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_smoke_tests_project_id ON smoke_tests(project_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_smoke_tests_idea_id ON smoke_tests(idea_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_smoke_tests_verdict ON smoke_tests(verdict);")

    def _migrate_4_to_5(self) -> None:
        conn = self.conn()

        # Provenance/citations: evidence items (URLs + summaries) that can be linked from claims/ideas.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL,
              idea_id TEXT,
              kind TEXT NOT NULL,
              title TEXT NOT NULL,
              url TEXT NOT NULL,
              retrieved_at TEXT NOT NULL,
              summary TEXT NOT NULL,
              relevance REAL,
              meta_json TEXT NOT NULL,
              artifact_path TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
              FOREIGN KEY(idea_id) REFERENCES ideas(id) ON DELETE SET NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_project_id ON evidence(project_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_idea_id ON evidence(idea_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_kind ON evidence(kind);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_url ON evidence(url);")

    def _migrate_5_to_6(self) -> None:
        conn = self.conn()

        # Idea lineage graph: edges between ideas (branching / revive / derivations).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS idea_edges (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              src_idea_id TEXT NOT NULL,
              dst_idea_id TEXT NOT NULL,
              relation TEXT NOT NULL,
              reason TEXT,
              meta_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(src_idea_id) REFERENCES ideas(id) ON DELETE CASCADE,
              FOREIGN KEY(dst_idea_id) REFERENCES ideas(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_idea_edges_src ON idea_edges(src_idea_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_idea_edges_dst ON idea_edges(dst_idea_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_idea_edges_relation ON idea_edges(relation);")

    def _migrate_6_to_7(self) -> None:
        conn = self.conn()

        # Add extracted text columns used by external-content FTS indexes.
        project_cols = conn.execute("PRAGMA table_info(projects);").fetchall()
        if not any(str(c["name"]) == "description" for c in project_cols):
            conn.execute("ALTER TABLE projects ADD COLUMN description TEXT NOT NULL DEFAULT '';")
            conn.execute(
                "UPDATE projects SET description = COALESCE(json_extract(meta_json, '$.description'), '') WHERE description = ''"
            )

        idea_cols = conn.execute("PRAGMA table_info(ideas);").fetchall()
        if not any(str(c["name"]) == "title" for c in idea_cols):
            conn.execute("ALTER TABLE ideas ADD COLUMN title TEXT NOT NULL DEFAULT '';")
            conn.execute("UPDATE ideas SET title = COALESCE(json_extract(data_json, '$.title'), '') WHERE title = ''")
        if not any(str(c["name"]) == "abstract" for c in idea_cols):
            conn.execute("ALTER TABLE ideas ADD COLUMN abstract TEXT NOT NULL DEFAULT '';")
            conn.execute(
                "UPDATE ideas SET abstract = COALESCE(json_extract(data_json, '$.abstract'), '') WHERE abstract = ''"
            )

        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_projects
                USING fts5(
                  title,
                  domain,
                  description,
                  stage,
                  content='projects',
                  content_rowid='rowid',
                  tokenize='unicode61'
                );
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_tasks
                USING fts5(
                  type,
                  status,
                  spec_json,
                  content='tasks',
                  content_rowid='rowid',
                  tokenize='unicode61'
                );
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_ideas
                USING fts5(
                  id,
                  status,
                  data_json,
                  title,
                  abstract,
                  content='ideas',
                  content_rowid='rowid',
                  tokenize='unicode61'
                );
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_evidence
                USING fts5(
                  kind,
                  title,
                  summary,
                  url,
                  content='evidence',
                  content_rowid='rowid',
                  tokenize='unicode61'
                );
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_reviews
                USING fts5(
                  stage,
                  reviewer,
                  rubric_json,
                  findings_json,
                  content='reviews',
                  content_rowid='rowid',
                  tokenize='unicode61'
                );
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_artifacts
                USING fts5(
                  kind,
                  meta_json,
                  path,
                  content='artifacts',
                  content_rowid='rowid',
                  tokenize='unicode61'
                );
                """
            )
        except sqlite3.OperationalError as e:
            if "fts5" in str(e).lower():
                log.warning("SQLite FTS5 unavailable in this environment; retrieval.search() will use LIKE fallback.")
                return
            raise

        conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS projects_fts_ai AFTER INSERT ON projects BEGIN
              INSERT INTO fts_projects(rowid, title, domain, description, stage)
              VALUES (new.rowid, new.title, new.domain, new.description, new.stage);
            END;
            CREATE TRIGGER IF NOT EXISTS projects_fts_ad AFTER DELETE ON projects BEGIN
              INSERT INTO fts_projects(fts_projects, rowid, title, domain, description, stage)
              VALUES ('delete', old.rowid, old.title, old.domain, old.description, old.stage);
            END;
            CREATE TRIGGER IF NOT EXISTS projects_fts_au AFTER UPDATE ON projects BEGIN
              INSERT INTO fts_projects(fts_projects, rowid, title, domain, description, stage)
              VALUES ('delete', old.rowid, old.title, old.domain, old.description, old.stage);
              INSERT INTO fts_projects(rowid, title, domain, description, stage)
              VALUES (new.rowid, new.title, new.domain, new.description, new.stage);
            END;

            CREATE TRIGGER IF NOT EXISTS tasks_fts_ai AFTER INSERT ON tasks BEGIN
              INSERT INTO fts_tasks(rowid, type, status, spec_json)
              VALUES (new.rowid, new.type, new.status, new.spec_json);
            END;
            CREATE TRIGGER IF NOT EXISTS tasks_fts_ad AFTER DELETE ON tasks BEGIN
              INSERT INTO fts_tasks(fts_tasks, rowid, type, status, spec_json)
              VALUES ('delete', old.rowid, old.type, old.status, old.spec_json);
            END;
            CREATE TRIGGER IF NOT EXISTS tasks_fts_au AFTER UPDATE ON tasks BEGIN
              INSERT INTO fts_tasks(fts_tasks, rowid, type, status, spec_json)
              VALUES ('delete', old.rowid, old.type, old.status, old.spec_json);
              INSERT INTO fts_tasks(rowid, type, status, spec_json)
              VALUES (new.rowid, new.type, new.status, new.spec_json);
            END;

            CREATE TRIGGER IF NOT EXISTS ideas_fts_ai AFTER INSERT ON ideas BEGIN
              INSERT INTO fts_ideas(rowid, id, status, data_json, title, abstract)
              VALUES (new.rowid, new.id, new.status, new.data_json, new.title, new.abstract);
            END;
            CREATE TRIGGER IF NOT EXISTS ideas_fts_ad AFTER DELETE ON ideas BEGIN
              INSERT INTO fts_ideas(fts_ideas, rowid, id, status, data_json, title, abstract)
              VALUES ('delete', old.rowid, old.id, old.status, old.data_json, old.title, old.abstract);
            END;
            CREATE TRIGGER IF NOT EXISTS ideas_fts_au AFTER UPDATE ON ideas BEGIN
              INSERT INTO fts_ideas(fts_ideas, rowid, id, status, data_json, title, abstract)
              VALUES ('delete', old.rowid, old.id, old.status, old.data_json, old.title, old.abstract);
              INSERT INTO fts_ideas(rowid, id, status, data_json, title, abstract)
              VALUES (new.rowid, new.id, new.status, new.data_json, new.title, new.abstract);
            END;

            CREATE TRIGGER IF NOT EXISTS evidence_fts_ai AFTER INSERT ON evidence BEGIN
              INSERT INTO fts_evidence(rowid, kind, title, summary, url)
              VALUES (new.rowid, new.kind, new.title, new.summary, new.url);
            END;
            CREATE TRIGGER IF NOT EXISTS evidence_fts_ad AFTER DELETE ON evidence BEGIN
              INSERT INTO fts_evidence(fts_evidence, rowid, kind, title, summary, url)
              VALUES ('delete', old.rowid, old.kind, old.title, old.summary, old.url);
            END;
            CREATE TRIGGER IF NOT EXISTS evidence_fts_au AFTER UPDATE ON evidence BEGIN
              INSERT INTO fts_evidence(fts_evidence, rowid, kind, title, summary, url)
              VALUES ('delete', old.rowid, old.kind, old.title, old.summary, old.url);
              INSERT INTO fts_evidence(rowid, kind, title, summary, url)
              VALUES (new.rowid, new.kind, new.title, new.summary, new.url);
            END;

            CREATE TRIGGER IF NOT EXISTS reviews_fts_ai AFTER INSERT ON reviews BEGIN
              INSERT INTO fts_reviews(rowid, stage, reviewer, rubric_json, findings_json)
              VALUES (new.rowid, new.stage, new.reviewer, new.rubric_json, new.findings_json);
            END;
            CREATE TRIGGER IF NOT EXISTS reviews_fts_ad AFTER DELETE ON reviews BEGIN
              INSERT INTO fts_reviews(fts_reviews, rowid, stage, reviewer, rubric_json, findings_json)
              VALUES ('delete', old.rowid, old.stage, old.reviewer, old.rubric_json, old.findings_json);
            END;
            CREATE TRIGGER IF NOT EXISTS reviews_fts_au AFTER UPDATE ON reviews BEGIN
              INSERT INTO fts_reviews(fts_reviews, rowid, stage, reviewer, rubric_json, findings_json)
              VALUES ('delete', old.rowid, old.stage, old.reviewer, old.rubric_json, old.findings_json);
              INSERT INTO fts_reviews(rowid, stage, reviewer, rubric_json, findings_json)
              VALUES (new.rowid, new.stage, new.reviewer, new.rubric_json, new.findings_json);
            END;

            CREATE TRIGGER IF NOT EXISTS artifacts_fts_ai AFTER INSERT ON artifacts BEGIN
              INSERT INTO fts_artifacts(rowid, kind, meta_json, path)
              VALUES (new.rowid, new.kind, new.meta_json, new.path);
            END;
            CREATE TRIGGER IF NOT EXISTS artifacts_fts_ad AFTER DELETE ON artifacts BEGIN
              INSERT INTO fts_artifacts(fts_artifacts, rowid, kind, meta_json, path)
              VALUES ('delete', old.rowid, old.kind, old.meta_json, old.path);
            END;
            CREATE TRIGGER IF NOT EXISTS artifacts_fts_au AFTER UPDATE ON artifacts BEGIN
              INSERT INTO fts_artifacts(fts_artifacts, rowid, kind, meta_json, path)
              VALUES ('delete', old.rowid, old.kind, old.meta_json, old.path);
              INSERT INTO fts_artifacts(rowid, kind, meta_json, path)
              VALUES (new.rowid, new.kind, new.meta_json, new.path);
            END;
            """
        )

        # Populate external-content FTS indexes for pre-existing rows.
        conn.execute("INSERT INTO fts_projects(fts_projects) VALUES('rebuild');")
        conn.execute("INSERT INTO fts_tasks(fts_tasks) VALUES('rebuild');")
        conn.execute("INSERT INTO fts_ideas(fts_ideas) VALUES('rebuild');")
        conn.execute("INSERT INTO fts_evidence(fts_evidence) VALUES('rebuild');")
        conn.execute("INSERT INTO fts_reviews(fts_reviews) VALUES('rebuild');")
        conn.execute("INSERT INTO fts_artifacts(fts_artifacts) VALUES('rebuild');")

    # --- Projects ---
    def insert_project(
        self,
        *,
        project_id: str,
        title: str,
        domain: str,
        stage: str,
        repo_path: str,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        now = utc_now_iso()
        description = ""
        if isinstance(meta, dict):
            description = str(meta.get("description") or "")
        self._exec(
            """
            INSERT INTO projects(id, title, domain, description, stage, repo_path, meta_json, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (project_id, title, domain, description, stage, repo_path, json.dumps(meta, ensure_ascii=False), now, now),
        )
        self._maybe_commit()
        return self.get_project(project_id)

    def list_projects(self) -> List[Dict[str, Any]]:
        rows = self._exec("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return rows

    def get_project(self, project_id: str) -> Dict[str, Any]:
        row = self._exec("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            raise SystemExit(f"Project not found: {project_id}")
        return row

    def update_project_stage(self, project_id: str, stage: str) -> Dict[str, Any]:
        now = utc_now_iso()
        self._exec("UPDATE projects SET stage = ?, updated_at = ? WHERE id = ?", (stage, now, project_id))
        self._maybe_commit()
        return self.get_project(project_id)

    # --- Tasks ---
    def insert_task(
        self,
        *,
        task_id: str,
        project_id: str,
        task_type: str,
        status: str,
        spec: Dict[str, Any],
        deps: List[str],
    ) -> Dict[str, Any]:
        now = utc_now_iso()
        self._exec(
            """
            INSERT INTO tasks(id, project_id, type, status, spec_json, deps_json, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                project_id,
                task_type,
                status,
                json.dumps(spec, ensure_ascii=False),
                json.dumps(deps, ensure_ascii=False),
                now,
                now,
            ),
        )
        self._maybe_commit()
        return self.get_task(task_id)

    def list_tasks(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if project_id:
            rows = self._exec(
                "SELECT * FROM tasks WHERE project_id = ? ORDER BY updated_at DESC",
                (project_id,),
            ).fetchall()
        else:
            rows = self._exec("SELECT * FROM tasks ORDER BY updated_at DESC").fetchall()
        return rows

    def get_task(self, task_id: str) -> Dict[str, Any]:
        row = self._exec("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise SystemExit(f"Task not found: {task_id}")
        return row

    def update_task_status(self, task_id: str, status: str) -> Dict[str, Any]:
        now = utc_now_iso()
        self._exec("UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?", (status, now, task_id))
        self._maybe_commit()
        return self.get_task(task_id)

    # --- Task runs + events ---
    def insert_task_run(
        self,
        *,
        run_id: str,
        task_id: str,
        status: str,
        jsonl_path: Optional[str],
        last_message_path: Optional[str],
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        started_at = utc_now_iso()
        self._exec(
            """
            INSERT INTO task_runs(id, task_id, started_at, status, jsonl_path, last_message_path, meta_json)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                task_id,
                started_at,
                status,
                jsonl_path,
                last_message_path,
                json.dumps(meta, ensure_ascii=False),
            ),
        )
        self._maybe_commit()
        return self.get_task_run(run_id)

    def finish_task_run(
        self, *, run_id: str, status: str, exit_code: int, meta_updates: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        finished_at = utc_now_iso()
        cur = self._exec("SELECT meta_json FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        if cur is None:
            raise SystemExit(f"Task run not found: {run_id}")
        meta = json.loads(cur["meta_json"]) if cur["meta_json"] else {}
        if meta_updates:
            meta.update(meta_updates)
        self._exec(
            """
            UPDATE task_runs
              SET finished_at = ?, status = ?, exit_code = ?, meta_json = ?
              WHERE id = ?
            """,
            (finished_at, status, exit_code, json.dumps(meta, ensure_ascii=False), run_id),
        )
        self._maybe_commit()
        return self.get_task_run(run_id)

    def get_task_run(self, run_id: str) -> Dict[str, Any]:
        row = self._exec("SELECT * FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise SystemExit(f"Task run not found: {run_id}")
        return row

    def insert_task_event(self, *, task_run_id: str, event_type: str, data: Dict[str, Any]) -> None:
        self._exec(
            "INSERT INTO task_events(task_run_id, ts, event_type, data_json) VALUES(?, ?, ?, ?)",
            (task_run_id, utc_now_iso(), event_type, json.dumps(data, ensure_ascii=False)),
        )
        self._maybe_commit()

    # --- Artifacts ---
    def insert_artifact(
        self,
        *,
        artifact_id: str,
        project_id: str,
        kind: str,
        path: str,
        sha256: Optional[str],
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        now = utc_now_iso()
        self._exec(
            """
            INSERT INTO artifacts(id, project_id, kind, path, sha256, meta_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (artifact_id, project_id, kind, path, sha256, json.dumps(meta, ensure_ascii=False), now),
        )
        self._maybe_commit()
        return self.get_artifact(artifact_id)

    def get_artifact(self, artifact_id: str) -> Dict[str, Any]:
        row = self._exec("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise SystemExit(f"Artifact not found: {artifact_id}")
        return row

    def list_artifacts(self, project_id: str, prefix: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        if limit < 1:
            limit = 1
        if limit > 500:
            limit = 500
        if prefix:
            return self._exec(
                "SELECT * FROM artifacts WHERE project_id = ? AND path LIKE ? ORDER BY created_at DESC LIMIT ?",
                (project_id, f"{prefix}%", limit),
            ).fetchall()
        return self._exec(
            "SELECT * FROM artifacts WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
            (project_id, limit),
        ).fetchall()

    # --- Reviews ---
    def insert_review(
        self,
        *,
        review_id: str,
        project_id: str,
        stage: str,
        reviewer: str,
        rubric: Dict[str, Any],
        findings: Dict[str, Any],
    ) -> Dict[str, Any]:
        now = utc_now_iso()
        self._exec(
            """
            INSERT INTO reviews(id, project_id, stage, reviewer, rubric_json, findings_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review_id,
                project_id,
                stage,
                reviewer,
                json.dumps(rubric, ensure_ascii=False),
                json.dumps(findings, ensure_ascii=False),
                now,
            ),
        )
        self._maybe_commit()
        return self.get_review(review_id)

    def get_review(self, review_id: str) -> Dict[str, Any]:
        row = self._exec("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        if row is None:
            raise SystemExit(f"Review not found: {review_id}")
        return row

    def list_reviews(self, project_id: str) -> List[Dict[str, Any]]:
        return self._exec("SELECT * FROM reviews WHERE project_id = ? ORDER BY created_at DESC", (project_id,)).fetchall()

    # --- Jobs ---
    def insert_job(
        self,
        *,
        job_id: str,
        project_id: Optional[str],
        provider: str,
        kind: str,
        status: str,
        spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        now = utc_now_iso()
        self._exec(
            """
            INSERT INTO jobs(id, project_id, provider, kind, status, spec_json, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, project_id, provider, kind, status, json.dumps(spec, ensure_ascii=False), now, now),
        )
        self._maybe_commit()
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        row = self._exec("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise SystemExit(f"Job not found: {job_id}")
        return row

    def find_jobs_by_remote_id(self, *, remote_id: str, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        if provider:
            return self._exec(
                "SELECT * FROM jobs WHERE provider = ? AND remote_id = ? ORDER BY updated_at DESC",
                (provider, remote_id),
            ).fetchall()
        return self._exec(
            "SELECT * FROM jobs WHERE remote_id = ? ORDER BY updated_at DESC",
            (remote_id,),
        ).fetchall()

    def list_jobs(self, project_id: Optional[str] = None, status: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        if limit < 1:
            limit = 1
        if limit > 500:
            limit = 500
        params: List[Any] = []
        sql = "SELECT * FROM jobs"
        clauses: List[str] = []
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        return self._exec(sql, params).fetchall()

    def update_job(
        self,
        *,
        job_id: str,
        status: Optional[str] = None,
        remote_id: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        started: bool = False,
        finished: bool = False,
        commit: bool = True,
    ) -> Dict[str, Any]:
        now = utc_now_iso()
        cur = self.get_job(job_id)
        fields: List[str] = ["updated_at = ?"]
        params: List[Any] = [now]
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if remote_id is not None:
            fields.append("remote_id = ?")
            params.append(remote_id)
        if result is not None:
            fields.append("result_json = ?")
            params.append(json.dumps(result, ensure_ascii=False))
        if error is not None:
            fields.append("error = ?")
            params.append(error)
        if started and not cur.get("started_at"):
            fields.append("started_at = ?")
            params.append(now)
        if finished:
            fields.append("finished_at = ?")
            params.append(now)
        params.append(job_id)
        self._exec(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", params)
        if commit:
            self._maybe_commit()
        return self.get_job(job_id)

    def insert_job_event(self, *, job_id: str, event_type: str, data: Dict[str, Any]) -> None:
        self._exec(
            "INSERT INTO job_events(job_id, ts, event_type, data_json) VALUES(?, ?, ?, ?)",
            (job_id, utc_now_iso(), event_type, json.dumps(data, ensure_ascii=False)),
        )
        self._maybe_commit()

    # --- Topic engine: ideas ---
    def upsert_idea(
        self,
        *,
        idea_id: str,
        project_id: str,
        status: str,
        score_total: Optional[float],
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        now = utc_now_iso()
        idea_title = str(data.get("title") or "") if isinstance(data, dict) else ""
        idea_abstract = str(data.get("abstract") or "") if isinstance(data, dict) else ""
        self._exec(
            """
            INSERT INTO ideas(id, project_id, status, score_total, data_json, title, abstract, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              project_id=excluded.project_id,
              status=excluded.status,
              score_total=excluded.score_total,
              data_json=excluded.data_json,
              title=excluded.title,
              abstract=excluded.abstract,
              updated_at=excluded.updated_at
            """,
            (
                idea_id,
                project_id,
                status,
                score_total,
                json.dumps(data, ensure_ascii=False),
                idea_title,
                idea_abstract,
                now,
                now,
            ),
        )
        self._maybe_commit()
        return self.get_idea(idea_id)

    def get_idea(self, idea_id: str) -> Dict[str, Any]:
        row = self._exec("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
        if row is None:
            raise SystemExit(f"Idea not found: {idea_id}")
        return row

    def list_ideas(
        self,
        *,
        project_id: str,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        if limit < 1:
            limit = 1
        if limit > 500:
            limit = 500
        if status:
            return self._exec(
                """
                SELECT * FROM ideas
                WHERE project_id = ? AND status = ?
                ORDER BY (score_total IS NULL) ASC, score_total DESC, updated_at DESC
                LIMIT ?
                """,
                (project_id, status, limit),
            ).fetchall()
        return self._exec(
            """
            SELECT * FROM ideas
            WHERE project_id = ?
            ORDER BY (score_total IS NULL) ASC, score_total DESC, updated_at DESC
            LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()

    def insert_idea_edge(
        self,
        *,
        src_idea_id: str,
        dst_idea_id: str,
        relation: str,
        reason: Optional[str],
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        now = utc_now_iso()
        cur = self._exec(
            """
            INSERT INTO idea_edges(src_idea_id, dst_idea_id, relation, reason, meta_json, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                src_idea_id,
                dst_idea_id,
                relation,
                reason,
                json.dumps(meta, ensure_ascii=False),
                now,
                now,
            ),
        )
        self._maybe_commit()
        row = self._exec("SELECT * FROM idea_edges WHERE id = ?", (cur.lastrowid,)).fetchone()
        if row is None:
            raise SystemExit("Failed to insert idea edge row.")
        return row

    def list_idea_edges(
        self,
        *,
        idea_ids: Optional[List[str]] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        if limit < 1:
            limit = 1
        if limit > 2000:
            limit = 2000

        if idea_ids:
            ids = [str(x) for x in idea_ids if x]
            if not ids:
                return []
            placeholders = ", ".join(["?"] * len(ids))
            sql = (
                "SELECT * FROM idea_edges "
                f"WHERE src_idea_id IN ({placeholders}) OR dst_idea_id IN ({placeholders}) "
                "ORDER BY updated_at DESC LIMIT ?"
            )
            params: List[Any] = list(ids) + list(ids) + [limit]
            return self._exec(sql, params).fetchall()

        return self._exec("SELECT * FROM idea_edges ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()

    # --- Topic engine: smoke tests ---
    def insert_smoke_test(
        self,
        *,
        idea_id: str,
        project_id: str,
        verdict: str,
        started_at: str,
        completed_at: Optional[str],
        result: Dict[str, Any],
        artifact_path: Optional[str],
    ) -> Dict[str, Any]:
        now = utc_now_iso()
        cur = self._exec(
            """
            INSERT INTO smoke_tests(
              idea_id, project_id, verdict, started_at, completed_at, result_json, artifact_path, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                idea_id,
                project_id,
                verdict,
                started_at,
                completed_at,
                json.dumps(result, ensure_ascii=False),
                artifact_path,
                now,
                now,
            ),
        )
        self._maybe_commit()
        row = self._exec("SELECT * FROM smoke_tests WHERE id = ?", (cur.lastrowid,)).fetchone()
        if row is None:
            raise SystemExit("Failed to insert smoke test row.")
        return row

    def list_smoke_tests(
        self,
        *,
        project_id: str,
        idea_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        if limit < 1:
            limit = 1
        if limit > 500:
            limit = 500
        if idea_id:
            return self._exec(
                """
                SELECT * FROM smoke_tests
                WHERE project_id = ? AND idea_id = ?
                ORDER BY started_at DESC, id DESC
                LIMIT ?
                """,
                (project_id, idea_id, limit),
            ).fetchall()
        return self._exec(
            """
            SELECT * FROM smoke_tests
            WHERE project_id = ?
            ORDER BY started_at DESC, id DESC
            LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()

    def get_smoke_test(self, smoke_test_id: int) -> Dict[str, Any]:
        row = self._exec("SELECT * FROM smoke_tests WHERE id = ?", (int(smoke_test_id),)).fetchone()
        if row is None:
            raise SystemExit(f"Smoke test not found: {smoke_test_id}")
        return row

    # --- Evidence ---
    def insert_evidence(
        self,
        *,
        evidence_id: str,
        project_id: str,
        idea_id: Optional[str],
        kind: str,
        title: str,
        url: str,
        retrieved_at: str,
        summary: str,
        relevance: Optional[float],
        meta: Dict[str, Any],
        artifact_path: Optional[str],
    ) -> Dict[str, Any]:
        now = utc_now_iso()
        self._exec(
            """
            INSERT INTO evidence(
              id, project_id, idea_id, kind, title, url, retrieved_at, summary, relevance, meta_json, artifact_path, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence_id,
                project_id,
                idea_id,
                kind,
                title,
                url,
                retrieved_at,
                summary,
                relevance,
                json.dumps(meta, ensure_ascii=False),
                artifact_path,
                now,
                now,
            ),
        )
        self._maybe_commit()
        return self.get_evidence(evidence_id)

    def get_evidence(self, evidence_id: str) -> Dict[str, Any]:
        row = self._exec("SELECT * FROM evidence WHERE id = ?", (evidence_id,)).fetchone()
        if row is None:
            raise SystemExit(f"Evidence not found: {evidence_id}")
        return row

    def list_evidence(
        self,
        *,
        project_id: str,
        idea_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        if limit < 1:
            limit = 1
        if limit > 500:
            limit = 500
        if idea_id:
            return self._exec(
                """
                SELECT * FROM evidence
                WHERE project_id = ? AND idea_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (project_id, idea_id, limit),
            ).fetchall()
        return self._exec(
            """
            SELECT * FROM evidence
            WHERE project_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()

    # --- Playbook ---
    def upsert_playbook_entry(self, *, entry_id: str, topic: str, rule: Dict[str, Any]) -> Dict[str, Any]:
        now = utc_now_iso()
        self._exec(
            """
            INSERT INTO playbook(id, topic, rule_json, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              topic=excluded.topic,
              rule_json=excluded.rule_json,
              updated_at=excluded.updated_at
            """,
            (
                entry_id,
                topic,
                json.dumps(rule, ensure_ascii=False),
                now,
                now,
            ),
        )
        self._maybe_commit()
        return self.get_playbook_entry(entry_id)

    def get_playbook_entry(self, entry_id: str) -> Dict[str, Any]:
        row = self._exec("SELECT * FROM playbook WHERE id = ?", (entry_id,)).fetchone()
        if row is None:
            raise SystemExit(f"Playbook entry not found: {entry_id}")
        return row

    def list_playbook_entries(self, *, topic: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        if limit < 1:
            limit = 1
        if limit > 500:
            limit = 500
        if topic:
            # Escape SQL LIKE wildcards so user input behaves as a substring search.
            topic_esc = str(topic).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            return self._exec(
                "SELECT * FROM playbook WHERE topic LIKE ? ESCAPE '\\' ORDER BY updated_at DESC LIMIT ?",
                (f"%{topic_esc}%", limit),
            ).fetchall()
        return self._exec("SELECT * FROM playbook ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
