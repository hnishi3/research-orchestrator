from __future__ import annotations

import json
import mimetypes
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from resorch.ledger import Ledger


@dataclass(frozen=True)
class Hit:
    id: str
    title: str
    snippet: str
    metadata: Dict[str, Any]


def _norm(s: str) -> str:
    return s.strip().lower()


def _like(query: str) -> str:
    return f"%{query}%"


def _make_snippet(text: str, query: str, max_len: int = 240) -> str:
    if not text:
        return ""
    q = _norm(query)
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    idx = t.lower().find(q) if q else -1
    if idx < 0:
        return (t[: max_len - 1] + "…") if len(t) > max_len else t
    start = max(0, idx - max_len // 4)
    end = min(len(t), start + max_len)
    snippet = t[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(t):
        snippet = snippet + "…"
    return snippet


def _safe_read_text(path: Path, max_bytes: int = 200_000) -> Optional[str]:
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes + 1)
    except OSError:
        return None

    if b"\x00" in data:
        return None  # likely binary
    if len(data) > max_bytes:
        data = data[:max_bytes]
    try:
        return data.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return None


def _guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".md", ".markdown"}:
        return "text/markdown"
    if ext in {".json"}:
        return "application/json"
    if ext in {".yaml", ".yml"}:
        return "text/yaml"
    ctype, _ = mimetypes.guess_type(str(path))
    return ctype or "text/plain"


_FTS_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _build_fts_match_query(query: str, *, max_terms: int = 8, max_term_len: int = 48) -> Optional[str]:
    tokens = _FTS_TOKEN_RE.findall(query.strip())
    if not tokens:
        return None

    out: List[str] = []
    for tok in tokens[:max_terms]:
        clipped = tok[:max_term_len].replace('"', '""')
        if clipped:
            out.append(f'"{clipped}"')
    if not out:
        return None
    return " AND ".join(out)


def _fts_available(conn: sqlite3.Connection, match_query: str) -> bool:
    try:
        conn.execute("SELECT rowid FROM fts_projects WHERE fts_projects MATCH ? LIMIT 1", (match_query,)).fetchall()
        return True
    except sqlite3.OperationalError:
        return False


def search(
    ledger: Ledger,
    *,
    query: str,
    project_id: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    q = query.strip()
    if not q:
        return {"hits": []}

    if limit < 1:
        limit = 1
    if limit > 50:
        limit = 50

    hits: List[Hit] = []
    seen_ids: set[str] = set()
    q_like = _like(q)
    conn = ledger.conn()
    fts_match = _build_fts_match_query(q)
    use_fts = bool(fts_match) and _fts_available(conn, fts_match or "")

    include_ledger = kind in (None, "", "ledger")
    include_artifact = kind in (None, "", "artifact")

    def _append(hit: Hit) -> None:
        if len(hits) >= limit:
            return
        if hit.id in seen_ids:
            return
        seen_ids.add(hit.id)
        hits.append(hit)

    # --- Ledger: projects ---
    if include_ledger and len(hits) < limit:
        rows: List[Dict[str, Any]] = []
        if use_fts and fts_match:
            params = [fts_match]
            sql = """
              SELECT p.id, p.title, p.domain, p.description, p.stage, p.repo_path, p.updated_at,
                     snippet(fts_projects, -1, '[', ']', ' … ', 20) AS fts_snippet,
                     bm25(fts_projects) AS rank
              FROM fts_projects
              JOIN projects p ON p.rowid = fts_projects.rowid
              WHERE fts_projects MATCH ?
            """
            if project_id:
                sql += " AND p.id = ?"
                params.append(project_id)
            sql += " ORDER BY rank, p.updated_at DESC LIMIT ?"
            params.append(limit - len(hits))
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                use_fts = False

        if not use_fts:
            params = [q_like, q_like, q_like, q_like]
            sql = """
              SELECT id, title, domain, description, stage, repo_path, updated_at
              FROM projects
              WHERE (title LIKE ? OR domain LIKE ? OR description LIKE ? OR stage LIKE ?)
            """
            if project_id:
                sql += " AND id = ?"
                params.append(project_id)
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit - len(hits))
            rows = conn.execute(sql, params).fetchall()

        for r in rows:
            snippet = str(r.get("fts_snippet") or "")
            if not snippet:
                snippet = _make_snippet(
                    f"{r.get('title') or ''} {r.get('domain') or ''} {r.get('description') or ''} stage={r.get('stage') or ''}",
                    q,
                )
            _append(
                Hit(
                    id=f"ledger:projects/{r['id']}",
                    title=f"project:{r['id']} {r['title']}",
                    snippet=snippet,
                    metadata={"type": "project", "project_id": r["id"], "stage": r["stage"], "repo_path": r["repo_path"]},
                )
            )

    # --- Ledger: tasks ---
    if include_ledger and len(hits) < limit:
        rows = []
        if use_fts and fts_match:
            params = [fts_match]
            sql = """
              SELECT t.id, t.project_id, t.type, t.status, t.spec_json, t.updated_at,
                     snippet(fts_tasks, -1, '[', ']', ' … ', 20) AS fts_snippet,
                     bm25(fts_tasks) AS rank
              FROM fts_tasks
              JOIN tasks t ON t.rowid = fts_tasks.rowid
              WHERE fts_tasks MATCH ?
            """
            if project_id:
                sql += " AND t.project_id = ?"
                params.append(project_id)
            sql += " ORDER BY rank, t.updated_at DESC LIMIT ?"
            params.append(limit - len(hits))
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                use_fts = False

        if not use_fts:
            params = [q_like, q_like, q_like]
            sql = """
              SELECT id, project_id, type, status, spec_json, updated_at
              FROM tasks
              WHERE (type LIKE ? OR status LIKE ? OR spec_json LIKE ?)
            """
            if project_id:
                sql += " AND project_id = ?"
                params.append(project_id)
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit - len(hits))
            rows = conn.execute(sql, params).fetchall()

        for r in rows:
            spec = r.get("spec_json") or ""
            snippet = str(r.get("fts_snippet") or "")
            if not snippet:
                snippet = _make_snippet(spec, q)
            _append(
                Hit(
                    id=f"ledger:tasks/{r['id']}",
                    title=f"task:{r['id']} ({r['type']}/{r['status']})",
                    snippet=snippet,
                    metadata={"type": "task", "task_id": r["id"], "project_id": r["project_id"], "status": r["status"], "task_type": r["type"]},
                )
            )

    # --- Ledger: ideas ---
    if include_ledger and len(hits) < limit:
        rows = []
        if use_fts and fts_match:
            params = [fts_match]
            sql = """
              SELECT i.id, i.project_id, i.status, i.score_total, i.data_json, i.title, i.updated_at,
                     snippet(fts_ideas, -1, '[', ']', ' … ', 20) AS fts_snippet,
                     bm25(fts_ideas) AS rank
              FROM fts_ideas
              JOIN ideas i ON i.rowid = fts_ideas.rowid
              WHERE fts_ideas MATCH ?
            """
            if project_id:
                sql += " AND i.project_id = ?"
                params.append(project_id)
            sql += " ORDER BY rank, i.updated_at DESC LIMIT ?"
            params.append(limit - len(hits))
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                use_fts = False

        if not use_fts:
            params = [q_like, q_like, q_like]
            sql = """
              SELECT id, project_id, status, score_total, data_json, title, updated_at
              FROM ideas
              WHERE (id LIKE ? OR status LIKE ? OR data_json LIKE ?)
            """
            if project_id:
                sql += " AND project_id = ?"
                params.append(project_id)
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit - len(hits))
            rows = conn.execute(sql, params).fetchall()

        for r in rows:
            data_json = str(r.get("data_json") or "")
            title = f"idea:{r['id']}"
            idea_title = str(r.get("title") or "").strip()
            if idea_title:
                title = f"idea:{r['id']} {idea_title}"
            else:
                try:
                    data = json.loads(data_json) if data_json else {}
                    if isinstance(data, dict) and data.get("title"):
                        title = f"idea:{r['id']} {data.get('title')}"
                except json.JSONDecodeError:
                    pass

            snippet = str(r.get("fts_snippet") or "")
            if not snippet:
                snippet = _make_snippet(data_json, q)

            _append(
                Hit(
                    id=f"ledger:ideas/{r['id']}",
                    title=title,
                    snippet=snippet,
                    metadata={
                        "type": "idea",
                        "idea_id": r["id"],
                        "project_id": r["project_id"],
                        "status": r["status"],
                        "score_total": r["score_total"],
                    },
                )
            )

    # --- Ledger: smoke tests ---
    if include_ledger and len(hits) < limit:
        params = [q_like, q_like, q_like]
        sql = """
          SELECT id, project_id, idea_id, verdict, started_at, result_json
          FROM smoke_tests
          WHERE (idea_id LIKE ? OR verdict LIKE ? OR result_json LIKE ?)
        """
        if project_id:
            sql += " AND project_id = ?"
            params.append(project_id)
        sql += " ORDER BY started_at DESC, id DESC LIMIT ?"
        params.append(limit - len(hits))
        rows = conn.execute(sql, params).fetchall()
        for r in rows:
            result_json = str(r.get("result_json") or "")
            title = f"smoke_test:{r['id']} idea={r.get('idea_id')} verdict={r.get('verdict')}"
            snippet = _make_snippet(result_json, q) if result_json else f"idea={r.get('idea_id')} verdict={r.get('verdict')}"
            _append(
                Hit(
                    id=f"ledger:smoke_tests/{r['id']}",
                    title=title,
                    snippet=snippet,
                    metadata={
                        "type": "smoke_test",
                        "smoke_test_id": r["id"],
                        "project_id": r["project_id"],
                        "idea_id": r["idea_id"],
                        "verdict": r["verdict"],
                    },
                )
            )

    # --- Ledger: evidence ---
    if include_ledger and len(hits) < limit:
        rows = []
        if use_fts and fts_match:
            params = [fts_match]
            sql = """
              SELECT e.id, e.project_id, e.idea_id, e.kind, e.title, e.url, e.summary, e.updated_at,
                     snippet(fts_evidence, -1, '[', ']', ' … ', 20) AS fts_snippet,
                     bm25(fts_evidence) AS rank
              FROM fts_evidence
              JOIN evidence e ON e.rowid = fts_evidence.rowid
              WHERE fts_evidence MATCH ?
            """
            if project_id:
                sql += " AND e.project_id = ?"
                params.append(project_id)
            sql += " ORDER BY rank, e.updated_at DESC LIMIT ?"
            params.append(limit - len(hits))
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                use_fts = False

        if not use_fts:
            params = [q_like, q_like, q_like, q_like]
            sql = """
              SELECT id, project_id, idea_id, kind, title, url, summary, updated_at
              FROM evidence
              WHERE (kind LIKE ? OR title LIKE ? OR url LIKE ? OR summary LIKE ?)
            """
            if project_id:
                sql += " AND project_id = ?"
                params.append(project_id)
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit - len(hits))
            rows = conn.execute(sql, params).fetchall()

        for r in rows:
            snippet = str(r.get("fts_snippet") or "")
            if not snippet:
                snippet = _make_snippet(f"{r.get('title')}\n{r.get('summary')}\n{r.get('url')}", q)
            _append(
                Hit(
                    id=f"ledger:evidence/{r['id']}",
                    title=f"evidence:{r['id']} {r['title']}",
                    snippet=snippet,
                    metadata={
                        "type": "evidence",
                        "evidence_id": r["id"],
                        "project_id": r["project_id"],
                        "idea_id": r["idea_id"],
                        "kind": r["kind"],
                        "url": r["url"],
                    },
                )
            )

    # --- Ledger: reviews ---
    if include_ledger and len(hits) < limit:
        rows = []
        if use_fts and fts_match:
            params = [fts_match]
            sql = """
              SELECT r.id, r.project_id, r.stage, r.reviewer, r.findings_json, r.created_at,
                     snippet(fts_reviews, -1, '[', ']', ' … ', 20) AS fts_snippet,
                     bm25(fts_reviews) AS rank
              FROM fts_reviews
              JOIN reviews r ON r.rowid = fts_reviews.rowid
              WHERE fts_reviews MATCH ?
            """
            if project_id:
                sql += " AND r.project_id = ?"
                params.append(project_id)
            sql += " ORDER BY rank, r.created_at DESC LIMIT ?"
            params.append(limit - len(hits))
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                use_fts = False

        if not use_fts:
            params = [q_like, q_like, q_like, q_like]
            sql = """
              SELECT id, project_id, stage, reviewer, findings_json, created_at
              FROM reviews
              WHERE (stage LIKE ? OR reviewer LIKE ? OR rubric_json LIKE ? OR findings_json LIKE ?)
            """
            if project_id:
                sql += " AND project_id = ?"
                params.append(project_id)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit - len(hits))
            rows = conn.execute(sql, params).fetchall()

        for r in rows:
            findings = r.get("findings_json") or ""
            snippet = str(r.get("fts_snippet") or "")
            if not snippet:
                snippet = _make_snippet(findings, q)
            _append(
                Hit(
                    id=f"ledger:reviews/{r['id']}",
                    title=f"review:{r['id']} ({r['stage']}/{r['reviewer']})",
                    snippet=snippet,
                    metadata={"type": "review", "review_id": r["id"], "project_id": r["project_id"], "stage": r["stage"], "reviewer": r["reviewer"]},
                )
            )

    # --- Artifact index + optional content match ---
    if include_artifact and len(hits) < limit:
        rows = []
        via_fts = False
        if use_fts and fts_match:
            params = [fts_match]
            sql = """
              SELECT a.id, a.project_id, a.kind, a.path, a.meta_json, p.repo_path,
                     snippet(fts_artifacts, -1, '[', ']', ' … ', 20) AS fts_snippet,
                     bm25(fts_artifacts) AS rank
              FROM fts_artifacts
              JOIN artifacts a ON a.rowid = fts_artifacts.rowid
              JOIN projects p ON p.id = a.project_id
              WHERE fts_artifacts MATCH ?
            """
            if project_id:
                sql += " AND a.project_id = ?"
                params.append(project_id)
            sql += " ORDER BY rank, a.created_at DESC LIMIT ?"
            params.append(limit - len(hits))
            try:
                rows = conn.execute(sql, params).fetchall()
                via_fts = bool(rows)
            except sqlite3.OperationalError:
                use_fts = False

        if via_fts:
            for r in rows:
                rel_path = Path(str(r["path"]))
                snippet = str(r.get("fts_snippet") or "")
                if not snippet:
                    snippet = f"{rel_path.as_posix()} ({r['kind']})"
                _append(
                    Hit(
                        id=f"artifact:{r['project_id']}/{rel_path.as_posix()}",
                        title=f"artifact:{rel_path.as_posix()}",
                        snippet=snippet,
                        metadata={"type": "artifact", "project_id": r["project_id"], "path": rel_path.as_posix(), "kind": r["kind"]},
                    )
                )
        else:
            params = [q_like, q_like]
            sql = """
              SELECT a.id, a.project_id, a.kind, a.path, a.meta_json, p.repo_path
              FROM artifacts a
              JOIN projects p ON p.id = a.project_id
              WHERE (a.path LIKE ? OR a.meta_json LIKE ?)
            """
            if project_id:
                sql += " AND a.project_id = ?"
                params.append(project_id)
            sql += " ORDER BY a.created_at DESC LIMIT ?"
            params.append(limit - len(hits))
            rows = conn.execute(sql, params).fetchall()
            fallback_mode = False

            # If no matches by path/meta, fall back to a small content scan across recent artifacts.
            if not rows:
                fallback_mode = True
                params2: List[Any] = []
                sql2 = """
                  SELECT a.id, a.project_id, a.kind, a.path, a.meta_json, p.repo_path
                  FROM artifacts a
                  JOIN projects p ON p.id = a.project_id
                """
                if project_id:
                    sql2 += " WHERE a.project_id = ?"
                    params2.append(project_id)
                sql2 += " ORDER BY a.created_at DESC LIMIT ?"
                params2.append(max(limit * 10, 50))
                rows = conn.execute(sql2, params2).fetchall()

            for r in rows:
                if len(hits) >= limit:
                    break
                rel_path = Path(str(r["path"]))
                ws_root = Path(str(r["repo_path"]))
                abs_path = (ws_root / rel_path).resolve()
                snippet = f"{rel_path.as_posix()} ({r['kind']})"

                q_l = q.lower()
                meta_json = str(r.get("meta_json") or "")

                content = _safe_read_text(abs_path)
                content_match = content is not None and q_l in content.lower()
                meta_match = q_l in meta_json.lower()
                path_match = q_l in rel_path.as_posix().lower()

                if content_match:
                    snippet = _make_snippet(content, q)
                elif fallback_mode and not (meta_match or path_match):
                    continue

                _append(
                    Hit(
                        id=f"artifact:{r['project_id']}/{rel_path.as_posix()}",
                        title=f"artifact:{rel_path.as_posix()}",
                        snippet=snippet,
                        metadata={"type": "artifact", "project_id": r["project_id"], "path": rel_path.as_posix(), "kind": r["kind"]},
                    )
                )

    return {"hits": [h.__dict__ for h in hits[:limit]]}


def fetch(ledger: Ledger, *, id: str) -> Dict[str, Any]:
    if id.startswith("ledger:projects/"):
        pid = id.split("/", 1)[1]
        row = ledger.get_project(pid)
        meta = json.loads(row.pop("meta_json") or "{}")
        row["meta"] = meta
        return {
            "id": id,
            "content": json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            "content_type": "application/json",
            "metadata": {"type": "project", "project_id": pid},
        }

    if id.startswith("ledger:tasks/"):
        tid = id.split("/", 1)[1]
        row = ledger.get_task(tid)
        row["spec"] = json.loads(row.pop("spec_json") or "{}")
        row["deps"] = json.loads(row.pop("deps_json") or "[]")
        return {
            "id": id,
            "content": json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            "content_type": "application/json",
            "metadata": {"type": "task", "task_id": tid, "project_id": row["project_id"]},
        }

    if id.startswith("ledger:ideas/"):
        iid = id.split("/", 1)[1]
        row = ledger.get_idea(iid)
        row["data"] = json.loads(row.pop("data_json") or "{}")
        return {
            "id": id,
            "content": json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            "content_type": "application/json",
            "metadata": {"type": "idea", "idea_id": iid, "project_id": row["project_id"], "status": row["status"]},
        }

    if id.startswith("ledger:smoke_tests/"):
        sid = id.split("/", 1)[1]
        row = ledger.get_smoke_test(int(sid))
        row["result"] = json.loads(row.pop("result_json") or "{}")
        return {
            "id": id,
            "content": json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            "content_type": "application/json",
            "metadata": {
                "type": "smoke_test",
                "smoke_test_id": int(sid),
                "project_id": row["project_id"],
                "idea_id": row["idea_id"],
                "verdict": row["verdict"],
            },
        }

    if id.startswith("ledger:evidence/"):
        eid = id.split("/", 1)[1]
        row = ledger.get_evidence(eid)
        row["meta"] = json.loads(row.pop("meta_json") or "{}")
        return {
            "id": id,
            "content": json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            "content_type": "application/json",
            "metadata": {
                "type": "evidence",
                "evidence_id": eid,
                "project_id": row["project_id"],
                "idea_id": row["idea_id"],
                "kind": row["kind"],
                "url": row["url"],
            },
        }

    if id.startswith("ledger:reviews/"):
        rid = id.split("/", 1)[1]
        row = ledger.get_review(rid)
        row["rubric"] = json.loads(row.pop("rubric_json") or "{}")
        row["findings"] = json.loads(row.pop("findings_json") or "{}")
        return {
            "id": id,
            "content": json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            "content_type": "application/json",
            "metadata": {"type": "review", "review_id": rid, "project_id": row["project_id"]},
        }

    if id.startswith("artifact:"):
        rest = id[len("artifact:") :]
        if "/" not in rest:
            raise SystemExit(f"Invalid artifact id (expected artifact:<project_id>/<path>): {id}")
        project_id, rel = rest.split("/", 1)
        project = ledger.get_project(project_id)
        repo_path = Path(project["repo_path"])
        rel_path = Path(rel)
        abs_path = (repo_path / rel_path).resolve()
        try:
            abs_path.relative_to(repo_path.resolve())
        except ValueError:
            raise SystemExit(f"Path traversal blocked: artifact path escapes workspace: {id}")
        content = _safe_read_text(abs_path)
        if content is None:
            raise SystemExit(f"Artifact not readable as text: {abs_path}")
        return {
            "id": id,
            "content": content,
            "content_type": _guess_content_type(abs_path),
            "metadata": {"type": "artifact", "project_id": project_id, "path": rel_path.as_posix()},
        }

    raise SystemExit(f"Unknown id scheme: {id}")
