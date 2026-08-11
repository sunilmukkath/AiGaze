"""AI Gaze project library — projects → folders → creatives → analysis runs.

DataWiz / TScribe-shaped. SQLite + files under AIGAZE_DATA_DIR.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DATA_DIR = Path(os.environ.get("AIGAZE_DATA_DIR", Path(__file__).resolve().parent / ".data"))
_DB_PATH = _DATA_DIR / "users.sqlite"  # same DB as auth_users
_FILES_DIR = _DATA_DIR / "library"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _connect() -> sqlite3.Connection:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _FILES_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_schema(conn)
    conn.commit()
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            client_name TEXT,
            owner_email TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS folders (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            parent_id TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS creatives (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            folder_id TEXT,
            name TEXT NOT NULL,
            file_name TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            width INTEGER NOT NULL DEFAULT 0,
            height INTEGER NOT NULL DEFAULT 0,
            storage_path TEXT NOT NULL,
            thumb_path TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS analysis_runs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            creative_id TEXT NOT NULL,
            engine TEXT NOT NULL,
            confidence REAL,
            label TEXT,
            meta_json TEXT NOT NULL,
            overlay_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
            FOREIGN KEY (creative_id) REFERENCES creatives(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_email);
        CREATE INDEX IF NOT EXISTS idx_folders_project ON folders(project_id);
        CREATE INDEX IF NOT EXISTS idx_creatives_project ON creatives(project_id);
        CREATE INDEX IF NOT EXISTS idx_runs_creative ON analysis_runs(creative_id);
        """
    )


def _row_project(r: sqlite3.Row, folders: list[dict] | None = None) -> dict:
    return {
        "id": r["id"],
        "name": r["name"],
        "clientName": r["client_name"],
        "ownerEmail": r["owner_email"],
        "createdAt": r["created_at"],
        "updatedAt": r["updated_at"],
        "folders": folders if folders is not None else [],
    }


def _row_folder(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "projectId": r["project_id"],
        "name": r["name"],
        "parentId": r["parent_id"],
        "createdAt": r["created_at"],
    }


def _row_creative(r: sqlite3.Row, latest_run_id: str | None = None) -> dict:
    return {
        "id": r["id"],
        "projectId": r["project_id"],
        "folderId": r["folder_id"],
        "name": r["name"],
        "fileName": r["file_name"],
        "mimeType": r["mime_type"],
        "width": int(r["width"] or 0),
        "height": int(r["height"] or 0),
        "createdAt": r["created_at"],
        "updatedAt": r["updated_at"],
        "latestRunId": latest_run_id,
    }


def _row_run(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "projectId": r["project_id"],
        "creativeId": r["creative_id"],
        "engine": r["engine"],
        "confidence": r["confidence"],
        "label": r["label"],
        "meta": json.loads(r["meta_json"] or "{}"),
        "overlays": json.loads(r["overlay_json"] or "{}"),
        "createdAt": r["created_at"],
    }


def ensure_default_project(owner_email: str) -> dict:
    """Create 'My creatives' if the user has no projects yet."""
    email = (owner_email or "").strip().lower()
    projects = list_projects(email)
    if projects:
        return projects[0]
    return create_project(email, "My creatives")


def list_projects(owner_email: str) -> list[dict]:
    email = (owner_email or "").strip().lower()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM projects WHERE owner_email = ? ORDER BY updated_at DESC",
            (email,),
        ).fetchall()
        out = []
        for r in rows:
            folders = [
                _row_folder(f)
                for f in conn.execute(
                    "SELECT * FROM folders WHERE project_id = ? ORDER BY name COLLATE NOCASE",
                    (r["id"],),
                ).fetchall()
            ]
            out.append(_row_project(r, folders))
        return out


def get_project(project_id: str, owner_email: str) -> dict | None:
    email = (owner_email or "").strip().lower()
    with _connect() as conn:
        r = conn.execute(
            "SELECT * FROM projects WHERE id = ? AND owner_email = ?",
            (project_id, email),
        ).fetchone()
        if not r:
            return None
        folders = [
            _row_folder(f)
            for f in conn.execute(
                "SELECT * FROM folders WHERE project_id = ? ORDER BY name COLLATE NOCASE",
                (project_id,),
            ).fetchall()
        ]
        return _row_project(r, folders)


def create_project(owner_email: str, name: str, client_name: str | None = None) -> dict:
    email = (owner_email or "").strip().lower()
    now = _now()
    pid = _nid("prj")
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO projects (id, name, client_name, owner_email, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (pid, (name or "Untitled project").strip(), (client_name or None), email, now, now),
        )
        conn.commit()
    return get_project(pid, email)  # type: ignore[return-value]


def update_project(
    project_id: str,
    owner_email: str,
    *,
    name: str | None = None,
    client_name: str | None = None,
) -> dict | None:
    email = (owner_email or "").strip().lower()
    proj = get_project(project_id, email)
    if not proj:
        return None
    new_name = name.strip() if isinstance(name, str) and name.strip() else proj["name"]
    new_client = client_name if client_name is not None else proj["clientName"]
    with _connect() as conn:
        conn.execute(
            "UPDATE projects SET name = ?, client_name = ?, updated_at = ? WHERE id = ?",
            (new_name, new_client, _now(), project_id),
        )
        conn.commit()
    return get_project(project_id, email)


def delete_project(project_id: str, owner_email: str) -> bool:
    email = (owner_email or "").strip().lower()
    proj = get_project(project_id, email)
    if not proj:
        return False
    # Remove files for creatives in this project
    with _connect() as conn:
        paths = [
            row["storage_path"]
            for row in conn.execute(
                "SELECT storage_path FROM creatives WHERE project_id = ?", (project_id,)
            ).fetchall()
        ]
        run_dirs = [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM analysis_runs WHERE project_id = ?", (project_id,)
            ).fetchall()
        ]
        conn.execute("DELETE FROM projects WHERE id = ? AND owner_email = ?", (project_id, email))
        conn.commit()
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass
    for rid in run_dirs:
        shutil.rmtree(_FILES_DIR / "runs" / rid, ignore_errors=True)
    return True


def create_folder(
    project_id: str, owner_email: str, name: str, parent_id: str | None = None
) -> dict | None:
    if not get_project(project_id, owner_email):
        return None
    fid = _nid("fld")
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO folders (id, project_id, name, parent_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (fid, project_id, (name or "New folder").strip(), parent_id, now),
        )
        conn.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id)
        )
        conn.commit()
        row = conn.execute("SELECT * FROM folders WHERE id = ?", (fid,)).fetchone()
    return _row_folder(row)


def delete_folder(folder_id: str, owner_email: str) -> bool:
    email = (owner_email or "").strip().lower()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT f.* FROM folders f
            JOIN projects p ON p.id = f.project_id
            WHERE f.id = ? AND p.owner_email = ?
            """,
            (folder_id, email),
        ).fetchone()
        if not row:
            return False
        # Unfile creatives rather than delete them
        conn.execute(
            "UPDATE creatives SET folder_id = NULL WHERE folder_id = ?", (folder_id,)
        )
        conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
        conn.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?", (_now(), row["project_id"])
        )
        conn.commit()
    return True


def list_creatives(project_id: str, owner_email: str) -> list[dict]:
    if not get_project(project_id, owner_email):
        return []
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT c.*, (
                SELECT r.id FROM analysis_runs r
                WHERE r.creative_id = c.id
                ORDER BY r.created_at DESC LIMIT 1
            ) AS latest_run_id
            FROM creatives c
            WHERE c.project_id = ?
            ORDER BY c.updated_at DESC
            """,
            (project_id,),
        ).fetchall()
        return [_row_creative(r, r["latest_run_id"]) for r in rows]


def get_creative(creative_id: str, owner_email: str) -> dict | None:
    email = (owner_email or "").strip().lower()
    with _connect() as conn:
        r = conn.execute(
            """
            SELECT c.*, (
                SELECT r.id FROM analysis_runs r
                WHERE r.creative_id = c.id
                ORDER BY r.created_at DESC LIMIT 1
            ) AS latest_run_id
            FROM creatives c
            JOIN projects p ON p.id = c.project_id
            WHERE c.id = ? AND p.owner_email = ?
            """,
            (creative_id, email),
        ).fetchone()
        if not r:
            return None
        return _row_creative(r, r["latest_run_id"])


def add_creative(
    project_id: str,
    owner_email: str,
    *,
    raw: bytes,
    file_name: str,
    mime_type: str,
    folder_id: str | None = None,
    name: str | None = None,
    width: int = 0,
    height: int = 0,
) -> dict | None:
    if not get_project(project_id, owner_email):
        return None
    cid = _nid("crv")
    now = _now()
    dest_dir = _FILES_DIR / "creatives" / project_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file_name).suffix.lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        ext = ".png"
    storage = dest_dir / f"{cid}{ext}"
    storage.write_bytes(raw)
    display = (name or Path(file_name).stem or "Creative").strip()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO creatives (
                id, project_id, folder_id, name, file_name, mime_type,
                width, height, storage_path, thumb_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                cid,
                project_id,
                folder_id,
                display,
                file_name,
                mime_type or "application/octet-stream",
                int(width),
                int(height),
                str(storage),
                now,
                now,
            ),
        )
        conn.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id)
        )
        conn.commit()
    return get_creative(cid, owner_email)


def read_creative_bytes(creative_id: str, owner_email: str) -> tuple[bytes, dict] | None:
    email = (owner_email or "").strip().lower()
    with _connect() as conn:
        r = conn.execute(
            """
            SELECT c.* FROM creatives c
            JOIN projects p ON p.id = c.project_id
            WHERE c.id = ? AND p.owner_email = ?
            """,
            (creative_id, email),
        ).fetchone()
        if not r:
            return None
        path = Path(r["storage_path"])
        if not path.is_file():
            return None
        return path.read_bytes(), _row_creative(r)


def delete_creative(creative_id: str, owner_email: str) -> bool:
    email = (owner_email or "").strip().lower()
    with _connect() as conn:
        r = conn.execute(
            """
            SELECT c.* FROM creatives c
            JOIN projects p ON p.id = c.project_id
            WHERE c.id = ? AND p.owner_email = ?
            """,
            (creative_id, email),
        ).fetchone()
        if not r:
            return False
        run_ids = [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM analysis_runs WHERE creative_id = ?", (creative_id,)
            ).fetchall()
        ]
        conn.execute("DELETE FROM creatives WHERE id = ?", (creative_id,))
        conn.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?", (_now(), r["project_id"])
        )
        conn.commit()
    try:
        Path(r["storage_path"]).unlink(missing_ok=True)
    except OSError:
        pass
    for rid in run_ids:
        shutil.rmtree(_FILES_DIR / "runs" / rid, ignore_errors=True)
    return True


def save_analysis_run(
    *,
    project_id: str,
    creative_id: str,
    owner_email: str,
    engine: str,
    confidence: float | None,
    meta: dict[str, Any],
    overlays_b64: dict[str, str],
    label: str | None = None,
) -> dict | None:
    """Persist run meta + PNG overlays (base64) to disk."""
    import base64

    if not get_creative(creative_id, owner_email):
        return None
    rid = _nid("run")
    now = _now()
    run_dir = _FILES_DIR / "runs" / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    overlay_paths: dict[str, str] = {}
    for kind, b64 in (overlays_b64 or {}).items():
        if not b64:
            continue
        try:
            raw = base64.b64decode(b64)
        except Exception:
            continue
        path = run_dir / f"{kind}.png"
        path.write_bytes(raw)
        overlay_paths[kind] = str(path)

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO analysis_runs (
                id, project_id, creative_id, engine, confidence, label,
                meta_json, overlay_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                project_id,
                creative_id,
                engine or "Unknown",
                confidence,
                label,
                json.dumps(meta or {}),
                json.dumps(overlay_paths),
                now,
            ),
        )
        conn.execute(
            "UPDATE creatives SET updated_at = ? WHERE id = ?", (now, creative_id)
        )
        conn.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id)
        )
        conn.commit()
        row = conn.execute("SELECT * FROM analysis_runs WHERE id = ?", (rid,)).fetchone()
    return _row_run(row)


def list_runs(creative_id: str, owner_email: str) -> list[dict]:
    if not get_creative(creative_id, owner_email):
        return []
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM analysis_runs
            WHERE creative_id = ?
            ORDER BY created_at DESC
            """,
            (creative_id,),
        ).fetchall()
        return [_row_run(r) for r in rows]


def get_run(run_id: str, owner_email: str) -> dict | None:
    email = (owner_email or "").strip().lower()
    with _connect() as conn:
        r = conn.execute(
            """
            SELECT ar.* FROM analysis_runs ar
            JOIN projects p ON p.id = ar.project_id
            WHERE ar.id = ? AND p.owner_email = ?
            """,
            (run_id, email),
        ).fetchone()
        if not r:
            return None
        return _row_run(r)


def read_overlay(run_id: str, owner_email: str, kind: str) -> bytes | None:
    run = get_run(run_id, owner_email)
    if not run:
        return None
    path = (run.get("overlays") or {}).get(kind)
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return p.read_bytes()


def library_snapshot(owner_email: str) -> dict:
    """Full library for studio boot: projects + creatives (all projects)."""
    email = (owner_email or "").strip().lower()
    ensure_default_project(email)
    projects = list_projects(email)
    creatives: list[dict] = []
    for p in projects:
        creatives.extend(list_creatives(p["id"], email))
    return {"projects": projects, "creatives": creatives}
