"""Email + password accounts for AI Gaze (SQLite)."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

_DATA_DIR = Path(os.environ.get("AIGAZE_DATA_DIR", Path(__file__).resolve().parent / ".data"))
_DB_PATH = _DATA_DIR / "users.sqlite"


def _connect() -> sqlite3.Connection:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            token_hash TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=bytes.fromhex(salt),
        n=16384,
        r=8,
        p=1,
        dklen=64,
    ).hex()
    return f"scrypt${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    parts = stored.split("$")
    if len(parts) != 3 or parts[0] != "scrypt":
        return False
    _, salt, digest_hex = parts
    try:
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt),
            n=16384,
            r=8,
            p=1,
            dklen=len(expected),
        )
    except Exception:
        return False
    return secrets.compare_digest(expected, actual)


def get_user_by_email(email: str) -> dict | None:
    normalized = email.strip().lower()
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?",
            (normalized,),
        ).fetchone()
    if not row:
        return None
    return {"id": row["id"], "email": row["email"], "password_hash": row["password_hash"]}


def create_user(email: str, password: str) -> dict:
    normalized = email.strip().lower()
    if get_user_by_email(normalized):
        raise ValueError("Account already exists")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    user_id = str(uuid.uuid4())
    password_hash = hash_password(password)
    created = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, normalized, password_hash, created),
        )
        conn.commit()
    return {"id": user_id, "email": normalized}


def authenticate(email: str, password: str) -> dict | None:
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        return None
    return {"id": user["id"], "email": user["email"]}


def update_password(email: str, password: str) -> bool:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    normalized = email.strip().lower()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash = ? WHERE email = ?",
            (hash_password(password), normalized),
        )
        conn.commit()
        return cur.rowcount > 0


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_reset_token(email: str) -> str:
    normalized = email.strip().lower()
    token = secrets.token_hex(32)
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    with _connect() as conn:
        conn.execute("DELETE FROM password_reset_tokens WHERE email = ?", (normalized,))
        conn.execute(
            "INSERT INTO password_reset_tokens (token_hash, email, expires_at) VALUES (?, ?, ?)",
            (_hash_token(token), normalized, expires),
        )
        conn.commit()
    return token


def consume_reset_token(token: str) -> str | None:
    now = datetime.now(timezone.utc).isoformat()
    th = _hash_token(token)
    with _connect() as conn:
        conn.execute("DELETE FROM password_reset_tokens WHERE expires_at <= ?", (now,))
        row = conn.execute(
            "SELECT email FROM password_reset_tokens WHERE token_hash = ?",
            (th,),
        ).fetchone()
        if not row:
            conn.commit()
            return None
        email = row["email"]
        conn.execute("DELETE FROM password_reset_tokens WHERE token_hash = ?", (th,))
        conn.commit()
        return email


def public_reset_url(token: str) -> str:
    origin = (
        os.environ.get("AIGAZE_PUBLIC_URL")
        or os.environ.get("NEXT_PUBLIC_APP_URL")
        or "https://www.elastictree.com/ai-gaze"
    ).rstrip("/")
    # Streamlit studio may be on a separate host; query param handled in app.py
    sep = "&" if "?" in origin else "?"
    return f"{origin}{sep}reset={token}"


def send_reset_email(to: str, reset_url: str) -> dict:
    api_key = (os.environ.get("RESEND_API_KEY") or "").strip()
    if not api_key:
        print(f"[AI Gaze] Password reset for {to}: {reset_url}")
        return {"status": "simulated", "reset_url": reset_url}

    from_addr = (
        (os.environ.get("RESEND_FROM_EMAIL") or "").strip()
        or "Elastic Tree <onboarding@resend.dev>"
    )
    html = f"""
      <div style="font-family:system-ui,sans-serif;line-height:1.5">
        <h2>Reset your AI Gaze password</h2>
        <p>This link expires in 1 hour.</p>
        <p><a href="{reset_url}">Choose a new password</a></p>
      </div>
    """
    payload = {
        "from": from_addr,
        "to": [to],
        "subject": "Reset your AI Gaze password",
        "html": html,
        "text": f"Reset your password: {reset_url}",
    }
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            if resp.status >= 400:
                return {"status": "error", "detail": resp.read().decode()}
        return {"status": "sent"}
    except urllib.error.URLError as exc:
        return {"status": "error", "detail": str(exc)}


def request_password_reset(email: str) -> dict:
    """Always returns a generic ok message; may include devResetUrl when email is simulated."""
    ok = {"ok": True, "message": "If an account exists for that email, we sent a password reset link."}
    user = get_user_by_email(email)
    if not user:
        return ok
    token = create_reset_token(user["email"])
    reset_url = public_reset_url(token)
    result = send_reset_email(user["email"], reset_url)
    if result.get("status") == "error":
        return {"ok": False, "error": "Could not send reset email."}
    if result.get("status") == "simulated":
        return {**ok, "devResetUrl": reset_url}
    return ok


def reset_password_with_token(token: str, password: str) -> None:
    email = consume_reset_token(token)
    if not email:
        raise ValueError("This reset link is invalid or has expired.")
    update_password(email, password)
