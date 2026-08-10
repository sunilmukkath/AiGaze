"""Email + password accounts for AI Gaze (SQLite).

Matches Elastic Tree studio auth used by DataWiz / QualView / TScribe:
- email + password register / sign-in / forgot-password
- shared pilot admin (admin@elastictree.com / elastic2026) always available
"""

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

# Shared Elastic Tree pilot admin — same credentials across ET studio apps.
SHARED_ADMIN_EMAIL = (
    os.environ.get("ET_ADMIN_EMAIL") or "admin@elastictree.com"
).strip().lower()


def employee_domain() -> str:
    return (os.environ.get("ET_EMPLOYEE_DOMAIN") or "elastictree.com").strip().lower()


def is_et_employee_email(email: str | None) -> bool:
    normalized = (email or "").strip().lower()
    domain = employee_domain()
    if not normalized or not domain or "@" not in normalized:
        return False
    return normalized.endswith(f"@{domain}")


def shared_admin_password() -> str:
    return (
        os.environ.get("ET_ADMIN_PASSWORD")
        or os.environ.get("AIGAZE_ACCESS_PASSWORD")
        or "elastic2026"
    ).strip()


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
    _ensure_user_columns(conn)
    conn.commit()
    return conn


def _ensure_user_columns(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    for name, decl in (
        ("plan", "TEXT"),
        ("analyses_quota", "INTEGER NOT NULL DEFAULT 0"),
        ("analyses_used", "INTEGER NOT NULL DEFAULT 0"),
        ("period_ends_at", "TEXT"),
        ("billing_period", "TEXT"),
        ("last_txn_id", "TEXT"),
        ("last_sku", "TEXT"),
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE users ADD COLUMN {name} {decl}")


PLAN_QUOTAS = {
    "starter": 20,
    "growth": 80,
    "enterprise": 999_999,
}


def get_user_by_email(email: str) -> dict | None:
    normalized = email.strip().lower()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, email, password_hash, plan, analyses_quota, analyses_used,
                   period_ends_at, billing_period, last_txn_id, last_sku
            FROM users WHERE email = ?
            """,
            (normalized,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "email": row["email"],
        "password_hash": row["password_hash"],
        "plan": row["plan"],
        "analyses_quota": int(row["analyses_quota"] or 0),
        "analyses_used": int(row["analyses_used"] or 0),
        "period_ends_at": row["period_ends_at"],
        "billing_period": row["billing_period"],
        "last_txn_id": row["last_txn_id"],
        "last_sku": row["last_sku"],
    }


def _period_end_iso(period: str, from_iso: str) -> str:
    d = datetime.fromisoformat(from_iso.replace("Z", "+00:00"))
    if period == "yearly":
        d = d.replace(year=d.year + 1)
    else:
        month = d.month + 1
        year = d.year
        if month > 12:
            month = 1
            year += 1
        day = min(d.day, 28)
        d = d.replace(year=year, month=month, day=day)
    return d.astimezone(timezone.utc).isoformat()


def is_plan_active(user: dict | None) -> bool:
    if not user:
        return False
    email = (user.get("email") or "").strip().lower()
    if email == SHARED_ADMIN_EMAIL or is_et_employee_email(email):
        return True
    plan = (user.get("plan") or "").lower()
    if plan not in PLAN_QUOTAS:
        return False
    ends = user.get("period_ends_at")
    if ends:
        try:
            if datetime.fromisoformat(ends.replace("Z", "+00:00")) < datetime.now(timezone.utc):
                return False
        except ValueError:
            pass
    return True


def can_run_analysis(user: dict | None) -> tuple[bool, str]:
    if not user:
        return False, "Sign in to run analyses."
    email = (user.get("email") or "").strip().lower()
    if user.get("admin") or email == SHARED_ADMIN_EMAIL or is_et_employee_email(email):
        return True, ""
    if not is_plan_active(user):
        checkout = (
            os.environ.get("PUBLIC_CHECKOUT_URL")
            or "https://www.elastictree.com/ai-gaze#pricing"
        )
        return False, f"No active paid plan. Subscribe at {checkout}"
    used = int(user.get("analyses_used") or 0)
    quota = int(user.get("analyses_quota") or 0)
    if quota > 0 and used >= quota:
        return False, "Analysis quota exhausted for this billing period. Upgrade or renew."
    return True, ""


def consume_analysis(email: str) -> bool:
    normalized = email.strip().lower()
    with _connect() as conn:
        row = conn.execute(
            "SELECT analyses_used FROM users WHERE email = ?",
            (normalized,),
        ).fetchone()
        if not row:
            return False
        conn.execute(
            "UPDATE users SET analyses_used = ? WHERE email = ?",
            (int(row["analyses_used"] or 0) + 1, normalized),
        )
        conn.commit()
    return True


def apply_paid_subscription(
    *,
    email: str,
    plan: str,
    period: str,
    txnid: str,
    sku: str | None = None,
    paid_at: str | None = None,
) -> dict:
    email_n = email.strip().lower()
    plan_n = (plan or "starter").lower().strip()
    if plan_n not in PLAN_QUOTAS:
        plan_n = "starter"
    period_n = "yearly" if period == "yearly" else "monthly"
    paid = paid_at or datetime.now(timezone.utc).isoformat()
    ends = _period_end_iso(period_n, paid)
    quota = PLAN_QUOTAS[plan_n]

    existing = get_user_by_email(email_n)
    if existing and existing.get("last_txn_id") == txnid:
        return {
            "ok": True,
            "alreadyApplied": True,
            "email": email_n,
            "plan": existing.get("plan") or plan_n,
            "period_ends_at": existing.get("period_ends_at") or ends,
        }

    with _connect() as conn:
        if existing:
            conn.execute(
                """
                UPDATE users SET plan=?, analyses_quota=?, analyses_used=0,
                  period_ends_at=?, billing_period=?, last_txn_id=?, last_sku=?
                WHERE email=?
                """,
                (plan_n, quota, ends, period_n, txnid, sku, email_n),
            )
            created = False
        else:
            user_id = str(uuid.uuid4())
            created_at = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO users (
                  id, email, password_hash, created_at, plan, analyses_quota, analyses_used,
                  period_ends_at, billing_period, last_txn_id, last_sku
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    email_n,
                    f"payu-pending:{txnid}",
                    created_at,
                    plan_n,
                    quota,
                    ends,
                    period_n,
                    txnid,
                    sku,
                ),
            )
            created = True
        conn.commit()

    return {
        "ok": True,
        "created": created,
        "email": email_n,
        "plan": plan_n,
        "period_ends_at": ends,
        "analyses_quota": quota,
        "txnid": txnid,
    }


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


def create_user(email: str, password: str) -> dict:
    normalized = email.strip().lower()
    existing = get_user_by_email(normalized)
    if existing:
        if str(existing.get("password_hash") or "").startswith("payu-pending:"):
            update_password(normalized, password)
            return {"id": existing["id"], "email": normalized}
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


def ensure_sso_user(email: str) -> dict:
    """Provision local AI Gaze user from central SSO email."""
    normalized = (email or "").strip().lower()
    if not normalized or "@" not in normalized:
        raise ValueError("Valid email required")
    existing = get_user_by_email(normalized)
    if not existing:
        create_user(normalized, f"sso:{uuid.uuid4().hex}")
    if is_et_employee_email(normalized):
        apply_paid_subscription(
            email=normalized,
            plan="enterprise",
            period="yearly",
            txnid=f"et-employee:{normalized}",
            sku="et-employee",
        )
    user = get_user_by_email(normalized)
    if not user:
        raise ValueError("Failed to provision SSO user")
    return {"id": user["id"], "email": user["email"]}


def consume_central_bridge(code: str) -> str:
    """Exchange one-time et_bridge code for email via elastictree.com accounts."""
    import hmac

    base = (
        os.environ.get("ET_ACCOUNTS_URL")
        or os.environ.get("NEXT_PUBLIC_ET_ACCOUNTS_URL")
        or "https://www.elastictree.com"
    ).rstrip("/")
    secret = (
        os.environ.get("AUTH_BRIDGE_SECRET") or os.environ.get("BILLING_FULFILL_SECRET") or ""
    ).strip()
    if not secret:
        raise ValueError(
            "AUTH_BRIDGE_SECRET (or BILLING_FULFILL_SECRET) is required for SSO bridge"
        )
    body = json.dumps({"code": code}).encode()
    headers = {
        "Content-Type": "application/json",
        "X-ET-Bridge-Signature": hmac.new(
            secret.encode(), code.encode(), hashlib.sha256
        ).hexdigest(),
    }
    req = urllib.request.Request(
        f"{base}/api/auth/bridge/consume",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode()).get("error")
        except Exception:
            detail = None
        raise ValueError(detail or "Invalid or expired SSO bridge code") from exc
    except urllib.error.URLError as exc:
        raise ValueError("Accounts service unavailable") from exc
    email = (data.get("email") or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("Invalid SSO response")
    return email


def authenticate(email: str, password: str) -> dict | None:
    """QualView-style: try account password, then shared pilot password."""
    ensure_shared_admin()
    normalized = (email or "").strip().lower()
    if normalized and "@" in normalized:
        user = get_user_by_email(normalized)
        if user and verify_password(password, user["password_hash"]):
            return {
                "id": user["id"],
                "email": user["email"],
                "plan": user.get("plan"),
                "analyses_quota": user.get("analyses_quota"),
                "analyses_used": user.get("analyses_used"),
                "period_ends_at": user.get("period_ends_at"),
            }
    # Soft-launch / admin gate (same across ET studios)
    if password and password == shared_admin_password():
        return {
            "id": "shared-admin",
            "email": normalized if normalized and "@" in normalized else SHARED_ADMIN_EMAIL,
            "admin": True,
        }
    return None


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


def ensure_shared_admin() -> None:
    """Create or reset shared admin so login always works after cold starts."""
    email = SHARED_ADMIN_EMAIL
    password = shared_admin_password()
    if "@" not in email or not password:
        return
    existing = get_user_by_email(email)
    if not existing:
        user_id = str(uuid.uuid4())
        created = datetime.now(timezone.utc).isoformat()
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (user_id, email, hash_password(password), created),
            )
            conn.commit()
        return
    if not verify_password(password, existing["password_hash"]):
        update_password(email, password)


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
        or os.environ.get("RAILWAY_PUBLIC_DOMAIN")
        or "https://aigaze-production.up.railway.app"
    ).rstrip("/")
    if not origin.startswith("http"):
        origin = f"https://{origin}"
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
    ensure_shared_admin()
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
