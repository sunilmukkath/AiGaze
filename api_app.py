"""AI Gaze HTTP API + studio (Railway). Replaces Streamlit gateway."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from starlette.middleware.sessions import SessionMiddleware

import auth_users as auth
from engine.analyze import build_pdf_from_bundle, decode_upload, run_analysis

logger = logging.getLogger("aigaze")

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
STATIC = ROOT / "static"

SESSION_SECRET = (
    os.environ.get("AIGAZE_SESSION_SECRET")
    or os.environ.get("AUTH_BRIDGE_SECRET")
    or os.environ.get("BILLING_FULFILL_SECRET")
    or "aigaze-dev-session-secret-change-me"
).strip()

CHECKOUT_URL = (
    os.environ.get("PUBLIC_CHECKOUT_URL") or "https://www.elastictree.com/ai-gaze#pricing"
).strip()
ACCOUNTS_URL = (
    os.environ.get("ET_ACCOUNTS_URL")
    or os.environ.get("NEXT_PUBLIC_ET_ACCOUNTS_URL")
    or "https://www.elastictree.com"
).rstrip("/")
PUBLIC_URL = (
    os.environ.get("AIGAZE_PUBLIC_URL") or "https://aigaze-production.up.railway.app"
).rstrip("/")
SSO_ON = os.environ.get("ET_SSO", "").strip() == "1" or os.environ.get(
    "NEXT_PUBLIC_ET_SSO", ""
).strip() == "1"

app = FastAPI(title="AI Gaze", version="2.0.0")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="aigaze_session",
    same_site="lax",
    https_only=os.environ.get("AIGAZE_HTTPS_ONLY", "1").strip() != "0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        PUBLIC_URL,
        "https://www.elastictree.com",
        "https://elastictree.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

# In-memory PDF bundles keyed by one-shot token (per process).
_PDF_CACHE: dict[str, dict[str, Any]] = {}


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class RegisterBody(Credentials):
    password: str = Field(min_length=8)


class ResetRequest(BaseModel):
    email: EmailStr


class ResetConfirm(BaseModel):
    token: str
    password: str = Field(min_length=8)


def _public_user(user: dict | None, email: str | None = None) -> dict[str, Any]:
    if not user and email:
        return {"email": email, "admin": True}
    if not user:
        return {}
    return {
        "id": user.get("id"),
        "email": user.get("email"),
        "plan": user.get("plan"),
        "analyses_quota": user.get("analyses_quota"),
        "analyses_used": user.get("analyses_used"),
        "period_ends_at": user.get("period_ends_at"),
        "admin": bool(user.get("admin")),
    }


def _require_user(request: Request) -> dict[str, Any]:
    email = (request.session.get("email") or "").strip().lower()
    if not email:
        raise HTTPException(status_code=401, detail="Sign in required")
    user = auth.get_user_by_email(email) or {
        "id": request.session.get("user_id") or "session",
        "email": email,
        "admin": bool(request.session.get("admin")),
    }
    return user


def _verify_billing(raw: bytes, signature: str | None) -> bool:
    secret = (os.environ.get("BILLING_FULFILL_SECRET") or "").strip()
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _warm_models() -> None:
    """Load DeepGaze in the background so the first user analyse is faster."""
    try:
        import numpy as np
        from engine.core import compute_saliency

        tiny = np.zeros((64, 64, 3), dtype=np.uint8)
        tiny[:] = (32, 64, 128)
        compute_saliency(tiny, enable_tta=False)
        logger.info("DeepGaze warm-up complete")
    except Exception as exc:
        logger.warning("DeepGaze warm-up skipped: %s", exc)


@app.on_event("startup")
def _startup() -> None:
    try:
        auth.ensure_shared_admin()
    except Exception:
        pass
    threading.Thread(target=_warm_models, name="aigaze-warmup", daemon=True).start()


@app.get("/health")
@app.get("/_stcore/health")  # keep old Railway healthcheck green during cutover
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
@app.get("/studio", response_class=HTMLResponse)
def studio_page(request: Request) -> HTMLResponse:
    # Consume SSO bridge on first hit if present.
    bridge = request.query_params.get("et_bridge") or ""
    if SSO_ON and bridge and not request.session.get("email"):
        try:
            email = auth.consume_central_bridge(bridge)
            user = auth.ensure_sso_user(email)
            request.session["email"] = user["email"]
            request.session["user_id"] = user["id"]
            request.session["admin"] = False
            return RedirectResponse(url="/studio", status_code=303)
        except Exception as exc:
            html = (TEMPLATES / "studio.html").read_text(encoding="utf-8")
            html = html.replace("{{SSO_ERROR}}", str(exc))
            return HTMLResponse(html)

    access = request.query_params.get("access") or ""
    if access and access == auth.shared_admin_password():
        request.session["email"] = auth.SHARED_ADMIN_EMAIL
        request.session["user_id"] = "shared-admin"
        request.session["admin"] = True
        return RedirectResponse(url="/studio", status_code=303)

    path = TEMPLATES / "studio.html"
    html = path.read_text(encoding="utf-8")
    html = html.replace("{{SSO_ERROR}}", "")
    html = html.replace("{{CHECKOUT_URL}}", CHECKOUT_URL)
    html = html.replace("{{ACCOUNTS_URL}}", ACCOUNTS_URL)
    html = html.replace("{{PUBLIC_URL}}", PUBLIC_URL)
    html = html.replace("{{SSO_ON}}", "1" if SSO_ON else "0")
    return HTMLResponse(html)


@app.get("/api/me")
def me(request: Request) -> dict[str, Any]:
    email = (request.session.get("email") or "").strip().lower()
    if not email:
        return {"authenticated": False}
    user = auth.get_user_by_email(email)
    return {
        "authenticated": True,
        "user": _public_user(user, email),
        "checkout_url": CHECKOUT_URL,
        "sso": SSO_ON,
        "accounts_url": ACCOUNTS_URL,
    }


@app.post("/api/auth/signin")
def signin(body: Credentials, request: Request) -> dict[str, Any]:
    auth.ensure_shared_admin()
    user = auth.authenticate(body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    is_admin = bool(user.get("admin")) or (
        (user.get("email") or "").lower() == auth.SHARED_ADMIN_EMAIL
    )
    request.session["email"] = user["email"]
    request.session["user_id"] = user.get("id")
    request.session["admin"] = is_admin
    public = _public_user(user)
    public["admin"] = is_admin
    return {"ok": True, "user": public}


@app.post("/api/auth/register")
def register(body: RegisterBody, request: Request) -> dict[str, Any]:
    try:
        user = auth.create_user(body.email, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    request.session["email"] = user["email"]
    request.session["user_id"] = user["id"]
    request.session["admin"] = False
    return {"ok": True, "user": _public_user(user)}


@app.post("/api/auth/signout")
def signout(request: Request) -> dict[str, bool]:
    request.session.clear()
    return {"ok": True}


@app.post("/api/auth/forgot")
def forgot(body: ResetRequest) -> dict[str, Any]:
    return auth.request_password_reset(body.email)


@app.post("/api/auth/reset")
def reset(body: ResetConfirm) -> dict[str, bool]:
    try:
        auth.reset_password_with_token(body.token, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.get("/api/auth/sso-url")
def sso_url() -> dict[str, str]:
    from urllib.parse import quote

    ret = quote(PUBLIC_URL, safe="")
    return {"url": f"{ACCOUNTS_URL}/accounts/signin?returnUrl={ret}"}


@app.post("/api/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    high_confidence: bool = Form(False),
) -> JSONResponse:
    user = _require_user(request)
    if not user.get("admin"):
        db_user = auth.get_user_by_email(user["email"])
        ok, reason = auth.can_run_analysis(db_user)
        if not ok:
            raise HTTPException(status_code=402, detail=reason or "Upgrade required")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")
    try:
        img = decode_upload(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        # CPU DeepGaze is blocking — keep the event loop free.
        result = await asyncio.to_thread(
            run_analysis, img, high_confidence=bool(high_confidence)
        )
    except Exception as exc:
        logger.exception("analyse failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    if not user.get("admin"):
        auth.consume_analysis(user["email"])

    token = secrets.token_urlsafe(16)
    _PDF_CACHE[token] = result.pop("_pdf_bundle")
    # Cap cache size
    while len(_PDF_CACHE) > 32:
        _PDF_CACHE.pop(next(iter(_PDF_CACHE)))

    result["pdf_token"] = token
    return JSONResponse(result)


@app.get("/api/report.pdf")
def report_pdf(request: Request, token: str) -> Response:
    _require_user(request)
    bundle = _PDF_CACHE.get(token)
    if not bundle:
        raise HTTPException(status_code=404, detail="Report expired — re-run analysis")
    pdf_bytes = build_pdf_from_bundle(bundle)
    if not pdf_bytes:
        raise HTTPException(status_code=500, detail="PDF generation unavailable")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="AI_Gaze_Report.pdf"'},
    )


@app.post("/api/billing/fulfill")
async def billing_fulfill(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not _verify_billing(raw, request.headers.get("X-ET-Billing-Signature")):
        raise HTTPException(status_code=401, detail="Invalid signature")
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    email = str(body.get("email") or "").strip().lower()
    plan = str(body.get("plan") or "starter")
    period = str(body.get("period") or "monthly")
    txnid = str(body.get("txnid") or "").strip()
    if not email or "@" not in email or not txnid:
        raise HTTPException(status_code=400, detail="email and txnid required")
    return auth.apply_paid_subscription(
        email=email,
        plan=plan,
        period=period,
        txnid=txnid,
        sku=body.get("sku"),
        paid_at=body.get("paidAt"),
    )


@app.get("/favicon.png")
def favicon() -> FileResponse:
    for candidate in (
        ROOT / "et_favicon.png",
        ROOT / "favicon-32.png",
        ROOT / "apps" / "web" / "public" / "favicon-32.png",
        ROOT / "aigaze_logo.png",
    ):
        if candidate.is_file():
            return FileResponse(candidate)
    raise HTTPException(status_code=404)
