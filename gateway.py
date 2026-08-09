"""Public HTTP gateway for AI Gaze: PayU fulfill + reverse-proxy to Streamlit.

Listens on $PORT (Railway). Streamlit runs on an internal port.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from auth_users import apply_paid_subscription

STREAMLIT_PORT = int(os.environ.get("AIGAZE_STREAMLIT_PORT", "8501"))
PUBLIC_PORT = int(os.environ.get("PORT", "8080"))


def _verify(raw: bytes, signature: str | None) -> bool:
    secret = (os.environ.get("BILLING_FULFILL_SECRET") or "").strip()
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path.rstrip("/") == "/api/billing/fulfill":
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            if not _verify(raw, self.headers.get("X-ET-Billing-Signature")):
                self._json(401, {"error": "Invalid signature"})
                return
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                self._json(400, {"error": "Invalid JSON"})
                return
            email = str(body.get("email") or "").strip().lower()
            plan = str(body.get("plan") or "starter")
            period = str(body.get("period") or "monthly")
            txnid = str(body.get("txnid") or "").strip()
            if not email or "@" not in email or not txnid:
                self._json(400, {"error": "email and txnid required"})
                return
            result = apply_paid_subscription(
                email=email,
                plan=plan,
                period=period,
                txnid=txnid,
                sku=body.get("sku"),
                paid_at=body.get("paidAt"),
            )
            self._json(200, result)
            return
        self._proxy()

    def do_GET(self) -> None:  # noqa: N802
        self._proxy()

    def do_PUT(self) -> None:  # noqa: N802
        self._proxy()

    def do_DELETE(self) -> None:  # noqa: N802
        self._proxy()

    def do_PATCH(self) -> None:  # noqa: N802
        self._proxy()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._proxy()

    def do_HEAD(self) -> None:  # noqa: N802
        self._proxy()

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        target = f"http://127.0.0.1:{STREAMLIT_PORT}{self.path}"
        req = urllib.request.Request(target, data=body, method=self.command)
        # Forward headers except hop-by-hop
        skip = {"host", "content-length", "transfer-encoding", "connection"}
        for key, value in self.headers.items():
            if key.lower() in skip:
                continue
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = resp.read()
                self.send_response(resp.status)
                for key, value in resp.headers.items():
                    if key.lower() in {
                        "transfer-encoding",
                        "connection",
                        "content-encoding",
                        "content-length",
                    }:
                        continue
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)
        except urllib.error.HTTPError as err:
            payload = err.read()
            self.send_response(err.code)
            self.send_header("Content-Type", err.headers.get("Content-Type", "text/plain"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as exc:
            msg = f"Upstream error: {exc}".encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)


def _wait_streamlit(timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{STREAMLIT_PORT}/_stcore/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status < 500:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Streamlit failed to become healthy")


def main() -> None:
    env = os.environ.copy()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            f"--server.port={STREAMLIT_PORT}",
            "--server.address=127.0.0.1",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ],
        env=env,
    )

    try:
        _wait_streamlit()
        server = ThreadingHTTPServer(("0.0.0.0", PUBLIC_PORT), GatewayHandler)
        print(f"[AI Gaze] gateway on :{PUBLIC_PORT} → streamlit :{STREAMLIT_PORT}", flush=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        proc.wait()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
