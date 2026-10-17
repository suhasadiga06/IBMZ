"""Who may call what, and how often.

Two modes, chosen with SAFEPAY_MODE:

  demo (default)  The demo pages and /demo endpoints are on. Locally nothing
                  asks for a password. With a demo password set (for a public
                  link), every page and API needs HTTP Basic auth as "judge",
                  and stricter rate limits apply.
  production      The demo pages and /demo endpoints are switched off (404).
                  Bank systems need an API key (X-API-Key), analysts need a
                  personal token (X-Analyst-Token), and the analyst's identity
                  in the audit log comes from that token, never from the request.

Pack verification (/v1/packs/verify, /v1/evidence/public-key) stays open in
both modes: police and courts must be able to check a pack without an account.

Secrets live in files outside the repository (default ~/.safepay/), created by
`python3 -m safepay.scripts.make_keys`.
"""
from __future__ import annotations

import base64
import hmac
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

HOME = Path(os.environ.get("SAFEPAY_SECRETS_DIR", Path.home() / ".safepay"))
OPEN_PATHS = ("/healthz", "/v1/packs/verify", "/v1/evidence/public-key")
DEMO_PREFIXES = ("/demo/", "/analyst", "/system", "/pack/", "/static/")
DEMO_EXACT = ("/", "/api/system", "/api/model-metrics")
DEMO_ANALYST = "analyst-demo"


def _read(path: Path) -> list[str]:
    try:
        return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    except OSError:
        return []


@dataclass
class Settings:
    mode: str = "demo"
    api_keys: list[str] = field(default_factory=list)
    analyst_tokens: dict[str, str] = field(default_factory=dict)   # token -> analyst name
    demo_password: str | None = None
    max_body_bytes: int = 1_000_000

    @property
    def production(self) -> bool:
        return self.mode == "production"

    @property
    def public(self) -> bool:
        return self.demo_password is not None and not self.production

    @classmethod
    def from_env(cls) -> "Settings":
        tokens = {}
        for line in _read(Path(os.environ.get("SAFEPAY_ANALYST_TOKENS_FILE", HOME / "analyst_tokens"))):
            tok, _, name = line.partition(" ")
            if tok and name:
                tokens[tok] = name.strip()
        pw_file = Path(os.environ.get("SAFEPAY_DEMO_PASSWORD_FILE", HOME / "demo_password"))
        pw = os.environ.get("SAFEPAY_DEMO_PASSWORD") or (_read(pw_file)[0] if os.environ.get("SAFEPAY_PUBLIC") == "1"
                                                          and _read(pw_file) else None)
        return cls(mode=os.environ.get("SAFEPAY_MODE", "demo"),
                   api_keys=_read(Path(os.environ.get("SAFEPAY_API_KEYS_FILE", HOME / "api_keys"))),
                   analyst_tokens=tokens, demo_password=pw)


settings = Settings.from_env()


def _match(secret: str | None, candidates) -> str | None:
    """Constant-time comparison against every candidate."""
    if not secret:
        return None
    found = None
    for c in candidates:
        if hmac.compare_digest(secret.encode(), c.encode()):
            found = c
    return found


# -- dependencies for routes --------------------------------------------------------
def require_bank(request: Request) -> None:
    """Bank systems: payment screening and recipient choices."""
    if settings.production and not _match(request.headers.get("x-api-key"), settings.api_keys):
        raise HTTPException(401, "missing or unknown API key", headers={"WWW-Authenticate": "ApiKey"})


def analyst(request: Request) -> str:
    """The analyst making this request, for the audit log. Never taken from a header the caller chooses."""
    if not settings.production:
        return DEMO_ANALYST
    tok = _match(request.headers.get("x-analyst-token"), settings.analyst_tokens)
    if not tok:
        raise HTTPException(401, "missing or unknown analyst token")
    return settings.analyst_tokens[tok]


# -- rate limits ----------------------------------------------------------------------
# (method or None for any, path prefix, requests, per seconds). First match wins; the public demo is stricter.
# Only starting a background replay is expensive; the console polls its progress with GET.
LIMITS = [("POST", "/demo/background", 1, 300), ("GET", "/demo/background", 120, 60), (None, "/demo/reset", 6, 60),
          (None, "/demo/script/", 20, 60), (None, "/demo/send", 30, 60), (None, "/v1/cases/", 60, 60),
          (None, "/v1/packs/verify", 30, 60), (None, "/v1/screen", 6000, 60)]
PUBLIC_DEFAULT, PRIVATE_DEFAULT = (240, 60), (100_000, 60)


class RateLimiter:
    def __init__(self):
        self.hits: dict[tuple[str, str], deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, client: str, path: str, public: bool, method: str = "GET") -> tuple[bool, int]:
        rule = next(((m, p, n, s) for m, p, n, s in LIMITS if path.startswith(p) and m in (None, method)), None)
        if rule is None:
            n, s = PUBLIC_DEFAULT if public else PRIVATE_DEFAULT
            key = "*"
        else:
            m, p, n, s = rule
            key = p if m is None else f"{m} {p}"
            if not public and not p.startswith("/demo/"):
                n *= 10
        now = time.monotonic()
        with self.lock:
            q = self.hits[(client, key)]
            while q and now - q[0] > s:
                q.popleft()
            if len(q) >= n:
                return False, int(s - (now - q[0])) + 1
            q.append(now)
        return True, 0

    def reset(self) -> None:
        with self.lock:
            self.hits.clear()


limiter = RateLimiter()

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": ("default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                                "font-src https://fonts.gstatic.com; img-src 'self' data:; "
                                "connect-src 'self'; frame-ancestors 'none'; form-action 'self'"),
    "Cache-Control": "no-store",
}


def _basic_ok(header: str | None) -> bool:
    if not header or not header.lower().startswith("basic "):
        return False
    try:
        user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
    except Exception:
        return False
    return hmac.compare_digest(user.encode(), b"judge") and hmac.compare_digest(pw.encode(), settings.demo_password.encode())


def is_demo_path(path: str) -> bool:
    return path in DEMO_EXACT or path.startswith(DEMO_PREFIXES)


async def middleware(request: Request, call_next) -> Response:
    path = request.url.path
    # 1. Production: the demo does not exist.
    if settings.production and is_demo_path(path):
        return JSONResponse({"detail": "Not Found"}, status_code=404, headers=SECURITY_HEADERS)
    # 2. Public demo: everything except the open paths needs the demo password.
    if settings.public and path not in OPEN_PATHS and not _basic_ok(request.headers.get("authorization")):
        return PlainTextResponse("SafePay demo: sign in with the user name and password from the submission.",
                                 status_code=401, headers={"WWW-Authenticate": 'Basic realm="SafePay demo"', **SECURITY_HEADERS})
    # 3. Body size, before anything reads the body.
    try:
        if int(request.headers.get("content-length") or 0) > settings.max_body_bytes:
            return JSONResponse({"detail": "request too large"}, status_code=413, headers=SECURITY_HEADERS)
    except ValueError:
        return JSONResponse({"detail": "bad content-length"}, status_code=400, headers=SECURITY_HEADERS)
    # 4. Rate limits per client address.
    ok, retry = limiter.allow(request.client.host if request.client else "?", path, settings.public, request.method)
    if not ok:
        return JSONResponse({"detail": "too many requests, slow down"}, status_code=429,
                            headers={"Retry-After": str(retry), **SECURITY_HEADERS})
    response = await call_next(request)
    for k, v in SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    return response
