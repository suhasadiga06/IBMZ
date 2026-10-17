"""Pseudonymous IDs: driver tags never leave the charge point exchange in clear.

The OCPP idTag (an RFID card or app account) is replaced by an HMAC-SHA256 token keyed
with a secret that stays on the central system, so stored sessions, incidents and the
dashboard can follow one driver's charges without knowing who the driver is.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from pathlib import Path

_KEY: bytes | None = None


def key() -> bytes:
    global _KEY
    if _KEY is None:
        path = Path(os.environ.get("CHARGEGUARD_KEY_FILE", Path.home() / ".chargeguard" / "key"))
        if path.exists():
            _KEY = path.read_bytes()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            _KEY = secrets.token_bytes(32)
            path.write_bytes(_KEY)
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
    return _KEY


def token(id_tag: str, kind: str = "drv") -> str:
    return f"{kind}_" + hmac.new(key(), id_tag.encode(), hashlib.sha256).hexdigest()[:12]
