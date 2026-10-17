"""Privacy controls: what the screening service may know and keep.

* Account numbers are replaced on arrival by keyed HMAC tokens. The models and
  the case store only ever see tokens; the bank can map back, we cannot.
* Message text is encrypted with AES-256-GCM wherever it is stored: as
  evidence for open cases, and in each relationship's state, which keeps the
  last 20 references so a case's record can start before the case opened.
  Each ciphertext is bound to its context (case, relationship, preferences),
  so it cannot be moved to another record and still decrypt.
* The key lives outside the repository (SAFEPAY_KEY_FILE, default
  ~/.safepay/key) and is created on first use.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_FILE = Path(os.environ.get("SAFEPAY_KEY_FILE", Path.home() / ".safepay" / "key"))


def _master_key() -> bytes:
    if not KEY_FILE.exists():
        KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        KEY_FILE.write_bytes(os.urandom(32))
        KEY_FILE.chmod(0o600)
    return KEY_FILE.read_bytes()


class Privacy:
    def __init__(self, key: bytes | None = None):
        master = key or _master_key()
        # Separate sub-keys so a leak of one use does not expose the other.
        self._tok_key = hmac.new(master, b"safepay-token", hashlib.sha256).digest()
        self._aead = AESGCM(hmac.new(master, b"safepay-evidence", hashlib.sha256).digest())

    def token(self, account: str) -> str:
        digest = hmac.new(self._tok_key, account.strip().upper().encode(), hashlib.sha256).digest()
        return "acct_" + base64.b32encode(digest[:10]).decode().lower()

    def seal(self, text: str, context: str) -> str:
        nonce = os.urandom(12)
        ct = self._aead.encrypt(nonce, text.encode(), context.encode())
        return base64.b64encode(nonce + ct).decode()

    def open(self, blob: str, context: str) -> str:
        raw = base64.b64decode(blob)
        return self._aead.decrypt(raw[:12], raw[12:], context.encode()).decode()
