"""Framework-independent authentication configuration for the web surface."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path


def load_user_tokens(configured_path: str) -> dict[str, str]:
    if not configured_path.strip():
        return {}
    payload = json.loads(
        Path(configured_path).expanduser().read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict) or not payload:
        raise ValueError("BPSD_ALIGNER_USERS_FILE must contain a non-empty JSON object")
    users = {
        str(username).strip(): str(token)
        for username, token in payload.items()
        if str(username).strip() and str(token)
    }
    if len(users) != len(payload):
        raise ValueError("every configured user requires a non-empty username and token")
    return users


def verify_access_token(supplied: str, configured: str) -> bool:
    """Verify plaintext or ``sha256:<hex>`` deployment tokens."""

    if configured.startswith("sha256:"):
        expected = configured.removeprefix("sha256:").lower()
        actual = hashlib.sha256(supplied.encode("utf-8")).hexdigest()
        return len(expected) == 64 and hmac.compare_digest(actual, expected)
    return bool(configured) and hmac.compare_digest(supplied, configured)


def configuration_digest(configured_token: str) -> str:
    """Bind an authenticated session to the current token configuration."""

    return hashlib.sha256(configured_token.encode("utf-8")).hexdigest()
