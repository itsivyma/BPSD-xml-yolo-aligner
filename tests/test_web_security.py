import hashlib
import json

import pytest

from bpsd_aligner.web_security import (
    configuration_digest,
    load_user_tokens,
    verify_access_token,
)


def test_web_tokens_support_hashed_configuration(tmp_path) -> None:
    token = "long-secret-token"
    configured = "sha256:" + hashlib.sha256(token.encode()).hexdigest()
    path = tmp_path / "users.json"
    path.write_text(json.dumps({"reviewer": configured}), encoding="utf-8")

    users = load_user_tokens(str(path))
    assert verify_access_token(token, users["reviewer"])
    assert not verify_access_token("wrong", users["reviewer"])
    assert configuration_digest(configured) == configuration_digest(configured)


def test_web_user_file_rejects_empty_identity(tmp_path) -> None:
    path = tmp_path / "users.json"
    path.write_text(json.dumps({"": "token"}), encoding="utf-8")

    with pytest.raises(ValueError, match="every configured user"):
        load_user_tokens(str(path))
