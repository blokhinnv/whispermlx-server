import os
from unittest.mock import patch

from whispermlx_server.config import Settings


def test_settings_from_env_defaults() -> None:
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings.from_env()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8081
    assert settings.hf_token is None


def test_settings_from_env() -> None:
    env = {
        "WHISPERMLX_HOST": "0.0.0.0",
        "WHISPERMLX_PORT": "9090",
        "HF_TOKEN": "hf_secret",
    }
    with patch.dict(os.environ, env, clear=True):
        settings = Settings.from_env()
    assert settings.host == "0.0.0.0"
    assert settings.port == 9090
    assert settings.hf_token == "hf_secret"


def test_settings_with_overrides() -> None:
    base = Settings(host="127.0.0.1", port=8081, hf_token=None)
    updated = base.with_overrides(port=9000, hf_token="hf_x")
    assert updated.port == 9000
    assert updated.hf_token == "hf_x"
    assert updated.host == "127.0.0.1"
