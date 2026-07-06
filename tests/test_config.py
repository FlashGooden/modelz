import pytest

from modelz import config
from modelz.errors import ConfigError


def test_load_api_token_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("REPLICATE_API_TOKEN=r8_test123\n")

    token = config.load_api_token(env_file)

    assert token == "r8_test123"


def test_load_api_token_uses_existing_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "r8_from_env")
    missing_env_file = tmp_path / ".env"

    token = config.load_api_token(missing_env_file)

    assert token == "r8_from_env"


def test_load_api_token_missing_raises_config_error(tmp_path, monkeypatch):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    missing_env_file = tmp_path / ".env"

    with pytest.raises(ConfigError):
        config.load_api_token(missing_env_file)
