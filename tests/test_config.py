# start tests/test_config.py
"""Tests for nanoclaw.config module.

Covers load_config, load_credentials, AppConfig validation, and all
sub-config model defaults and property conversions.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from nanoclaw.config import (
    AppConfig,
    AssistantConfig,
    ContainerConfig,
    Credentials,
    LoggingConfig,
    PathsConfig,
    TimingConfig,
    load_config,
    load_credentials,
)


# ---------------------------------------------------------------------------
# Tests: load_config()
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for the load_config() function."""

    def test_load_config_with_valid_yaml(self, tmp_path: Path) -> None:
        """Valid YAML file is parsed and returned as AppConfig."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("assistant:\n  name: TestBot\n")
        config = load_config(config_file)
        assert config.assistant.name == "TestBot"

    def test_load_config_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Missing config file raises FileNotFoundError (line 175)."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config(tmp_path / "nonexistent.yml")

    def test_load_config_invalid_yaml_raises(self, tmp_path: Path) -> None:
        """Invalid YAML that fails Pydantic validation raises an error (line 178 validation)."""
        config_file = tmp_path / "config.yml"
        # Write YAML that is syntactically valid but fails model validation
        config_file.write_text("logging:\n  level: INVALID_LEVEL\n")
        with pytest.raises(Exception):
            load_config(config_file)

    def test_load_config_returns_defaults_for_empty_yaml(self, tmp_path: Path) -> None:
        """An empty (but existing) YAML file returns a default AppConfig (line 177 empty dict)."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("")
        config = load_config(config_file)
        assert isinstance(config, AppConfig)
        assert config.assistant.name == "Andy"

    def test_load_config_timezone_mapping(self, tmp_path: Path) -> None:
        """Timezone field is read correctly from YAML."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("timezone: Europe/London\n")
        config = load_config(config_file)
        assert config.timezone == "Europe/London"

    def test_load_config_nested_keys(self, tmp_path: Path) -> None:
        """Nested YAML keys map to nested Pydantic fields."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            "timing:\n  poll_interval_ms: 5000\n"
            "container:\n  max_concurrent: 10\n"
        )
        config = load_config(config_file)
        assert config.timing.poll_interval_ms == 5000
        assert config.container.max_concurrent == 10

    def test_load_config_default_path_resolves_cwd(self, tmp_path: Path, monkeypatch) -> None:
        """When config_path is None, defaults to 'config.yml' in cwd (line 173)."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "config.yml"
        config_file.write_text("assistant:\n  name: CwdBot\n")
        config = load_config(None)
        assert config.assistant.name == "CwdBot"


# ---------------------------------------------------------------------------
# Tests: load_credentials()
# ---------------------------------------------------------------------------


class TestLoadCredentials:
    """Tests for the load_credentials() function."""

    def test_load_credentials_with_valid_yaml(self, tmp_path: Path) -> None:
        """Valid credentials YAML is parsed into a Credentials object (lines 195-197)."""
        creds_file = tmp_path / "credentials.yml"
        creds_file.write_text("anthropic_api_key: sk-test-key\n")
        creds = load_credentials(creds_file)
        assert creds.anthropic_api_key == "sk-test-key"

    def test_load_credentials_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """Missing credentials file returns empty Credentials (lines 192-194)."""
        creds = load_credentials(tmp_path / "nonexistent.yml")
        assert isinstance(creds, Credentials)
        assert creds.anthropic_api_key == ""
        assert creds.claude_code_oauth_token == ""

    def test_load_credentials_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        """An empty credentials file returns defaults without error (line 196 empty dict)."""
        creds_file = tmp_path / "credentials.yml"
        creds_file.write_text("")
        creds = load_credentials(creds_file)
        assert isinstance(creds, Credentials)
        assert creds.anthropic_api_key == ""

    def test_load_credentials_both_keys(self, tmp_path: Path) -> None:
        """Both API key and OAuth token are loaded when present."""
        creds_file = tmp_path / "credentials.yml"
        creds_file.write_text(
            "anthropic_api_key: key-abc\n"
            "claude_code_oauth_token: token-xyz\n"
        )
        creds = load_credentials(creds_file)
        assert creds.anthropic_api_key == "key-abc"
        assert creds.claude_code_oauth_token == "token-xyz"

    def test_load_credentials_default_path_resolves_cwd(self, tmp_path: Path, monkeypatch) -> None:
        """When credentials_path is None, defaults to 'credentials.yml' in cwd (line 192)."""
        monkeypatch.chdir(tmp_path)
        creds_file = tmp_path / "credentials.yml"
        creds_file.write_text("anthropic_api_key: cwd-key\n")
        creds = load_credentials(None)
        assert creds.anthropic_api_key == "cwd-key"


# ---------------------------------------------------------------------------
# Tests: LoggingConfig.validate_level
# ---------------------------------------------------------------------------


class TestLoggingConfigValidateLevel:
    """Tests for the LoggingConfig.validate_level field validator (lines 116-119)."""

    def test_valid_level_debug(self) -> None:
        """DEBUG is accepted and normalised to uppercase."""
        cfg = LoggingConfig(level="debug")
        assert cfg.level == "DEBUG"

    def test_valid_level_info(self) -> None:
        """INFO is accepted."""
        cfg = LoggingConfig(level="INFO")
        assert cfg.level == "INFO"

    def test_valid_level_warning(self) -> None:
        """WARNING is accepted."""
        cfg = LoggingConfig(level="Warning")
        assert cfg.level == "WARNING"

    def test_valid_level_error(self) -> None:
        """ERROR is accepted."""
        cfg = LoggingConfig(level="ERROR")
        assert cfg.level == "ERROR"

    def test_valid_level_critical(self) -> None:
        """CRITICAL is accepted."""
        cfg = LoggingConfig(level="CRITICAL")
        assert cfg.level == "CRITICAL"

    def test_invalid_level_raises_value_error(self) -> None:
        """An unrecognised level raises a ValueError (lines 117-118)."""
        with pytest.raises(Exception, match="Invalid log level"):
            LoggingConfig(level="VERBOSE")

    def test_invalid_level_nonsense_raises(self) -> None:
        """Any non-standard string is rejected (lines 117-118)."""
        with pytest.raises(Exception):
            LoggingConfig(level="TRACE")


# ---------------------------------------------------------------------------
# Tests: TimingConfig properties
# ---------------------------------------------------------------------------


class TestTimingConfigProperties:
    """Tests for the TimingConfig ms -> seconds property conversions (lines 48, 53, 58, 63)."""

    def test_poll_interval_s(self) -> None:
        """poll_interval_s converts poll_interval_ms to seconds (line 48)."""
        cfg = TimingConfig(poll_interval_ms=2000)
        assert cfg.poll_interval_s == pytest.approx(2.0)

    def test_scheduler_poll_interval_s(self) -> None:
        """scheduler_poll_interval_s converts scheduler_poll_interval_ms (line 53)."""
        cfg = TimingConfig(scheduler_poll_interval_ms=60000)
        assert cfg.scheduler_poll_interval_s == pytest.approx(60.0)

    def test_idle_timeout_s(self) -> None:
        """idle_timeout_s converts idle_timeout_ms to seconds (line 58)."""
        cfg = TimingConfig(idle_timeout_ms=1800000)
        assert cfg.idle_timeout_s == pytest.approx(1800.0)

    def test_ipc_poll_interval_s(self) -> None:
        """ipc_poll_interval_s converts ipc_poll_interval_ms to seconds (line 63)."""
        cfg = TimingConfig(ipc_poll_interval_ms=1000)
        assert cfg.ipc_poll_interval_s == pytest.approx(1.0)

    def test_defaults(self) -> None:
        """TimingConfig defaults match expected ms values."""
        cfg = TimingConfig()
        assert cfg.poll_interval_ms == 2000
        assert cfg.scheduler_poll_interval_ms == 60000
        assert cfg.idle_timeout_ms == 1800000
        assert cfg.ipc_poll_interval_ms == 1000


# ---------------------------------------------------------------------------
# Tests: ContainerConfig properties
# ---------------------------------------------------------------------------


class TestContainerConfigProperties:
    """Tests for ContainerConfig defaults and timeout_s property (line 84)."""

    def test_timeout_s(self) -> None:
        """timeout_s converts timeout_ms to seconds (line 84)."""
        cfg = ContainerConfig(timeout_ms=30000)
        assert cfg.timeout_s == pytest.approx(30.0)

    def test_defaults(self) -> None:
        """ContainerConfig defaults match expected values."""
        cfg = ContainerConfig()
        assert cfg.image == "nanoclaw-agent:latest"
        assert cfg.timeout_ms == 1800000
        assert cfg.max_output_size_bytes == 10485760
        assert cfg.max_concurrent == 5


# ---------------------------------------------------------------------------
# Tests: AppConfig.trigger_pattern
# ---------------------------------------------------------------------------


class TestAppConfigTriggerPattern:
    """Tests for the AppConfig.trigger_pattern computed property (lines 144-145)."""

    def test_trigger_pattern_matches_at_mention(self) -> None:
        """trigger_pattern matches '@Andy' at the start of a string (line 144-145)."""
        config = AppConfig(assistant=AssistantConfig(name="Andy"))
        assert config.trigger_pattern.match("@Andy hello")

    def test_trigger_pattern_is_case_insensitive(self) -> None:
        """trigger_pattern is case-insensitive."""
        config = AppConfig(assistant=AssistantConfig(name="Andy"))
        assert config.trigger_pattern.match("@andy hello")

    def test_trigger_pattern_does_not_match_mid_string(self) -> None:
        """trigger_pattern is anchored to the start of the string."""
        config = AppConfig(assistant=AssistantConfig(name="Andy"))
        assert not config.trigger_pattern.match("hello @Andy")

    def test_trigger_pattern_uses_assistant_name(self) -> None:
        """trigger_pattern uses the configured assistant name."""
        config = AppConfig(assistant=AssistantConfig(name="Zara"))
        assert config.trigger_pattern.match("@Zara help")
        assert not config.trigger_pattern.match("@Andy help")

    def test_trigger_pattern_escapes_special_chars(self) -> None:
        """Special regex characters in the name are properly escaped."""
        config = AppConfig(assistant=AssistantConfig(name="Bot.AI"))
        # The dot should be treated literally, not as a regex wildcard
        assert config.trigger_pattern.match("@Bot.AI help")


# ---------------------------------------------------------------------------
# Tests: PathsConfig defaults
# ---------------------------------------------------------------------------


class TestPathsConfig:
    """Tests for PathsConfig defaults."""

    def test_defaults(self) -> None:
        """PathsConfig has the expected default directory names."""
        cfg = PathsConfig()
        assert cfg.store_dir == "store"
        assert cfg.groups_dir == "groups"
        assert cfg.data_dir == "data"


# ---------------------------------------------------------------------------
# Tests: AssistantConfig defaults
# ---------------------------------------------------------------------------


class TestAssistantConfig:
    """Tests for AssistantConfig defaults."""

    def test_defaults(self) -> None:
        """AssistantConfig has the expected default values."""
        cfg = AssistantConfig()
        assert cfg.name == "Andy"
        assert cfg.has_own_number is False

    def test_custom_values(self) -> None:
        """AssistantConfig accepts custom name and has_own_number."""
        cfg = AssistantConfig(name="Jarvis", has_own_number=True)
        assert cfg.name == "Jarvis"
        assert cfg.has_own_number is True


# end tests/test_config.py
