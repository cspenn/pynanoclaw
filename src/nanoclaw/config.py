# start src/nanoclaw/config.py
"""Configuration loader for NanoClaw.

Loads config.yml and credentials.yml at startup and validates them into
Pydantic models. Any missing required field raises at boot, not at runtime.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class AssistantConfig(BaseModel):
    """Configuration for the assistant personality.

    Attributes:
        name: The assistant's name used in trigger patterns and message prefixes.
        has_own_number: True if the bot has its own dedicated phone number.
            When False, messages are prefixed with the assistant name.
    """

    name: str = "Andy"
    has_own_number: bool = False


class TimingConfig(BaseModel):
    """Timing intervals in milliseconds.

    Attributes:
        poll_interval_ms: How often to check for new messages.
        scheduler_poll_interval_ms: How often to check for due scheduled tasks.
        idle_timeout_ms: How long to keep a container alive after its last output.
        ipc_poll_interval_ms: How often to scan IPC directories for new files.
    """

    poll_interval_ms: int = 2000
    scheduler_poll_interval_ms: int = 60000
    idle_timeout_ms: int = 1800000
    ipc_poll_interval_ms: int = 1000

    @property
    def poll_interval_s(self) -> float:
        """Poll interval in seconds."""
        return self.poll_interval_ms / 1000.0

    @property
    def scheduler_poll_interval_s(self) -> float:
        """Scheduler poll interval in seconds."""
        return self.scheduler_poll_interval_ms / 1000.0

    @property
    def idle_timeout_s(self) -> float:
        """Idle timeout in seconds."""
        return self.idle_timeout_ms / 1000.0

    @property
    def ipc_poll_interval_s(self) -> float:
        """IPC poll interval in seconds."""
        return self.ipc_poll_interval_ms / 1000.0


class ContainerConfig(BaseModel):
    """Container runtime configuration.

    Attributes:
        image: Docker/Apple Container image name.
        timeout_ms: Hard timeout for container execution in milliseconds.
        max_output_size_bytes: Maximum bytes to buffer from container stdout/stderr.
        max_concurrent: Maximum number of simultaneously running containers.
    """

    image: str = "nanoclaw-agent:latest"
    timeout_ms: int = 1800000
    max_output_size_bytes: int = 10485760
    max_concurrent: int = 5

    @property
    def timeout_s(self) -> float:
        """Timeout in seconds."""
        return self.timeout_ms / 1000.0


class PathsConfig(BaseModel):
    """Filesystem path configuration (relative to project root).

    Attributes:
        store_dir: Directory for the SQLite database and WhatsApp auth state.
        groups_dir: Directory for per-group workspaces.
        data_dir: Directory for IPC files and session state.
    """

    store_dir: str = "store"
    groups_dir: str = "groups"
    data_dir: str = "data"


class LoggingConfig(BaseModel):
    """Logging configuration.

    Attributes:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        file: Path to the log file (relative to project root).
    """

    level: str = "INFO"
    file: str = "logs/nanoclaw.log"

    @field_validator("level")
    @classmethod
    def validate_level(_cls, v: str) -> str:  # noqa: N804
        """Validate log level is a known value."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"Invalid log level '{v}'. Must be one of: {valid}")
        return v.upper()


class AppConfig(BaseModel):
    """Root application configuration loaded from config.yml.

    Attributes:
        assistant: Assistant personality settings.
        timing: Timing interval settings.
        container: Container runtime settings.
        paths: Filesystem path settings.
        logging: Logging settings.
        timezone: Timezone string for cron scheduling (e.g., 'America/New_York').
    """

    assistant: AssistantConfig = Field(default_factory=AssistantConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    container: ContainerConfig = Field(default_factory=ContainerConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    timezone: str = "America/New_York"

    @property
    def trigger_pattern(self) -> re.Pattern[str]:
        """Compiled regex for matching trigger mentions."""
        escaped = re.escape(self.assistant.name)
        return re.compile(rf"^@{escaped}\b", re.IGNORECASE)


class Credentials(BaseModel):
    """Secret credentials loaded from credentials.yml (gitignored).

    Attributes:
        anthropic_api_key: Anthropic API key for claude-agent-sdk.
        claude_code_oauth_token: OAuth token for Claude Code agent SDK.
    """

    anthropic_api_key: str = ""
    claude_code_oauth_token: str = ""


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load and validate application configuration from config.yml.

    Args:
        config_path: Path to config.yml. Defaults to config.yml in the current directory.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If config.yml does not exist.
        ValueError: If config values fail Pydantic validation.
    """
    path = config_path or Path("config.yml")
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return AppConfig.model_validate(data)


def load_credentials(credentials_path: Path | None = None) -> Credentials:
    """Load secrets from credentials.yml (gitignored).

    Args:
        credentials_path: Path to credentials.yml. Defaults to credentials.yml
            in the current directory.

    Returns:
        Credentials instance. Returns empty credentials if file does not exist
        (allows running without secrets for testing).
    """
    path = credentials_path or Path("credentials.yml")
    if not path.exists():
        return Credentials()
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return Credentials.model_validate(data)


# end src/nanoclaw/config.py
