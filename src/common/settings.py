"""Typed configuration over .env secrets and versioned YAML decisions."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class Settings(BaseSettings):
    """Environment-backed settings, one field per .env key."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: str = ""
    gemini_api_key: str = ""
    llm_primary_model: str = "gpt-4o-mini"
    llm_fallback_model: str = "gemini/gemini-2.0-flash"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1
    llm_timeout_seconds: int = 20

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    mlflow_tracking_uri: str = "./mlflow"
    mlflow_experiment_name: str = "claim-approval-agent"

    data_path: Path = Field(default=Path("./data/claim_use_case_dataset.xlsx"))
    model_path: Path = Field(default=Path("./artifacts/models/"))
    prompt_path: Path = Field(default=Path("./prompts/"))

    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Empty disables auth, keeping the Postman demo zero-setup; production sets it.
    api_key: str = ""

    @property
    def llm_mock_mode(self) -> bool:
        """True when no provider key is configured."""
        # Offline mode keeps CI deterministic and free of provider cost.
        return not (self.openai_api_key or self.gemini_api_key)

    @property
    def langfuse_enabled(self) -> bool:
        """True when both Langfuse keys are present."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def auth_enabled(self) -> bool:
        """True when an API key is configured."""
        return bool(self.api_key)

    def resolve(self, path: Path) -> Path:
        """Make a configured relative path absolute against the project root."""
        p = Path(path)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


@functools.lru_cache(maxsize=8)
def load_yaml_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml."""
    # Cached: these files hold decisions, not runtime state.
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_training_config() -> dict[str, Any]:
    """Training pipeline configuration."""
    return load_yaml_config("training")


def get_llm_config() -> dict[str, Any]:
    """GenAI and provider configuration."""
    return load_yaml_config("llm")


def get_api_config() -> dict[str, Any]:
    """API and human-in-the-loop configuration."""
    return load_yaml_config("api")
