import os
from functools import lru_cache
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        """Centralized Configuration using Pydantic Settings."""
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        APP_NAME: str = "Enterprise Foundation Model Platform"
        APP_VERSION: str = "1.0.0"
        ENVIRONMENT: str = "development"
        LOG_LEVEL: str = "INFO"
        APP_HOST: str = "0.0.0.0"
        APP_PORT: int = 8000

        ANTHROPIC_API_KEY: Optional[str] = None
        ANTHROPIC_DEFAULT_MODEL: str = "claude-3-5-sonnet-20241022"

        AZURE_OPENAI_API_KEY: Optional[str] = None
        AZURE_OPENAI_ENDPOINT: Optional[str] = None
        AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
        AZURE_OPENAI_CHAT_DEPLOYMENT: str = "gpt-4o"
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-3-small"

        AZURE_SEARCH_ENDPOINT: Optional[str] = None
        AZURE_SEARCH_API_KEY: Optional[str] = None
        AZURE_SEARCH_INDEX_NAME: str = "enterprise-knowledge-base"

        AZURE_CONTENT_SAFETY_ENDPOINT: Optional[str] = None
        AZURE_CONTENT_SAFETY_API_KEY: Optional[str] = None

        DEFAULT_PRIMARY_PROVIDER: str = "anthropic"
        FALLBACK_PROVIDER: str = "azure_openai"
        MAX_RETRIES: int = 3
        REQUEST_TIMEOUT_SECONDS: float = 30.0
        ENABLE_SAFETY_GUARDRAILS: bool = True
        ENABLE_SIMULATION_MODE: bool = False

except ImportError:
    class Settings:
        """Lightweight configuration fallback when pydantic-settings is not installed."""
        def __init__(self, **kwargs):
            self.APP_NAME = kwargs.get("APP_NAME", os.getenv("APP_NAME", "Enterprise Foundation Model Platform"))
            self.APP_VERSION = kwargs.get("APP_VERSION", os.getenv("APP_VERSION", "1.0.0"))
            self.ENVIRONMENT = kwargs.get("ENVIRONMENT", os.getenv("ENVIRONMENT", "development"))
            self.LOG_LEVEL = kwargs.get("LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO"))
            self.APP_HOST = kwargs.get("APP_HOST", os.getenv("APP_HOST", "0.0.0.0"))
            self.APP_PORT = int(kwargs.get("APP_PORT", os.getenv("APP_PORT", 8000)))

            self.ANTHROPIC_API_KEY = kwargs.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
            self.ANTHROPIC_DEFAULT_MODEL = kwargs.get("ANTHROPIC_DEFAULT_MODEL", os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-5-sonnet-20241022"))

            self.AZURE_OPENAI_API_KEY = kwargs.get("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY"))
            self.AZURE_OPENAI_ENDPOINT = kwargs.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT"))
            self.AZURE_OPENAI_API_VERSION = kwargs.get("AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"))
            self.AZURE_OPENAI_CHAT_DEPLOYMENT = kwargs.get("AZURE_OPENAI_CHAT_DEPLOYMENT", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"))
            self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT = kwargs.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"))

            self.AZURE_SEARCH_ENDPOINT = kwargs.get("AZURE_SEARCH_ENDPOINT", os.getenv("AZURE_SEARCH_ENDPOINT"))
            self.AZURE_SEARCH_API_KEY = kwargs.get("AZURE_SEARCH_API_KEY", os.getenv("AZURE_SEARCH_API_KEY"))
            self.AZURE_SEARCH_INDEX_NAME = kwargs.get("AZURE_SEARCH_INDEX_NAME", os.getenv("AZURE_SEARCH_INDEX_NAME", "enterprise-knowledge-base"))

            self.AZURE_CONTENT_SAFETY_ENDPOINT = kwargs.get("AZURE_CONTENT_SAFETY_ENDPOINT", os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT"))
            self.AZURE_CONTENT_SAFETY_API_KEY = kwargs.get("AZURE_CONTENT_SAFETY_API_KEY", os.getenv("AZURE_CONTENT_SAFETY_API_KEY"))

            self.DEFAULT_PRIMARY_PROVIDER = kwargs.get("DEFAULT_PRIMARY_PROVIDER", os.getenv("DEFAULT_PRIMARY_PROVIDER", "anthropic"))
            self.FALLBACK_PROVIDER = kwargs.get("FALLBACK_PROVIDER", os.getenv("FALLBACK_PROVIDER", "azure_openai"))
            self.MAX_RETRIES = int(kwargs.get("MAX_RETRIES", os.getenv("MAX_RETRIES", 3)))
            self.REQUEST_TIMEOUT_SECONDS = float(kwargs.get("REQUEST_TIMEOUT_SECONDS", os.getenv("REQUEST_TIMEOUT_SECONDS", 30.0)))
            self.ENABLE_SAFETY_GUARDRAILS = str(kwargs.get("ENABLE_SAFETY_GUARDRAILS", os.getenv("ENABLE_SAFETY_GUARDRAILS", "true"))).lower() in ("true", "1")
            self.ENABLE_SIMULATION_MODE = str(kwargs.get("ENABLE_SIMULATION_MODE", os.getenv("ENABLE_SIMULATION_MODE", "false"))).lower() in ("true", "1")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
