"""全局配置, 从环境变量读取."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o-mini"
    chat_model_mini: str = "gpt-4o-mini"

    # mem0
    mem0_store: str = "local"  # local | qdrant | chroma
    mem0_data_dir: str = "./mem0_data"
    mem0_embedder: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    # ALF
    alf_user_id: str = "oliver"
    alf_agent_name: str = "小奥"

    @property
    def mem0_data_path(self) -> Path:
        p = Path(self.mem0_data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
