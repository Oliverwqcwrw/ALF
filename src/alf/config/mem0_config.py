"""mem0 配置生成.

提供两种后端:
- local: 默认, 使用本地 sqlite (sqlite_vec) + 本地 embedding
- qdrant / chroma: 可选, 需自行启动服务

按 user_id 隔离记忆空间, 实现"私人"陪伴.
"""
from __future__ import annotations

from typing import Any

from .settings import settings


def build_mem0_config() -> dict[str, Any]:
    """构建 mem0 的配置字典."""
    if settings.mem0_store == "local":
        cfg: dict[str, Any] = {
            "vector_store": {
                "provider": "sqlite_vec",
                "config": {
                    "db_file": str(settings.mem0_data_path / "mem0.db"),
                    "table_name": "memories",
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": settings.openai_api_key,
                    "base_url": settings.openai_base_url,
                    "model": settings.chat_model,
                },
            },
            "embedder": {
                "provider": settings.mem0_embedder,
                "config": {
                    "api_key": settings.openai_api_key,
                    "base_url": settings.openai_base_url,
                    "model": settings.embedding_model,
                },
            },
        }
        return cfg

    if settings.mem0_store == "qdrant":
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {"host": "localhost", "port": 6333},
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": settings.openai_api_key,
                    "base_url": settings.openai_base_url,
                    "model": settings.chat_model,
                },
            },
            "embedder": {
                "provider": settings.mem0_embedder,
                "config": {
                    "api_key": settings.openai_api_key,
                    "model": settings.embedding_model,
                },
            },
        }

    raise ValueError(f"未支持的 mem0_store: {settings.mem0_store}")
