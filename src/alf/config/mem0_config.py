"""mem0 配置生成.

支持的后端:
- local: 默认, 使用本地 ChromaDB (纯 Python, 文件存储, 零外部服务)
- qdrant: 可选, 需自行启动 Qdrant 服务

按 user_id 隔离记忆空间, 实现"私人"陪伴.
"""
from __future__ import annotations

from typing import Any

from .settings import settings


def _llm_config() -> dict[str, Any]:
    return {
        "provider": "openai",
        "config": {
            "api_key": settings.openai_api_key,
            "openai_base_url": settings.openai_base_url,
            "model": settings.chat_model,
        },
    }


def _embedder_config() -> dict[str, Any]:
    return {
        "provider": settings.mem0_embedder,
        "config": {
            "api_key": settings.openai_api_key,
            "openai_base_url": settings.openai_base_url,
            "model": settings.embedding_model,
        },
    }


def build_mem0_config() -> dict[str, Any]:
    """构建 mem0 的配置字典."""
    if settings.mem0_store == "local":
        return {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "alf_memories",
                    "path": str(settings.mem0_data_path / "chroma"),
                },
            },
            "llm": _llm_config(),
            "embedder": _embedder_config(),
            "history_db_path": str(settings.mem0_data_path / "history.db"),
        }

    if settings.mem0_store == "qdrant":
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {"host": "localhost", "port": 6333},
            },
            "llm": _llm_config(),
            "embedder": _embedder_config(),
            "history_db_path": str(settings.mem0_data_path / "history.db"),
        }

    raise ValueError(f"未支持的 mem0_store: {settings.mem0_store}")
