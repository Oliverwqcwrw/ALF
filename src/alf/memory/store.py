"""mem0 记忆层封装.

提供:
- 单例 Memory 客户端 (避免反复初始化)
- 按 user_id 隔离的增删查
- 便于 LLM 注入的 search() -> 文本格式

mem0 会自动从对话中抽取事实型记忆, 这里在之上做语义增强:
- 每条消息打上 metadata.category: emotion | preference | event | fact
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from mem0 import Memory

from ..config import settings
from ..config.mem0_config import build_mem0_config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_memory_client() -> Memory:
    """单例 Memory 客户端."""
    cfg = build_mem0_config()
    return Memory.from_config(cfg)


def add_message(
    messages: list[dict[str, str]],
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """把一段对话喂给 mem0 抽取记忆.

    messages 形如 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    uid = user_id or settings.alf_user_id
    client = get_memory_client()
    result = client.add(messages, user_id=uid, metadata=metadata or {})
    logger.debug("mem0 add result: %s", result)
    return result


def search_memory(
    query: str,
    user_id: str | None = None,
    top_k: int = 6,
) -> list[dict[str, Any]]:
    """语义检索相关记忆."""
    uid = user_id or settings.alf_user_id
    client = get_memory_client()
    results = client.search(query=query, user_id=uid, limit=top_k)
    return list(results)


def get_all_memories(user_id: str | None = None) -> list[dict[str, Any]]:
    uid = user_id or settings.alf_user_id
    client = get_memory_client()
    return list(client.get_all(user_id=uid))


def delete_memory(memory_id: str) -> None:
    client = get_memory_client()
    client.delete(memory_id)


def format_memories(memories: list[dict[str, Any]]) -> str:
    """把检索到的记忆格式化为可注入 prompt 的文本."""
    if not memories:
        return "(暂无相关记忆)"
    lines: list[str] = []
    for m in memories:
        text = m.get("memory") or m.get("text") or ""
        if not text:
            continue
        meta = m.get("metadata") or {}
        category = meta.get("category", "fact")
        lines.append(f"- [{category}] {text}")
    return "\n".join(lines) if lines else "(暂无相关记忆)"
