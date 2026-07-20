"""mem0 记忆层封装.

提供:
- 单例 Memory 客户端 (避免反复初始化)
- 按 user_id 隔离的增删查
- 便于 LLM 注入的 search() -> 文本格式

mem0 会自动从对话中抽取事实型记忆, 这里在之上做语义增强:
- 每条消息打上 metadata.category: emotion | preference | event | fact
"""
from __future__ import annotations

import contextlib
import io
import logging
import warnings
from functools import lru_cache
from typing import Any

from mem0 import Memory

from ..config import settings
from ..config.mem0_config import build_mem0_config

logger = logging.getLogger(__name__)


def _quiet_mem0_loggers() -> None:
    """关闭 mem0 依赖的遥测与可选功能日志，不影响 ALF 自己的日志。"""
    # PostHog 会在创建 Client 时重置自己的日志级别；disabled 不会被重置，
    # 且能阻止后台上传线程在重定向结束后继续往终端打印。
    logging.getLogger("posthog").disabled = True
    logging.getLogger("posthog.consumer").disabled = True
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    logging.getLogger("mem0").setLevel(logging.ERROR)


def _memory_records(result: Any) -> list[dict[str, Any]]:
    """兼容 mem0 早期列表返回与 2.x 的 {"results": [...]} 返回."""
    if isinstance(result, dict):
        result = result.get("results", [])
    return [item for item in result if isinstance(item, dict)]


@lru_cache(maxsize=1)
def get_memory_client() -> Memory:
    """创建单例 mem0 客户端，不将第三方初始化提示输出到 CLI。"""
    cfg = build_mem0_config()
    # mem0 / Chroma / PostHog 会在初始化阶段直接向 stdout、stderr 打印
    # 可选功能和遥测提示。这些不是用户可操作的对话内容，静默即可；真正的
    # 初始化异常仍会照常抛出并由 CLI 展示。
    with (
        warnings.catch_warnings(),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        warnings.simplefilter("ignore")
        client = Memory.from_config(cfg)
    _quiet_mem0_loggers()
    return client


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
    results = client.search(query=query, filters={"user_id": uid}, limit=top_k)
    # mem0 OSS 2.x 返回 {"results": [...]}; 早期版本则直接返回列表。
    # 对字典直接调用 list() 只会得到 "results" 等键名，后续格式化会把
    # 这些字符串当作记忆对象，从而触发 "str has no attribute get"。
    return _memory_records(results)


def get_all_memories(user_id: str | None = None) -> list[dict[str, Any]]:
    uid = user_id or settings.alf_user_id
    results = get_memory_client().get_all(filters={"user_id": uid})
    return _memory_records(results)


def delete_memory(memory_id: str) -> None:
    client = get_memory_client()
    client.delete(memory_id)


def format_memories(memories: list[dict[str, Any]]) -> str:
    """把检索到的记忆格式化为可注入 prompt 的文本."""
    if not memories:
        return "(暂无相关记忆)"
    lines: list[str] = []
    for m in memories:
        if not isinstance(m, dict):
            logger.warning("skip malformed memory record: %r", m)
            continue
        text = m.get("memory") or m.get("text") or ""
        if not text:
            continue
        meta = m.get("metadata") or {}
        category = meta.get("category", "fact")
        lines.append(f"- [{category}] {text}")
    return "\n".join(lines) if lines else "(暂无相关记忆)"
