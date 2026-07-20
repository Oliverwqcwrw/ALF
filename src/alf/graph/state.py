"""对话状态定义."""
from __future__ import annotations

from typing import Any, TypedDict


class ConversationState(TypedDict, total=False):
    # 输入
    user_id: str
    user_message: str

    # 流程中产生
    memories: list[dict[str, Any]]  # 检索到的相关记忆
    intent: dict[str, Any]  # {emotion, topic, is_significant}

    # 对话历史 (含本轮回复)
    messages: list[dict[str, str]]  # [{role, content}, ...]

    # 输出
    reply: str
