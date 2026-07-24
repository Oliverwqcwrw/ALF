"""对话状态定义."""
from __future__ import annotations

from typing import Any, TypedDict


class ConversationState(TypedDict, total=False):
    # 输入
    user_id: str
    user_message: str
    messages: list[dict[str, str]]  # 近期对话历史 [{role, content}, ...]

    # 跨轮情绪状态 (runner 注入并持久化)
    emotion_history: list[str]  # 最近 N 轮情绪标签
    consecutive_low: int  # 连续低落轮数 (sad/anxious/angry)

    # 流程中产生
    memories: list[dict[str, Any]]  # 检索到的相关记忆
    intent: dict[str, Any]  # {emotion, topic, is_significant, is_crisis}
    impression: str  # 小奥对该用户的整体印象画像 (runner 注入并持久化)
    recent_events: list[dict[str, Any]]  # 近期情绪-事件 (runner 注入, 时序关联)

    # 路由决策
    route: str  # crisis | empathize | normal
    proactively_care: bool  # 连续低落, 本轮主动关怀

    # 输出
    reply: str
    self_check_passed: bool
    check_reason: str  # 自检未通过的原因, 重生成时注入
    retry_count: int
