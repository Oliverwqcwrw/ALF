"""对话入口: 管理持久化历史 + 情绪状态 + 调用 graph."""
from __future__ import annotations

import logging

from . import store
from .config import settings
from .graph import app_graph
from .graph.nodes import (
    analyze_intent,
    maybe_write_memory,
    retrieve_memories,
    stream_reply,
    update_impression,
)

logger = logging.getLogger(__name__)

# 进入本轮前连续低落达到该轮数 → 触发主动关怀.
PROACTIVE_CARE_THRESHOLD = 2


def _load_state(user_id: str, user_message: str) -> dict:
    history = store.get_history(user_id)
    emotion_history = store.get_emotion_history(user_id)
    consec_before = store.consecutive_low_count(user_id)
    impression = store.get_impression(user_id)
    recent_events = store.get_recent_emotion_events(user_id)
    return {
        "user_id": user_id,
        "user_message": user_message,
        "messages": history,
        "emotion_history": emotion_history,
        "proactively_care": consec_before >= PROACTIVE_CARE_THRESHOLD,
        "impression": impression,
        "recent_events": recent_events,
    }


def _persist(user_id: str, user_message: str, reply: str, emotion: str) -> None:
    store.append_message(user_id, "user", user_message)
    store.append_message(user_id, "assistant", reply)
    store.append_emotion(user_id, emotion)


def _persist_impression(user_id: str, result: dict) -> None:
    """若 update_impression 节点改写了画像, 写回 SQLite."""
    new = result.get("impression")
    if new:
        store.set_impression(user_id, new)


def chat(user_message: str, user_id: str | None = None) -> str:
    """单轮对话: 输入用户消息, 返回小奥的回复."""
    uid = user_id or settings.alf_user_id
    state = _load_state(uid, user_message)

    result = app_graph.invoke(state)
    reply = result.get("reply", "")
    emotion = result.get("intent", {}).get("emotion", "neutral")

    _persist(uid, user_message, reply, emotion)
    _persist_impression(uid, result)
    return reply


def reset(user_id: str | None = None) -> None:
    uid = user_id or settings.alf_user_id
    store.clear_history(uid)


def stream(user_message: str, user_id: str | None = None):
    """流式输出回复文本, 并在完成后更新会话与长期记忆.

    流式路径跳过 self_check (无法中途重生成); 路由/共情/主动关怀仍生效.
    """
    uid = user_id or settings.alf_user_id
    state = _load_state(uid, user_message)
    state.update(retrieve_memories(state))
    state.update(analyze_intent(state))

    reply_parts: list[str] = []
    for text in stream_reply(state):
        reply_parts.append(text)
        yield text

    reply = "".join(reply_parts).strip()
    state["reply"] = reply
    maybe_write_memory(state)
    state.update(update_impression(state))
    emotion = state.get("intent", {}).get("emotion", "neutral")
    _persist(uid, user_message, reply, emotion)
    _persist_impression(uid, state)
