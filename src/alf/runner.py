"""对话入口: 管理历史 + 调用 graph."""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque

from .config import settings
from .graph import app_graph
from .graph.nodes import analyze_intent, maybe_write_memory, retrieve_memories, stream_reply

logger = logging.getLogger(__name__)

# 简单的内存内会话历史. 多用户/持久化可换 Redis / DB.
_histories: dict[str, Deque[dict[str, str]]] = {}
MAX_HISTORY = 20


def _get_history(user_id: str) -> Deque[dict[str, str]]:
    if user_id not in _histories:
        _histories[user_id] = deque(maxlen=MAX_HISTORY)
    return _histories[user_id]


def chat(user_message: str, user_id: str | None = None) -> str:
    """单轮对话: 输入用户消息, 返回小奥的回复."""
    uid = user_id or settings.alf_user_id
    history = list(_get_history(uid))

    state = {
        "user_id": uid,
        "user_message": user_message,
        "messages": history,
    }

    result = app_graph.invoke(state)
    reply = result.get("reply", "")

    # 更新历史
    _get_history(uid).append({"role": "user", "content": user_message})
    _get_history(uid).append({"role": "assistant", "content": reply})

    return reply


def reset(user_id: str | None = None) -> None:
    uid = user_id or settings.alf_user_id
    _histories.pop(uid, None)


def stream(user_message: str, user_id: str | None = None):
    """流式输出回复文本，并在完成后更新会话与长期记忆。"""
    uid = user_id or settings.alf_user_id
    history = list(_get_history(uid))
    state = {"user_id": uid, "user_message": user_message, "messages": history}
    state.update(retrieve_memories(state))
    state.update(analyze_intent(state))

    reply_parts: list[str] = []
    for text in stream_reply(state):
        reply_parts.append(text)
        yield text

    reply = "".join(reply_parts).strip()
    state["reply"] = reply
    maybe_write_memory(state)
    _get_history(uid).append({"role": "user", "content": user_message})
    _get_history(uid).append({"role": "assistant", "content": reply})
