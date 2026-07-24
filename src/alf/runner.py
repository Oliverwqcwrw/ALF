"""对话入口: 管理持久化历史 + 情绪状态 + 调用 graph."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import store
from .memory import store as memory_store
from .config import settings
from .graph import app_graph
from .graph.nodes import (
    analyze_intent,
    format_emotion_events,
    maybe_write_memory,
    retrieve_memories,
    stream_reply,
    update_impression,
)
from .persona import PROACTIVE_MESSAGE_PROMPT

logger = logging.getLogger(__name__)

# 进入本轮前连续低落达到该轮数 → 触发主动关怀.
PROACTIVE_CARE_THRESHOLD = 2
# 长期记忆能让回复更贴近用户，但不能长期阻塞首个 token。意图/危机分析仍必须完成。
STREAM_MEMORY_WAIT_SECONDS = 0.35

_proactive_llm: ChatOpenAI | None = None


def _get_proactive_llm() -> ChatOpenAI:
    global _proactive_llm
    if _proactive_llm is None:
        _proactive_llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.85,
        )
    return _proactive_llm


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


def _prepare_stream_state(state: dict) -> dict:
    """并行完成流式回复前的安全分析与记忆检索。

    危机/情绪分析决定回复策略，必须在生成前完成；长期记忆则设很短的
    等待预算，慢查询不应让用户一直等到第一个 token。
    """
    started_at = perf_counter()
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="alf-preflight")
    memory_future = executor.submit(retrieve_memories, state)
    intent_future = executor.submit(analyze_intent, state)

    try:
        # 这一步保留同步等待，避免危机消息在未经安全路由时开始生成。
        state.update(intent_future.result())
    except Exception:  # noqa: BLE001
        logger.exception("stream intent analysis failed; using safe normal fallback")
        state.update({"intent": {"emotion": "neutral", "is_crisis": False}, "route": "normal"})

    try:
        state.update(memory_future.result(timeout=STREAM_MEMORY_WAIT_SECONDS))
    except TimeoutError:
        # 检索会继续在后台收尾，但本轮不再等待，避免把“记得”变成“迟到”。
        state["memories"] = []
        logger.info("stream memory retrieval exceeded %.0fms; continuing without it", STREAM_MEMORY_WAIT_SECONDS * 1000)
    except Exception:  # noqa: BLE001
        state["memories"] = []
        logger.exception("stream memory retrieval failed; continuing without it")
    finally:
        # 不等待超时的检索任务，已开始的请求自然结束；尚未开始的可取消。
        executor.shutdown(wait=False, cancel_futures=True)

    logger.info("stream preflight completed in %.0fms", (perf_counter() - started_at) * 1000)
    return state


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


def get_memories(user_id: str | None = None) -> list[dict]:
    """返回小奥为该用户保存的长期记忆，供用户查看与管理。"""
    uid = user_id or settings.alf_user_id
    return memory_store.get_all_memories(uid)


def forget_memory(memory_id: str, user_id: str | None = None) -> bool:
    """仅删除属于该用户的一条长期记忆。"""
    uid = user_id or settings.alf_user_id
    memories = memory_store.get_all_memories(uid)
    if not any(str(memory.get("id")) == memory_id for memory in memories):
        return False
    memory_store.delete_memory(memory_id)
    return True


def forget_everything(user_id: str | None = None) -> None:
    """清除小奥为用户保留的全部本地状态和长期记忆。"""
    uid = user_id or settings.alf_user_id
    for memory in memory_store.get_all_memories(uid):
        memory_id = memory.get("id")
        if memory_id:
            memory_store.delete_memory(str(memory_id))
    store.clear_history(uid)
    store.clear_personal_context(uid)


def generate_proactive(user_id: str, reason: str) -> str | None:
    """生成一条主动开口消息 (深夜/久未对话触发). 不走完整 graph.

    没有用户消息, 不经 analyze_intent/self_check; 只加载印象/近期事件,
    用主动开口 prompt + 主 LLM 生成一条朋友式主动消息.
    """
    uid = user_id or settings.alf_user_id
    impression = store.get_impression(uid)
    recent_events = store.get_recent_emotion_events(uid)
    history = store.get_history(uid)

    system_prompt = PROACTIVE_MESSAGE_PROMPT.format(
        alf_agent_name=settings.alf_agent_name,
        alf_user_id=settings.alf_user_id,
        reason=reason,
        impression=impression or "(还没印象)",
        recent_events=format_emotion_events(recent_events),
    )

    lc_messages: list = [SystemMessage(content=system_prompt)]
    # 附最近几轮对话, 让主动消息能呼应近期上下文.
    for m in history[-4:]:
        if m.get("role") == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m.get("role") == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    try:
        from .graph.nodes import _content_to_text

        resp = _get_proactive_llm().invoke(lc_messages)
        text = _content_to_text(resp.content).strip()
        return text or None
    except Exception as e:  # noqa: BLE001
        logger.warning("generate_proactive failed: %s", e)
        return None


def stream(user_message: str, user_id: str | None = None):
    """流式输出回复文本, 并在完成后更新会话与长期记忆.

    流式路径跳过 self_check (无法中途重生成); 路由/共情/主动关怀仍生效.
    """
    uid = user_id or settings.alf_user_id
    state = _load_state(uid, user_message)
    _prepare_stream_state(state)

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
