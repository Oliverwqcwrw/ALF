"""langgraph 节点: 检索记忆 / 分析意图 / 生成回复 / 写入记忆."""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import settings
from ..memory import format_memories, search_memory
from ..persona import PERSONA_SYSTEM
from ..persona.analyzer import analyzer
from .state import ConversationState

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.85,  # 陪伴对话需要一点温度
        )
    return _llm


def retrieve_memories(state: ConversationState) -> ConversationState:
    """根据用户当前消息检索相关记忆."""
    msg = state["user_message"]
    memories = search_memory(query=msg, user_id=state["user_id"], top_k=6)
    return {"memories": memories}


def analyze_intent(state: ConversationState) -> ConversationState:
    """轻量分析用户消息: 情绪/主题/是否值得记忆."""
    intent = analyzer.extract_intent(state["user_message"])
    logger.info("intent: %s", intent)
    return {"intent": intent}


def generate_reply(state: ConversationState) -> ConversationState:
    """调用 LLM 生成回复, 注入人格 + 历史 + 记忆."""
    llm = _get_llm()

    memory_block = format_memories(state.get("memories", []))
    intent = state.get("intent", {})
    emotion_hint = intent.get("emotion", "neutral")
    topic_hint = intent.get("topic", "")

    system_prompt = (
        PERSONA_SYSTEM
        + f"\n\n# 当前用户情绪: {emotion_hint}"
        + (f"\n# 当前话题: {topic_hint}" if topic_hint else "")
        + "\n\n# 关于用户的过往记忆\n"
        + memory_block
    )

    history = state.get("messages", [])
    lc_messages: list = [SystemMessage(content=system_prompt)]
    for m in history:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    lc_messages.append(HumanMessage(content=state["user_message"]))

    resp = llm.invoke(lc_messages)
    reply = resp.content.strip()
    return {"reply": reply}


def maybe_write_memory(state: ConversationState) -> ConversationState:
    """把本轮对话喂给 mem0, 让它抽取记忆.

    仅当消息有信息量时才写入, 避免噪音.
    """
    intent = state.get("intent", {})
    is_significant = intent.get("is_significant", False)

    if not is_significant:
        # 二次确认: 对很短 / 无信息量的消息跳过
        if not analyzer.should_remember(state["user_message"]):
            logger.info("skip memory write: not significant")
            return {}

    messages = [
        {"role": "user", "content": state["user_message"]},
        {"role": "assistant", "content": state.get("reply", "")},
    ]
    try:
        from ..memory import add_message

        add_message(
            messages=messages,
            user_id=state["user_id"],
            metadata={
                "category": "event" if is_significant else "fact",
                "emotion": intent.get("emotion", "neutral"),
                "topic": intent.get("topic", ""),
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("write memory failed: %s", e)
    return {}
