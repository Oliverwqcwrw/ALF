"""langgraph 节点: 检索记忆 / 分析意图(含危机检测) / 路由 / 生成回复 / 自检 / 写入记忆."""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import settings
from ..memory import format_memories, search_memory
from ..persona import (
    CRISIS_SYSTEM,
    EMPATHIZE_PREFIX_HINT,
    PERSONA_SYSTEM,
    PROACTIVE_CARE_HINT,
)
from ..persona.analyzer import LOW_EMOTIONS, analyzer
from .state import ConversationState

logger = logging.getLogger(__name__)

# 自检失败后最多重生成次数 (再多则成本失控, 放过).
MAX_RETRY = 1

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
    """根据「本轮消息 + 近两轮用户消息」组合检索, 提升上下文相关性.

    单条 query 只能命中语义相近的记忆, 组入近期对话后能召回
    "上周说项目压力大" 这类真正的情绪上下文.
    """
    msg = state["user_message"]
    history = state.get("messages", [])
    recent_user = [m["content"] for m in history if m.get("role") == "user"][-2:]
    query = " ".join(recent_user + [msg])[-512:]
    memories = search_memory(query=query, user_id=state["user_id"], top_k=6)
    return {"memories": memories}


def analyze_intent(state: ConversationState) -> ConversationState:
    """分析意图 + 危机检测, 更新跨轮情绪状态, 决定路由.

    情绪历史与连续低落计数是跨轮状态, 由 runner 注入并持久化;
    这里在本轮计算后写回 state, runner 再读回.
    """
    intent = analyzer.extract_intent(state["user_message"])
    emotion = intent.get("emotion", "neutral")
    logger.info("intent: %s", intent)

    emotion_history = list(state.get("emotion_history", []))
    emotion_history.append(emotion)
    emotion_history = emotion_history[-10:]  # 只保留最近 10 轮

    # 从最近往前数连续低落轮数
    consecutive_low = 0
    for e in reversed(emotion_history):
        if e in LOW_EMOTIONS:
            consecutive_low += 1
        else:
            break

    if intent.get("is_crisis"):
        route = "crisis"
    elif emotion in LOW_EMOTIONS:
        route = "empathize"
    else:
        route = "normal"

    return {
        "intent": intent,
        "emotion_history": emotion_history,
        "consecutive_low": consecutive_low,
        "route": route,
    }


def route_by_intent(state: ConversationState) -> str:
    """条件边: 根据 analyze_intent 算出的 route 决定下一步."""
    return state.get("route", "normal")


def build_reply_messages(state: ConversationState) -> list:
    """构造回复所需的消息, 按 route 选择 system prompt, 普通/流式复用."""
    route = state.get("route", "normal")

    if route == "crisis":
        system_prompt = CRISIS_SYSTEM.format(alf_agent_name=settings.alf_agent_name)
    else:
        system_prompt = PERSONA_SYSTEM
        if route == "empathize":
            system_prompt += EMPATHIZE_PREFIX_HINT
        # 连续低落触发的主动关怀 (危机分支不用, 危机 prompt 优先)
        if state.get("proactively_care"):
            system_prompt += PROACTIVE_CARE_HINT

    # 注入记忆
    memory_block = format_memories(state.get("memories", []))
    system_prompt += "\n\n# 关于用户的过往记忆\n" + memory_block

    # 自检失败重试: 提示上一版的问题
    retry = state.get("retry_count", 0)
    if retry > 0 and not state.get("self_check_passed", True):
        system_prompt += (
            "\n\n# 注意: 上一版回复没过自检"
            f"({state.get('check_reason', '')}), 请据此调整这一版."
        )

    history = state.get("messages", [])
    lc_messages: list = [SystemMessage(content=system_prompt)]
    for m in history:
        if m.get("role") == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m.get("role") == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    lc_messages.append(HumanMessage(content=state["user_message"]))
    return lc_messages


def generate_reply(state: ConversationState) -> ConversationState:
    """调用 LLM 生成回复, 注入人格 + 路由策略 + 历史 + 记忆."""
    resp = _get_llm().invoke(build_reply_messages(state))
    reply = _content_to_text(resp.content).strip()
    return {"reply": reply}


def self_check(state: ConversationState) -> ConversationState:
    """回复自检: 共情/说教/危机资源. 不达标触发一次重生成."""
    reply = state.get("reply", "")
    intent = state.get("intent", {})
    emotion = intent.get("emotion", "neutral")
    is_crisis = intent.get("is_crisis", False)
    passed, reason = analyzer.check_reply(reply, emotion, is_crisis)
    logger.info("self_check passed=%s reason=%s", passed, reason)
    return {
        "self_check_passed": passed,
        "check_reason": reason,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def after_self_check(state: ConversationState) -> str:
    """条件边: 通过或重试超限则写记忆, 否则回退重生成."""
    if state.get("self_check_passed", True):
        return "maybe_write_memory"
    if state.get("retry_count", 0) >= MAX_RETRY:
        return "maybe_write_memory"
    return "generate_reply"


def stream_reply(state: ConversationState):
    """逐片段产出 LLM 回复文本 (流式路径, 跳过自检)."""
    for chunk in _get_llm().stream(build_reply_messages(state)):
        text = _content_to_text(chunk.content)
        if text:
            yield text


def _content_to_text(content: object) -> str:
    """兼容 OpenAI 兼容接口可能返回的字符串或内容块列表."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def maybe_write_memory(state: ConversationState) -> ConversationState:
    """把本轮对话喂给 mem0, 让它抽取记忆.

    仅当消息有信息量时才写入, 避免噪音.
    """
    intent = state.get("intent", {})
    is_significant = intent.get("is_significant", False)

    # 短路: 显著则直接写; 否则再做一次轻量确认, 对很短 / 无信息量消息跳过.
    if not is_significant and not analyzer.should_remember(state["user_message"]):
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
