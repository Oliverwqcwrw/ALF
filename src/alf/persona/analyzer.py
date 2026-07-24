"""对话轻量分析: 情绪/主题/危机检测/是否值得记忆/回复自检.

用一个 mini 模型做快速分类, 避免主回复模型做太多 meta 工作.
危机检测先用关键词预筛 (零成本、不漏判), 再与 LLM 的 is_crisis 取或.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from langchain_openai import ChatOpenAI

from ..config import settings

logger = logging.getLogger(__name__)

Emotion = Literal["happy", "sad", "anxious", "angry", "calm", "mixed", "neutral"]

# 视为"低落"的情绪, 触发共情前置策略 / 连续低落计数.
LOW_EMOTIONS = {"sad", "anxious", "angry"}


class Analyzer:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.chat_model_mini,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.0,
        )

    def extract_intent(self, message: str) -> dict[str, Any]:
        """返回 {emotion, topic, is_significant, is_crisis}.

        危机检测: 关键词命中即 True (不依赖 LLM, 避免漏判隐晦表达);
        LLM 的 is_crisis 作为补充, 两者取或.
        """
        from .prompt import CRISIS_KEYWORDS, EXTRACT_INTENT

        keyword_crisis = any(kw in message for kw in CRISIS_KEYWORDS)
        try:
            raw = self.llm.invoke(EXTRACT_INTENT.format(message=message)).content
            data = json.loads(self._strip_code_fence(raw))
            llm_crisis = bool(data.get("is_crisis", False))
        except Exception as e:  # noqa: BLE001
            logger.warning("extract_intent failed: %s", e)
            data = {}
            llm_crisis = False

        return {
            "emotion": data.get("emotion", "neutral"),
            "topic": data.get("topic", ""),
            "is_significant": bool(data.get("is_significant", False)),
            # 关键词或 LLM 任一命中即视为危机.
            "is_crisis": keyword_crisis or llm_crisis,
        }

    def should_remember(self, message: str) -> bool:
        from .prompt import SHOULD_REMEMBER

        try:
            raw = self.llm.invoke(SHOULD_REMEMBER.format(message=message)).content
            return "yes" in raw.strip().lower()
        except Exception as e:  # noqa: BLE001
            logger.warning("should_remember failed: %s", e)
            return False

    def check_reply(self, reply: str, emotion: str, is_crisis: bool) -> tuple[bool, str]:
        """回复自检: 返回 (是否通过, 原因). 失败时触发一次重生成."""
        from .prompt import SELF_CHECK_PROMPT

        try:
            raw = self.llm.invoke(
                SELF_CHECK_PROMPT.format(reply=reply, emotion=emotion, is_crisis=is_crisis)
            ).content
            data = json.loads(self._strip_code_fence(raw))
            return bool(data.get("pass", True)), str(data.get("reason", ""))
        except Exception as e:  # noqa: BLE001
            # 自检失败不阻断主流程, 默认通过.
            logger.warning("check_reply failed: %s", e)
            return True, ""

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """去掉模型可能包的 ```json ... ```."""
        text = text.strip()
        m = re.match(r"^```(?:json)?\s*(.*)\s*```$", text, re.DOTALL)
        return m.group(1).strip() if m else text


analyzer = Analyzer()
