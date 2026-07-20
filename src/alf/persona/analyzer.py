"""对话轻量分析: 情绪/主题/是否值得记忆.

用一个 mini 模型做快速分类, 避免主回复模型做太多 meta 工作.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Literal

from langchain_openai import ChatOpenAI

from ..config import settings

logger = logging.getLogger(__name__)

Emotion = Literal["happy", "sad", "anxious", "angry", "calm", "mixed", "neutral"]


class Analyzer:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.chat_model_mini,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.0,
        )

    def extract_intent(self, message: str) -> dict:
        """返回 {emotion, topic, is_significant}."""
        from .prompt import EXTRACT_INTENT

        try:
            raw = self.llm.invoke(EXTRACT_INTENT.format(message=message)).content
            data = json.loads(self._strip_code_fence(raw))
            return {
                "emotion": data.get("emotion", "neutral"),
                "topic": data.get("topic", ""),
                "is_significant": bool(data.get("is_significant", False)),
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("extract_intent failed: %s", e)
            return {"emotion": "neutral", "topic": "", "is_significant": False}

    def should_remember(self, message: str) -> bool:
        from .prompt import SHOULD_REMEMBER

        try:
            raw = self.llm.invoke(SHOULD_REMEMBER.format(message=message)).content
            return "yes" in raw.strip().lower()
        except Exception as e:  # noqa: BLE001
            logger.warning("should_remember failed: %s", e)
            return False

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """去掉模型可能包的 ```json ... ```."""
        text = text.strip()
        m = re.match(r"^```(?:json)?\s*(.*)\s*```$", text, re.DOTALL)
        return m.group(1).strip() if m else text


analyzer = Analyzer()
