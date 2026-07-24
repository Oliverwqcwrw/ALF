"""冒烟测试: 只验证模块能 import 与状态类型."""
from alf.graph import ConversationState
from alf.memory import store
from alf.persona import EXTRACT_INTENT, PERSONA_SYSTEM, SHOULD_REMEMBER


def test_imports():
    # 不真正调用 LLM, 避免单测依赖外部服务
    assert settings_module_ok()


def settings_module_ok() -> bool:
    from alf import config

    return config.settings.alf_agent_name == "小奥"


def test_state_fields():
    s: ConversationState = {"user_message": "hi", "user_id": "u"}
    assert s["user_message"] == "hi"


def test_search_memory_accepts_mem0_v2_response(monkeypatch):
    class MemoryClient:
        def search(self, **kwargs):
            return {"results": [{"memory": "用户喜欢雨天"}]}

    monkeypatch.setattr(store, "get_memory_client", lambda: MemoryClient())

    assert store.search_memory("天气", "u") == [{"memory": "用户喜欢雨天"}]


def test_prompts_nonempty():
    assert "小奥" in PERSONA_SYSTEM
    assert "{message}" in EXTRACT_INTENT
    assert "{message}" in SHOULD_REMEMBER


def test_stream_preflight_parallelizes_memory_and_intent(monkeypatch):
    """慢记忆检索不应拖住安全分析或首个流式 token。"""
    from threading import Event

    from alf import runner

    memory_started = Event()
    release_memory = Event()
    monkeypatch.setattr(runner, "STREAM_MEMORY_WAIT_SECONDS", 0.001)

    def slow_memory(_state):
        memory_started.set()
        release_memory.wait(timeout=1)
        return {"memories": [{"memory": "迟到的记忆"}]}

    def intent(_state):
        assert memory_started.wait(timeout=0.1)
        return {"intent": {"emotion": "neutral", "is_crisis": False}, "route": "normal"}

    monkeypatch.setattr(runner, "retrieve_memories", slow_memory)
    monkeypatch.setattr(runner, "analyze_intent", intent)

    state = runner._prepare_stream_state({"user_message": "hi"})
    release_memory.set()

    assert state["route"] == "normal"
    assert state["memories"] == []


def test_fast_intent_skips_remote_model_for_daily_emotion():
    from alf.persona.analyzer import Analyzer

    assert Analyzer._fast_intent("今天工作好累", False)["emotion"] == "sad"
    assert Analyzer._fast_intent("今天工作很顺利", False)["emotion"] == "happy"


def test_fast_intent_keeps_ambiguous_safety_language_for_model_review():
    from alf.persona.analyzer import Analyzer

    assert Analyzer._fast_intent("我感觉撑不住了", False) is None
    assert Analyzer._fast_intent("我想自杀", True)["is_crisis"] is True


def test_chat_request_requires_a_nonzero_five_digit_user_id():
    import pytest
    from pydantic import ValidationError

    from alf.api.app import ChatRequest

    assert ChatRequest(message="你好", user_id="12345").user_id == "12345"
    for invalid_id in ("01234", "1234", "123456", "abcde"):
        with pytest.raises(ValidationError):
            ChatRequest(message="你好", user_id=invalid_id)
