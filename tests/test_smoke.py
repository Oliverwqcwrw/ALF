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
