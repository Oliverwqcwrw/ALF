"""冒烟测试: 只验证模块能 import 与状态类型."""
from alf.graph import ConversationState
from alf.persona import PERSONA_SYSTEM, EXTRACT_INTENT, SHOULD_REMEMBER


def test_imports():
    # 不真正调用 LLM, 避免单测依赖外部服务
    assert settings_module_ok()


def settings_module_ok() -> bool:
    from alf import config

    return config.settings.alf_agent_name == "小奥"


def test_state_fields():
    s: ConversationState = {"user_message": "hi", "user_id": "u"}
    assert s["user_message"] == "hi"


def test_prompts_nonempty():
    assert "小奥" in PERSONA_SYSTEM
    assert "{message}" in EXTRACT_INTENT
    assert "{message}" in SHOULD_REMEMBER
