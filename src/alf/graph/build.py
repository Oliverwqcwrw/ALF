"""构建 langgraph 对话工作流.

结构:
  retrieve_memories ──▶ analyze_intent ──▶ route_by_intent (条件)
        │                                       │
        │                  ┌────────────────────┼────────────────────┐
        │                  ▼                    ▼                    ▼
        │             crisis             empathize              normal
        │                  └────────────────────┴────────────────────┘
        │                                       │
        │                                       ▼
        │                                  self_check (条件) ──┘
        │                                       │
        │                                       ▼
        └───────────────────────────▶ maybe_write_memory ──▶ update_impression ──▶ END
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    after_self_check,
    analyze_intent,
    generate_reply,
    maybe_write_memory,
    retrieve_memories,
    route_by_intent,
    self_check,
    update_impression,
)
from .state import ConversationState


def build_graph():
    g = StateGraph(ConversationState)

    g.add_node("retrieve_memories", retrieve_memories)
    g.add_node("analyze_intent", analyze_intent)
    g.add_node("generate_reply", generate_reply)
    g.add_node("self_check", self_check)
    g.add_node("maybe_write_memory", maybe_write_memory)
    g.add_node("update_impression", update_impression)

    g.set_entry_point("retrieve_memories")
    g.add_edge("retrieve_memories", "analyze_intent")

    # 条件路由: crisis / empathize / normal 都进同一个 generate_reply
    # (内部按 route 选 prompt), 三个目标合并到一处, 避免重复节点.
    g.add_conditional_edges(
        "analyze_intent",
        route_by_intent,
        {
            "crisis": "generate_reply",
            "empathize": "generate_reply",
            "normal": "generate_reply",
        },
    )

    g.add_edge("generate_reply", "self_check")
    # 自检通过 / 重试超限 → 写记忆; 否则回退重生成一次.
    g.add_conditional_edges(
        "self_check",
        after_self_check,
        {
            "maybe_write_memory": "maybe_write_memory",
            "generate_reply": "generate_reply",
        },
    )
    g.add_edge("maybe_write_memory", "update_impression")
    g.add_edge("update_impression", END)

    return g.compile()


app_graph = build_graph()
