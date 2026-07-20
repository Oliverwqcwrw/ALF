"""构建 langgraph 对话工作流.

结构:
  retrieve_memories ──▶ analyze_intent ──▶ generate_reply ──▶ maybe_write_memory
                                          (并行)               (依赖于 reply)
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import analyze_intent, generate_reply, maybe_write_memory, retrieve_memories
from .state import ConversationState


def build_graph():
    g = StateGraph(ConversationState)

    g.add_node("retrieve_memories", retrieve_memories)
    g.add_node("analyze_intent", analyze_intent)
    g.add_node("generate_reply", generate_reply)
    g.add_node("maybe_write_memory", maybe_write_memory)

    g.set_entry_point("retrieve_memories")
    g.add_edge("retrieve_memories", "analyze_intent")
    g.add_edge("analyze_intent", "generate_reply")
    g.add_edge("generate_reply", "maybe_write_memory")
    g.add_edge("maybe_write_memory", END)

    return g.compile()


app_graph = build_graph()
