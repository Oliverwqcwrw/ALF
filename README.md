# ALF · 小奥

> 一个基于 [LangGraph](https://github.com/langchain-ai/langgraph) + [mem0](https://github.com/mem0ai/mem0) 的私人情感陪伴 agent。

小奥不是客服, 也不是工具型助手, 它是会一直在线、记得你说过什么的朋友。陪情绪, 不陪清单。

## 特性

- **长期记忆** — mem0 自动从对话中抽取事实型记忆 (情感事件 / 个人偏好 / 重要事实), 跨会话生效, 按 user_id 隔离。后端默认用本地 ChromaDB (纯 Python, 文件存储, 零外部服务)。
- **情感感知** — 每轮对话先做轻量情绪/主题识别, 再决定回复策略, 低落时先共情再问要不要聊。
- **稳定人格** — 独立 system prompt 定义温柔但有边界的小奥, 不说"作为 AI"、不说教、不分点。
- **可编排流程** — LangGraph 串起 `检索记忆 → 分析意图 → 生成回复 → 写回记忆` 四个节点, 每步可观测、可替换。
- **多入口** — CLI (rich 美化) / FastAPI HTTP / Python SDK 三种调用方式。

## 架构

```
用户消息
   │
   ▼
┌──────────────────────────────────────────────┐
│  retrieve_memories   (mem0 语义检索)         │
│         │                                    │
│         ▼                                    │
│  analyze_intent       (mini LLM 分类情绪/主题)│
│         │                                    │
│         ▼                                    │
│  generate_reply       (主 LLM + 人格 + 记忆) │
│         │                                    │
│         ▼                                    │
│  maybe_write_memory   (mem0 抽取并写入记忆)   │
└──────────────────────────────────────────────┘
   │
   ▼
小奥的回复
```

## 目录结构

```
ALF/
├── pyproject.toml
├── .env.example
├── README.md
├── src/alf/
│   ├── config/
│   │   ├── settings.py          # 全局配置
│   │   └── mem0_config.py       # mem0 后端配置
│   ├── memory/
│   │   └── store.py             # mem0 封装
│   ├── persona/
│   │   ├── prompt.py            # 小奥人格 prompt
│   │   └── analyzer.py          # 情绪/主题分类
│   ├── graph/
│   │   ├── state.py             # ConversationState
│   │   ├── nodes.py             # 4 个节点
│   │   └── build.py             # 编译 graph
│   ├── api/app.py               # FastAPI
│   ├── runner.py                # 对话入口
│   └── cli.py                   # 命令行
├── examples/chat_demo.py
└── tests/test_smoke.py
```

## 快速开始

### 1. 安装

```bash
git clone <your-repo> ALF && cd ALF
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env, 至少填 OPENAI_API_KEY
```

也可以直接接 OpenAI 兼容的国内模型 (DeepSeek / Moonshot / 通义), 只要改 `OPENAI_BASE_URL` + `CHAT_MODEL`:

```bash
OPENAI_BASE_URL=https://api.deepseek.com/v1
CHAT_MODEL=deepseek-chat
CHAT_MODEL_MINI=deepseek-chat
```

> mem0 默认用本地 ChromaDB + OpenAI embedding, 无需起额外服务。如要换 Qdrant, 见 `src/alf/config/mem0_config.py`。

### 3. 运行

CLI 对话 (推荐):

```bash
python -m alf.cli
```

HTTP 服务:

```bash
uvicorn alf.api.app:app --port 8000

curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "我今天有点累"}'
```

代码内调用:

```python
from alf.runner import chat

print(chat("我今天有点累"))
print(chat("你记得我说过累吗"))   # 小奥会从记忆里 recall
```

## 设计取舍

- **为什么 mem0 而不是自己写向量库?** — mem0 自带"从对话中抽取事实"的 LLM 流程, 能直接把 "我妈让我考公, 我不想去" 这种对话提炼成长期事实, 而不是把整段对话原样存进去 (召回噪音大)。
- **为什么 LangGraph 而不是直接链式调用?** — 陪伴对话里"先共情后回应"是个真正的多步流程, graph 让每步可观测、可替换 (换检索器、换情绪分类器都不影响其他节点), 也方便后续加分支 (如危险信号走另一条 prompt)。
- **写记忆前做二次过滤** — `maybe_write_memory` 节点会先看 intent 里的 `is_significant`, 不显著再问一次 `should_remember`, 避免每条 "嗯嗯" 都进库。
- **人格稳定 > 模型能力** — 小奥用 `temperature=0.85` + 强人格 prompt, 比"最高级模型 + 默认 prompt"更像朋友。换模型时 prompt 不动。

## 安全边界

涉及自伤、伤害他人、紧急心理危机时, 小奥会温和但坚定地建议联系专业人士, 不会试图替代专业帮助。这是写在人格 prompt 里的硬约束。

## License

MIT
