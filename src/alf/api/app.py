"""FastAPI 接口, 提供 HTTP 调用方式."""
from __future__ import annotations

import json
import logging
import queue
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints

from .. import store
from ..runner import chat, forget_everything, forget_memory, get_memories, reset, stream
from ..scheduler import get_proactive_queue, start_scheduler, stop_scheduler

# Uvicorn 默认将该日志器输出到终端；用于 SSE 运行状态与耗时日志。
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title="ALF", description="私人情感陪伴 agent 小奥", lifespan=lifespan)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

USER_ID_PATTERN = r"^[1-9]\d{4}$"
UserId = Annotated[str, StringConstraints(pattern=USER_ID_PATTERN)]
UserIdQuery = Annotated[str, Query(pattern=USER_ID_PATTERN)]


class ChatRequest(BaseModel):
    message: str
    user_id: UserId


class SessionRequest(BaseModel):
    user_id: UserId


class ChatResponse(BaseModel):
    reply: str
    user_id: str


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    """小奥的浏览器聊天界面。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    store.register_user(req.user_id)
    reply = chat(req.message, user_id=req.user_id)
    return ChatResponse(reply=reply, user_id=req.user_id)


@app.post("/session")
def start_session(req: SessionRequest) -> dict:
    """校验并登记用户 ID；该 ID 是数据隔离标识，不是密码认证。"""
    store.register_user(req.user_id)
    return {"ok": True, "user_id": req.user_id}


@app.post("/chat/stream")
def stream_chat_endpoint(req: ChatRequest) -> StreamingResponse:
    """以 Server-Sent Events 将生成中的文本片段传给浏览器。"""

    def event(name: str, payload: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate():
        started_at = perf_counter()
        first_token = True
        try:
            store.register_user(req.user_id)
            for text in stream(req.message, user_id=req.user_id):
                if first_token:
                    logger.info("stream first token in %.0fms", (perf_counter() - started_at) * 1000)
                    first_token = False
                yield event("token", {"text": text})
        except Exception:
            logger.exception("stream chat failed")
            yield event("error", {"message": "小奥暂时没能接上这句话，请稍后再试。"})
        else:
            yield event("done", {"user_id": req.user_id})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/reset")
def reset_endpoint(user_id: UserIdQuery) -> dict:
    reset(user_id=user_id)
    return {"ok": True}


@app.get("/memories")
def memories_endpoint(user_id: UserIdQuery) -> dict:
    """用户可见的长期记忆清单。"""
    memories = get_memories(user_id=user_id)
    return {
        "memories": [
            {"id": str(item.get("id", "")), "text": item.get("memory") or item.get("text") or ""}
            for item in memories
            if item.get("id") and (item.get("memory") or item.get("text"))
        ]
    }


@app.delete("/memories/{memory_id}")
def forget_memory_endpoint(memory_id: str, user_id: UserIdQuery) -> dict:
    if not forget_memory(memory_id, user_id=user_id):
        raise HTTPException(status_code=404, detail="没有找到这条记忆")
    return {"ok": True}


@app.delete("/memory")
def forget_everything_endpoint(user_id: UserIdQuery) -> dict:
    forget_everything(user_id=user_id)
    return {"ok": True}


@app.get("/proactive/stream")
def proactive_stream(user_id: UserIdQuery):
    """订阅该用户的主动开口消息 (深夜/久未对话), 以 SSE 推送."""
    q = get_proactive_queue(user_id)

    def generate():
        while True:
            try:
                msg = q.get(timeout=15)
                yield f"event: proactive\ndata: {json.dumps({'text': msg}, ensure_ascii=False)}\n\n"
            except queue.Empty:
                # keepalive, 防止代理/浏览器因空闲断连.
                yield ": ping\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
