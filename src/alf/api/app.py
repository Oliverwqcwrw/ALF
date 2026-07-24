"""FastAPI 接口, 提供 HTTP 调用方式."""
from __future__ import annotations

import json
import logging
import queue
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..runner import chat, forget_everything, forget_memory, get_memories, reset, stream
from ..scheduler import get_proactive_queue, start_scheduler, stop_scheduler

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title="ALF", description="私人情感陪伴 agent 小奥", lifespan=lifespan)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str
    user_id: str | None = None


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
    reply = chat(req.message, user_id=req.user_id)
    return ChatResponse(reply=reply, user_id=req.user_id or "default")


@app.post("/chat/stream")
def stream_chat_endpoint(req: ChatRequest) -> StreamingResponse:
    """以 Server-Sent Events 将生成中的文本片段传给浏览器。"""

    def event(name: str, payload: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate():
        try:
            for text in stream(req.message, user_id=req.user_id):
                yield event("token", {"text": text})
        except Exception:
            logger.exception("stream chat failed")
            yield event("error", {"message": "小奥暂时没能接上这句话，请稍后再试。"})
        else:
            yield event("done", {"user_id": req.user_id or "default"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/reset")
def reset_endpoint(user_id: str | None = None) -> dict:
    reset(user_id=user_id)
    return {"ok": True}


@app.get("/memories")
def memories_endpoint(user_id: str | None = None) -> dict:
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
def forget_memory_endpoint(memory_id: str, user_id: str | None = None) -> dict:
    if not forget_memory(memory_id, user_id=user_id):
        raise HTTPException(status_code=404, detail="没有找到这条记忆")
    return {"ok": True}


@app.delete("/memory")
def forget_everything_endpoint(user_id: str | None = None) -> dict:
    forget_everything(user_id=user_id)
    return {"ok": True}


@app.get("/proactive/stream")
def proactive_stream(user_id: str | None = None):
    """订阅该用户的主动开口消息 (深夜/久未对话), 以 SSE 推送."""
    uid = user_id or "default"
    q = get_proactive_queue(uid)

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
