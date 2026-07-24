"""FastAPI 接口, 提供 HTTP 调用方式."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..runner import chat, reset, stream

app = FastAPI(title="ALF", description="私人情感陪伴 agent 小奥")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
logger = logging.getLogger(__name__)


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
