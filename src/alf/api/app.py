"""FastAPI 接口, 提供 HTTP 调用方式."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from ..runner import chat, reset

app = FastAPI(title="ALF", description="私人情感陪伴 agent 小奥")


class ChatRequest(BaseModel):
    message: str
    user_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    user_id: str


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    reply = chat(req.message, user_id=req.user_id)
    return ChatResponse(reply=reply, user_id=req.user_id or "default")


@app.post("/reset")
def reset_endpoint(user_id: str | None = None) -> dict:
    reset(user_id=user_id)
    return {"ok": True}
