"""SQLite 会话状态持久化.

存两样东西 (按 user_id 隔离):
- 短期对话历史: 重启后仍能接上最近 N 轮上下文, 不像原来那样重启即失忆.
- 情绪轨迹: 最近每轮的 emotion 标签, 用于连续低落检测 / 主动关怀.

纯标准库 sqlite3, 零外部依赖. 进程内单例连接.
"""
from __future__ import annotations

import sqlite3
import threading

from .config import settings

_HISTORY_LIMIT = 20
_EMOTION_LIMIT = 10

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(settings.alf_state_path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT    NOT NULL,
                role    TEXT    NOT NULL,
                content TEXT    NOT NULL,
                ts      REAL    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS emotions (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT    NOT NULL,
                emotion TEXT    NOT NULL,
                ts      REAL    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);
            CREATE INDEX IF NOT EXISTS idx_emotions_user ON emotions(user_id, id);
            """
        )
        _conn.commit()
    return _conn


def get_history(user_id: str, limit: int = _HISTORY_LIMIT) -> list[dict[str, str]]:
    """返回最近 limit 条对话, 按时间正序."""
    rows = _get_conn().execute(
        "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def append_message(user_id: str, role: str, content: str) -> None:
    import time

    with _lock:
        _get_conn().execute(
            "INSERT INTO messages(user_id, role, content, ts) VALUES(?, ?, ?, ?)",
            (user_id, role, content, time.time()),
        )
        _get_conn().commit()
    _trim(user_id, "messages", _HISTORY_LIMIT)


def clear_history(user_id: str) -> None:
    with _lock:
        _get_conn().execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        _get_conn().execute("DELETE FROM emotions WHERE user_id = ?", (user_id,))
        _get_conn().commit()


def get_emotion_history(user_id: str, limit: int = _EMOTION_LIMIT) -> list[str]:
    rows = _get_conn().execute(
        "SELECT emotion FROM emotions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [r["emotion"] for r in reversed(rows)]


def append_emotion(user_id: str, emotion: str) -> None:
    import time

    with _lock:
        _get_conn().execute(
            "INSERT INTO emotions(user_id, emotion, ts) VALUES(?, ?, ?)",
            (user_id, emotion, time.time()),
        )
        _get_conn().commit()
    _trim(user_id, "emotions", _EMOTION_LIMIT)


def consecutive_low_count(user_id: str) -> int:
    """基于持久化的情绪轨迹算连续低落轮数 (不含本轮)."""
    from .persona.analyzer import LOW_EMOTIONS

    history = get_emotion_history(user_id)
    n = 0
    for e in reversed(history):
        if e in LOW_EMOTIONS:
            n += 1
        else:
            break
    return n


def _trim(user_id: str, table: str, limit: int) -> None:
    """保留每个 user_id 最近 limit 条, 删旧的."""
    _get_conn().execute(
        f"DELETE FROM {table} WHERE user_id = ? AND id NOT IN "
        f"(SELECT id FROM {table} WHERE user_id = ? ORDER BY id DESC LIMIT ?)",
        (user_id, user_id, limit),
    )
    _get_conn().commit()
