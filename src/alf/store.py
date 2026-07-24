"""SQLite 会话状态持久化.

存两样东西 (按 user_id 隔离):
- 短期对话历史: 重启后仍能接上最近 N 轮上下文, 不像原来那样重启即失忆.
- 情绪轨迹: 最近每轮的 emotion 标签, 用于连续低落检测 / 主动关怀.

纯标准库 sqlite3, 零外部依赖. 进程内单例连接.
"""
from __future__ import annotations

import sqlite3
import threading
from typing import Any

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
            CREATE TABLE IF NOT EXISTS users (
                user_id    TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                last_seen  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS emotions (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT    NOT NULL,
                emotion TEXT    NOT NULL,
                ts      REAL    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS impressions (
                user_id    TEXT PRIMARY KEY,
                impression TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS emotion_events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   TEXT    NOT NULL,
                emotion   TEXT    NOT NULL,
                situation TEXT    NOT NULL,
                topic     TEXT    NOT NULL,
                ts        REAL    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);
            CREATE INDEX IF NOT EXISTS idx_users_last_seen ON users(last_seen);
            CREATE INDEX IF NOT EXISTS idx_emotions_user ON emotions(user_id, id);
            CREATE INDEX IF NOT EXISTS idx_emotion_events_user ON emotion_events(user_id, id);
            """
        )
        _conn.commit()
    return _conn


def register_user(user_id: str) -> None:
    """登记一个已进入小奥的用户，供多用户调度与数据隔离使用。"""
    import time

    now = time.time()
    with _lock:
        _get_conn().execute(
            "INSERT INTO users(user_id, created_at, last_seen) VALUES(?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET last_seen=excluded.last_seen",
            (user_id, now, now),
        )
        _get_conn().commit()


def get_registered_users() -> list[str]:
    """返回所有已登录过的用户，用于逐用户运行主动关怀调度。"""
    rows = _get_conn().execute("SELECT user_id FROM users ORDER BY last_seen DESC").fetchall()
    return [str(row["user_id"]) for row in rows]


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


def clear_personal_context(user_id: str) -> None:
    """删除 SQLite 中与用户有关的画像和情绪事件。"""
    with _lock:
        for table in ("impressions", "emotion_events"):
            _get_conn().execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
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


def get_last_active(user_id: str) -> float:
    """该用户最近一条消息的时间戳 (用于久未对话判定), 无记录返回 0."""
    row = _get_conn().execute(
        "SELECT ts FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    return row["ts"] if row else 0.0


def get_impression(user_id: str) -> str:
    """读取小奥对该用户的整体印象画像 (可能为空)."""
    row = _get_conn().execute(
        "SELECT impression FROM impressions WHERE user_id = ?", (user_id,)
    ).fetchone()
    return row["impression"] if row else ""


def set_impression(user_id: str, impression: str) -> None:
    """写入/覆盖印象画像 (UPSERT)."""
    import time

    with _lock:
        _get_conn().execute(
            "INSERT INTO impressions(user_id, impression, updated_at) VALUES(?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "impression=excluded.impression, updated_at=excluded.updated_at",
            (user_id, impression, time.time()),
        )
        _get_conn().commit()


_EMOTION_EVENT_KEEP = 50


def record_emotion_event(
    user_id: str, emotion: str, situation: str, topic: str
) -> None:
    """记录一条情绪-事件 (情绪 + 触发情境), 用于时序关联"这次为什么"."""
    import time

    with _lock:
        _get_conn().execute(
            "INSERT INTO emotion_events(user_id, emotion, situation, topic, ts) "
            "VALUES(?, ?, ?, ?, ?)",
            (user_id, emotion, situation, topic, time.time()),
        )
        _get_conn().commit()
    _trim(user_id, "emotion_events", _EMOTION_EVENT_KEEP)


def get_recent_emotion_events(
    user_id: str, limit: int = 5, days: int = 14
) -> list[dict[str, Any]]:
    """最近 days 天内最近 limit 条情绪事件 (时序正序, 用于注入关联)."""
    import time as _time

    cutoff = _time.time() - days * 86400
    rows = _get_conn().execute(
        "SELECT emotion, situation, topic, ts FROM emotion_events "
        "WHERE user_id = ? AND ts >= ? ORDER BY id DESC LIMIT ?",
        (user_id, cutoff, limit),
    ).fetchall()
    return [
        {
            "emotion": r["emotion"],
            "situation": r["situation"],
            "topic": r["topic"],
            "ts": r["ts"],
        }
        for r in reversed(rows)
    ]


def _trim(user_id: str, table: str, limit: int) -> None:
    """保留每个 user_id 最近 limit 条, 删旧的."""
    _get_conn().execute(
        f"DELETE FROM {table} WHERE user_id = ? AND id NOT IN "
        f"(SELECT id FROM {table} WHERE user_id = ? ORDER BY id DESC LIMIT ?)",
        (user_id, user_id, limit),
    )
    _get_conn().commit()
