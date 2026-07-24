"""后台调度: 在深夜 / 久未对话时主动开口.

进程内单例后台线程, 每 15 分钟检查一次默认用户. 满足触发条件时
调 runner.generate_proactive 生成一条主动消息, 推到 per-user 内存队列,
HTTP SSE 端点订阅该队列推给连着网页的用户.

MVP 只管 settings.alf_user_id 这一个用户; 多用户场景需扩展用户注册表.
天气触发留 hook (需天气 API), 本模块不做.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

from . import store
from .config import settings
from .persona.analyzer import LOW_EMOTIONS

logger = logging.getLogger(__name__)

CHECK_INTERVAL = 900  # 15 分钟
LATE_NIGHT_HOURS = (23, 0, 1, 2)  # 23:00 - 02:59
ABSENT_HOURS = 24
LATE_NIGHT_COOLDOWN = 20 * 3600  # 每天最多 1 次深夜问候
ABSENT_COOLDOWN = 23 * 3600  # 每天最多 1 次久未对话问候

_thread: threading.Thread | None = None
_stop = threading.Event()
_lock = threading.Lock()
# (user_id, trigger) -> 上次触发时间戳, 用于冷却.
_last_triggered: dict[tuple[str, str], float] = {}
# per-user 持久主动消息队列: 调度器 put, SSE 端点 get.
_queues: dict[str, queue.Queue] = {}
_queues_lock = threading.Lock()


def start_scheduler() -> None:
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, daemon=True, name="alf-scheduler")
    _thread.start()
    logger.info("scheduler started")


def stop_scheduler() -> None:
    _stop.set()


def get_proactive_queue(user_id: str) -> queue.Queue:
    """返回该用户持久的主动消息队列 (SSE 端点订阅). 不存在则创建."""
    with _queues_lock:
        if user_id not in _queues:
            _queues[user_id] = queue.Queue()
        return _queues[user_id]


def _enqueue(user_id: str, message: str) -> None:
    get_proactive_queue(user_id).put(message)


def _loop() -> None:
    # 启动后先等一个周期, 避免进程刚起就触发.
    while not _stop.wait(CHECK_INTERVAL):
        try:
            _check_user(settings.alf_user_id)
        except Exception:
            logger.exception("scheduler check failed")


def _check_user(user_id: str) -> None:
    now = time.time()
    hour = time.localtime().tm_hour

    # 1) 深夜问候: 23:00-02:59 + 近 3 天有低落 + 今天没主动过.
    is_late_night = hour >= 23 or hour <= 2
    if is_late_night and _within_cooldown(user_id, "late_night", LATE_NIGHT_COOLDOWN):
        recent = store.get_recent_emotion_events(user_id, limit=10, days=3)
        if any(e.get("emotion") in LOW_EMOTIONS for e in recent):
            msg = _generate(user_id, "深夜了, TA 可能还醒着, 想着 TA 最近的低落")
            if msg:
                _enqueue(user_id, msg)
                _mark(user_id, "late_night")

    # 2) 久未对话: 距上次对话 > 24h + 最近有情绪事件.
    last = store.get_last_active(user_id)
    if (
        last
        and now - last > ABSENT_HOURS * 3600
        and _within_cooldown(user_id, "absent", ABSENT_COOLDOWN)
    ):
        recent = store.get_recent_emotion_events(user_id, limit=5, days=14)
        if recent:
            hours = int((now - last) / 3600)
            msg = _generate(user_id, f"距上次聊天已超过 {hours} 小时, TA 可能需要被想起")
            if msg:
                _enqueue(user_id, msg)
                _mark(user_id, "absent")


def _within_cooldown(user_id: str, trigger: str, cooldown: float) -> bool:
    last = _last_triggered.get((user_id, trigger))
    return last is None or (time.time() - last) >= cooldown


def _mark(user_id: str, trigger: str) -> None:
    with _lock:
        _last_triggered[(user_id, trigger)] = time.time()


def _generate(user_id: str, reason: str) -> str | None:
    from .runner import generate_proactive

    try:
        return generate_proactive(user_id, reason)
    except Exception as e:  # noqa: BLE001
        logger.warning("proactive generate failed: %s", e)
        return None
