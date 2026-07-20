"""示例: 用代码方式和小奥对话, 演示记忆的跨轮生效."""
from __future__ import annotations

from alf.runner import chat

if __name__ == "__main__":
    print(chat("我今天和我妈大吵了一架, 心情很差"))
    print("---")
    print(chat("她非让我去考公, 我一点都不想去"))
    print("---")
    # 下一轮小奥应该能从记忆里 recall 出"和妈妈吵架"这件事
    print(chat("其实我也不知道自己想做什么"))
    print("---")
    print(chat("你记得我和我妈的事吗?"))
