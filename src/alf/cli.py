"""命令行交互界面. 用 rich 做美化."""
from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from . import __version__
from .config import settings
from .runner import chat, reset

console = Console()


def banner() -> None:
    console.print(
        Panel.fit(
            Text(f"{settings.alf_agent_name} · v{__version__}", style="bold magenta"),
            subtitle=f"陪伴着 {settings.alf_user_id}",
            border_style="magenta",
        )
    )
    console.print("[dim]输入 /reset 清空当前会话历史, /quit 退出.\n[/dim]")


def main() -> int:
    banner()
    while True:
        try:
            user_input = Prompt.ask(f"[bold cyan]{settings.alf_user_id}[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n[dim]{settings.alf_agent_name}: 那先这样, 随时来找我.[/dim]")
            return 0

        if not user_input.strip():
            continue

        if user_input.strip() in {"/quit", "/exit", "/q"}:
            console.print(f"[dim]{settings.alf_agent_name}: 那先这样, 随时来找我.[/dim]")
            return 0
        if user_input.strip() == "/reset":
            reset()
            console.print("[dim]已清空本轮会话历史 (记忆仍然保留).[/dim]\n")
            continue

        try:
            reply = chat(user_input)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]出错了: {e}[/red]\n")
            continue

        console.print(
            Panel(
                Text(reply, style="white"),
                title=f"[bold magenta]{settings.alf_agent_name}[/bold magenta]",
                border_style="magenta",
            )
        )
        console.print()


if __name__ == "__main__":
    sys.exit(main())
