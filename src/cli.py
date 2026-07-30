"""
Interactive command-line loop for Delve.

Brick 1: a plain REPL — type a question, get an answer, repeat.
No conversation memory yet (each question is independent) — that's brick 3.
"""

from rich.console import Console
from rich.panel import Panel

import config
from src import agent

console = Console()


def run() -> None:
    config.validate_config()

    console.print(Panel.fit(
        "[bold cyan]Delve[/bold cyan] — brick 1: single-turn Q&A\n"
        "Type your question, or 'exit' to quit.",
    ))

    while True:
        try:
            user_input = console.input("[bold green]you>[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            console.print("[dim]bye[/dim]")
            break

        with console.status("[dim]thinking...[/dim]"):
            answer = agent.ask(user_input)

        console.print(Panel(answer, title="Delve", border_style="cyan"))
