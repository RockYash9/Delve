"""
Interactive command-line loop for Delve.

Brick 3: one Agent instance persists for the whole session, so
follow-up questions carry context from earlier in the conversation.
Type 'reset' to start a fresh conversation without restarting the app.
"""

from rich.console import Console
from rich.panel import Panel

import config
from src.agent import Agent

console = Console()


def run() -> None:
    config.validate_config()
    agent = Agent()

    console.print(Panel.fit(
        "[bold cyan]Delve[/bold cyan] — brick 3: conversational search\n"
        "Type your question, 'reset' to start fresh, or 'exit' to quit.",
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
        if user_input.lower() == "reset":
            agent.reset()
            console.print("[dim]Conversation reset — starting fresh.[/dim]")
            continue

        with console.status("[dim]thinking...[/dim]"):
            answer = agent.ask(user_input)

        console.print(Panel(answer, title="Delve", border_style="cyan"))