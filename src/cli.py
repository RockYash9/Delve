"""
Interactive command-line loop for Delve.

Brick 5: help and export commands, plus a proper startup banner —
this is the "feels finished" layer on top of bricks 1-4.
"""

from rich.console import Console
from rich.panel import Panel

import config
from src import reports
from src.agent import Agent

console = Console()

COMMANDS = {
    "help": "show this list of commands",
    "reset": "start a fresh conversation (clears memory, keeps the knowledge cache)",
    "export": "save this conversation as a markdown report with sources",
    "exit / quit": "leave Delve",
}


def print_banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]Delve[/bold cyan] — conversational research engine\n"
        "Type a question, or one of: " + ", ".join(COMMANDS.keys()),
    ))


def print_help() -> None:
    lines = [f"[bold]{cmd}[/bold] — {desc}" for cmd, desc in COMMANDS.items()]
    console.print(Panel("\n".join(lines), title="Commands", border_style="cyan"))


def run() -> None:
    config.validate_config()
    agent = Agent()

    print_banner()

    while True:
        try:
            user_input = console.input("[bold green]you>[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye[/dim]")
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command in {"exit", "quit"}:
            console.print("[dim]bye[/dim]")
            break

        if command == "help":
            print_help()
            continue

        if command == "reset":
            agent.reset()
            console.print("[dim]Conversation reset — starting fresh.[/dim]")
            continue

        if command == "export":
            transcript = agent.get_transcript()
            if not transcript:
                console.print("[dim]Nothing to export yet — ask something first.[/dim]")
                continue
            report_text = reports.build_report(transcript, agent.get_sources())
            path = reports.save_report(report_text)
            console.print(f"[green]Saved report to {path}[/green]")
            continue

        with console.status("[dim]thinking...[/dim]"):
            answer = agent.ask(user_input)

        console.print(Panel(answer, title="Delve", border_style="cyan"))
