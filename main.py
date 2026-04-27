from dotenv import load_dotenv
load_dotenv()

import os
import sys

from rich.console import Console
from rich.prompt import Prompt

from flow import create_conversation_flow


def main():
    console = Console()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]ERROR: OPENROUTER_API_KEY environment variable not set.[/red]")
        console.print("Set it with: [bold]export OPENROUTER_API_KEY=your-key-here[/bold]")
        sys.exit(1)

    model = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")
    console.print(f"[dim]Using model: {model}[/dim]")
    console.print()

    topic = Prompt.ask("[bold]Enter a conversation topic[/bold]")
    if not topic.strip():
        console.print("[red]Topic cannot be empty.[/red]")
        sys.exit(1)

    max_turns_raw = os.environ.get("MAX_TURNS", "50")
    try:
        max_turns = int(max_turns_raw)
    except ValueError:
        max_turns = 50

    shared = {
        "topic": topic.strip(),
        "personas": [],
        "conversation": [],
        "turn": 0,
        "max_turns": max_turns,
        "last_speaker": None,
        "moderator_notes": None,
        "moderator_interventions": [],
        "next_speaker": None,
        "speaker_turns": {},
        "stall_count": 0,
        "summary": "",
    }

    flow = create_conversation_flow()
    console.print(f"[dim]Generating personas and starting {max_turns}-turn conversation...[/dim]")
    console.print()

    flow.run(shared)


if __name__ == "__main__":
    main()
