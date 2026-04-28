from __future__ import annotations

import logging
import os
import sys
import signal
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.prompt import Prompt

from flow import create_conversation_flow
from utils.constants import (
    TOPIC, PERSONAS, CONVERSATION, TURN, MAX_TURNS, LAST_SPEAKER,
    MODERATOR_NOTES, MODERATOR_INTERVENTIONS, NEXT_SPEAKER,
    SPEAKER_TURNS, STALL_COUNT, RESEARCH_COUNT, RESEARCH_NOTES, SUMMARY,
)


def _setup_logging() -> None:
    """Log all INFO+ to file; console shows only WARNING+ (Rich handles the UI)."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"conversation_{datetime.now():%Y%m%d_%H%M%S}.log")

    root = logging.getLogger()
    root.handlers.clear()

    # File handler — everything preserved
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root.addHandler(fh)

    # Console handler — only warnings+ (no httpx noise, no INFO spam)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    root.addHandler(ch)

    root.setLevel(logging.INFO)

    # Note: httpx/httpcore log at INFO level too, which is fine —


def main() -> None:
    _setup_logging()
    logger = logging.getLogger(__name__)

    # ── Graceful shutdown handling ──
    _shutdown_requested = False

    def _handle_shutdown(signum: int, frame: object) -> None:
        nonlocal _shutdown_requested
        sig_name = signal.Signals(signum).name
        logger.warning("Received %s — shutting down gracefully. Press again to force quit.", sig_name)
        if _shutdown_requested:
            logger.warning("Second %s received — forcing exit.", sig_name)
            sys.exit(1)
        _shutdown_requested = True
        console.print()
        console.print("[yellow]Shutting down...[/yellow]")

    signal.signal(signal.SIGINT, _handle_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_shutdown)

    console = Console()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]ERROR: OPENROUTER_API_KEY environment variable not set.[/red]")
        console.print("Set it with: [bold]export OPENROUTER_API_KEY=your-key-here[/bold]")
        sys.exit(1)

    model = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")
    logger.info("Starting conversation with model=%s, max_turns=%s", model, os.environ.get("MAX_TURNS", "50"))
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

    shared: dict[str, object] = {
        TOPIC: topic.strip(),
        PERSONAS: [],
        CONVERSATION: [],
        TURN: 0,
        MAX_TURNS: max_turns,
        LAST_SPEAKER: None,
        MODERATOR_NOTES: None,
        MODERATOR_INTERVENTIONS: [],
        NEXT_SPEAKER: None,
        SPEAKER_TURNS: {},
        STALL_COUNT: 0,
        RESEARCH_COUNT: 0,
        RESEARCH_NOTES: [],
        SUMMARY: "",
    }

    flow = create_conversation_flow()
    console.print(f"[dim]Generating personas and starting {max_turns}-turn conversation...[/dim]")
    console.print()

    logger.info("Starting flow execution")
    try:
        flow.run(shared)
    except KeyboardInterrupt:
        logger.info("Flow interrupted by user")
        console.print("[yellow]Conversation interrupted.[/yellow]")
        sys.exit(0)
    logger.info("Flow execution completed")


if __name__ == "__main__":
    main()
