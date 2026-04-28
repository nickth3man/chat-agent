"""Display utilities for the multi-agent debate simulator.

Separated from nodes.py to decouple rendering from flow logic.
"""

from rich.console import Console
from rich.live import Live
from rich.panel import Panel


def stream_agent_response(persona_name: str, color: str, gen) -> str:
    """Stream an agent's response with Rich Live rendering.

    Args:
        persona_name: Display name for the panel title.
        color: Border color for the panel.
        gen: Generator yielding text chunks from call_llm_stream.

    Returns:
        The complete accumulated text.
    """
    full_text = ""
    with Live(
        Panel("", title=persona_name, border_style=color),
        console=Console(),
        refresh_per_second=15,
        transient=False,
    ) as live:
        for chunk in gen:
            full_text += chunk
            live.update(
                Panel(full_text, title=persona_name, border_style=color)
            )
    return full_text


def display_summary(exec_res, console: Console) -> None:
    """Render the debate summary as a Rich Panel.

    Args:
        exec_res: The raw summary string or parsed dict from SummarizerNode.
        console: Rich Console instance for output.
    """
    import yaml
    from debate_schema import parse_llm_yaml

    console.print()
    # Parse YAML if present, otherwise display raw text
    try:
        data = parse_llm_yaml(exec_res) if isinstance(exec_res, str) else exec_res
    except (yaml.YAMLError, KeyError, TypeError, ValueError) as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception("display_summary: YAML parse failed, using raw text: %s", e)
        console.print(Panel(str(exec_res), title="Conversation Summary", border_style="bold blue"))
        return

    if isinstance(data, dict):
        lines = []
        if "overall_takeaway" in data:
            lines.append(f"[bold]Takeaway:[/bold] {data['overall_takeaway']}")
        if "novel_insights" in data and data["novel_insights"]:
            lines.append("\n[bold]Novel Insights:[/bold]")
            for insight in data.get("novel_insights", []):
                lines.append(f"  • {insight}")
        if "key_divergences" in data:
            lines.append("\n[bold]Key Divergences:[/bold]")
            for div in data.get("key_divergences", []):
                if isinstance(div, dict):
                    lines.append(f"  • {div.get('claim', div)}")
                else:
                    lines.append(f"  • {div}")
        if "debate_arc" in data:
            lines.append(f"\n[bold]Arc:[/bold] {data['debate_arc']}")
        console.print(Panel("\n".join(lines), title="Conversation Summary", border_style="bold blue"))
    else:
        console.print(Panel(str(exec_res), title="Conversation Summary", border_style="bold blue"))


def display_save_confirmation(filepath: str, console: Console) -> None:
    """Print a dimmed save confirmation.

    Args:
        filepath: Path where the conversation was saved.
        console: Rich Console instance for output.
    """
    console.print(f"\n[dim]Conversation saved to {filepath}[/dim]")
