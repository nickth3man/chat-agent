"""Non-interactive test runner for prompt engineering iterations.

Runs the full debate flow with a given topic, no user input required.
Saves output to data/conversations/ for later analysis.
"""
import logging
import os
import sys
from dotenv import load_dotenv
load_dotenv(override=True)

from utils.constants import (
    TOPIC, PERSONAS, CONVERSATION, TURN, MAX_TURNS, LAST_SPEAKER,
    MODERATOR_NOTES, MODERATOR_INTERVENTIONS, NEXT_SPEAKER,
    SPEAKER_TURNS, STALL_COUNT, RESEARCH_COUNT, RESEARCH_BUDGET, RESEARCH_MAX_COUNT, RESEARCH_NOTES, SUMMARY,
)

def run_test(topic: str, max_turns: int = 8, model: str | None = None) -> dict:
    """Run a complete debate on the given topic and return the shared store."""
    if model:
        os.environ["LLM_MODEL"] = model
    
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
        RESEARCH_BUDGET: RESEARCH_MAX_COUNT,
        RESEARCH_NOTES: [],
        SUMMARY: "",
    }

    from flow import create_conversation_flow
    flow = create_conversation_flow()
    flow.run(shared)
    return shared


if __name__ == "__main__":
    # Accept topic from CLI argument
    topic = sys.argv[1] if len(sys.argv) > 1 else "Should AI development be paused for safety research?"
    max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    print(f"\n{'='*60}")
    print(f"TOPIC: {topic}")
    print(f"MAX TURNS: {max_turns}")
    print(f"{'='*60}\n")
    
    result = run_test(topic, max_turns)
    print("\nDone.")
