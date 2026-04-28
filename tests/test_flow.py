"""Integration tests for the full debate flow.

Tests the complete graph: InitNode → ModeratorNode → AgentSpeakNode → SummarizerNode → SaveNode.
All LLM calls are mocked to return canned YAML responses.
"""

from __future__ import annotations

from unittest.mock import patch

from flow import create_conversation_flow
from tests.constants import INIT_YAML_RESPONSE, MODERATOR_YAML_RESPONSE, SUMMARIZER_YAML_RESPONSE
from utils.constants import (
    TOPIC, PERSONAS, CONVERSATION, TURN, MAX_TURNS, LAST_SPEAKER,
    MODERATOR_NOTES, MODERATOR_INTERVENTIONS, NEXT_SPEAKER,
    SPEAKER_TURNS, STALL_COUNT, RESEARCH_COUNT, RESEARCH_NOTES, SUMMARY,
)


class TestFullFlow:
    """Integration tests for the complete flow graph."""

    def test_full_flow_completes(self):
        """Run the full flow with 2 turns — should complete without errors."""
        shared = {
            TOPIC: "AI Safety",
            PERSONAS: [],
            CONVERSATION: [],
            TURN: 0,
            MAX_TURNS: 2,
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

        # Mock all external calls
        with (
            patch("nodes.call_llm") as mock_llm,
            patch("nodes.call_llm_stream") as mock_stream,
            patch("nodes.search_web_raw") as mock_search,
            patch("nodes.json.dump"),
            patch("nodes.console.print"),
            patch("nodes.os.makedirs"),
            patch("nodes.os.listdir", return_value=[]),
        ):
            mock_search.return_value = "No results found."

            # Return different responses based on call order
            mock_llm.side_effect = [
                INIT_YAML_RESPONSE,          # InitNode
                MODERATOR_YAML_RESPONSE,     # ModeratorNode turn 1
                MODERATOR_YAML_RESPONSE.replace(
                    "should_end: false", "should_end: true"
                ),                             # ModeratorNode turn 2 → summarizes
                SUMMARIZER_YAML_RESPONSE,    # SummarizerNode
            ]
            mock_stream.return_value = iter(
                ["I believe ", "AI safety ", "is critical."]
            )

            flow = create_conversation_flow()
            flow.run(shared)

            # Verify the flow completed
            assert len(shared[PERSONAS]) == 4
            assert len(shared[CONVERSATION]) >= 2  # init + at least 1 agent turn
            assert shared[SUMMARY] != ""

    def test_flow_with_research_branch(self):
        """Test flow when moderator requests research."""
        shared = {
            TOPIC: "Climate Change",
            PERSONAS: [],
            CONVERSATION: [],
            TURN: 0,
            MAX_TURNS: 3,
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

        moderator_with_research = """
```yaml
loop_detected: false
repeated_argument: null
drift_detected: false
moderator_notes: null
research_needed: true
next_speaker: Jordan
should_end: false
reasoning: Fact-check needed
```
"""
        researcher_response = """
```yaml
fact_checks:
  - claim: "CO2 levels at 420ppm"
    verdict: supported
    evidence: NOAA data confirms
    source_url: https://noaa.gov
    source_quality: high
overall_confidence: high
caveat: null
```
"""

        with (
            patch("nodes.call_llm") as mock_llm,
            patch("nodes.call_llm_stream") as mock_stream,
            patch("nodes.search_web_raw") as mock_search,
            patch("nodes.json.dump"),
            patch("nodes.console.print"),
            patch("nodes.os.makedirs"),
            patch("nodes.os.listdir", return_value=[]),
        ):
            mock_search.return_value = "[1] NOAA\n    URL: https://noaa.gov\n    CO2 levels at 420ppm confirmed."
            mock_llm.side_effect = [
                INIT_YAML_RESPONSE,
                moderator_with_research,       # Moderator → research
                researcher_response,           # ResearchNode
                MODERATOR_YAML_RESPONSE,       # Back to moderator
                MODERATOR_YAML_RESPONSE.replace(
                    "should_end: false", "should_end: true"
                ),
                SUMMARIZER_YAML_RESPONSE,
            ]
            mock_stream.return_value = iter(
                ["Climate action ", "requires ", "global cooperation."]
            )

            flow = create_conversation_flow()
            flow.run(shared)

            # Research node should have added findings
            assert shared[RESEARCH_COUNT] >= 1
            assert len(shared[RESEARCH_NOTES]) >= 1
            assert shared[SUMMARY] != ""

    def test_flow_with_moderator_intervention(self):
        """Test flow when stall_count triggers moderator notes."""
        shared = {
            TOPIC: "UBI",
            PERSONAS: [],
            CONVERSATION: [],
            TURN: 0,
            MAX_TURNS: 5,
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

        moderator_looping = """
```yaml
loop_detected: true
repeated_argument: "UBI is too expensive"
drift_detected: false
moderator_notes: "What evidence would change your mind about cost?"
research_needed: false
next_speaker: Sam
should_end: false
reasoning: Cost argument repeating
```
"""

        # Simulate 4 loops (stall ≥ 3 → moderator_notes injected)
        loop_responses = [moderator_looping] * 4 + [
            MODERATOR_YAML_RESPONSE.replace(
                "should_end: false", "should_end: true"
            )
        ]

        with (
            patch("nodes.call_llm") as mock_llm,
            patch("nodes.call_llm_stream") as mock_stream,
            patch("nodes.search_web_raw") as mock_search,
            patch("nodes.json.dump"),
            patch("nodes.console.print"),
            patch("nodes.os.makedirs"),
            patch("nodes.os.listdir", return_value=[]),
        ):
            mock_search.return_value = "No results."
            mock_llm.side_effect = [INIT_YAML_RESPONSE] + loop_responses + [SUMMARIZER_YAML_RESPONSE]
            mock_stream.return_value = iter(["UBI is ", "a complex ", "topic."])

            flow = create_conversation_flow()
            flow.run(shared)

            # Moderator should have intervened at least once
            assert shared[STALL_COUNT] >= 3
            assert len(shared[MODERATOR_INTERVENTIONS]) >= 1
            assert shared[SUMMARY] != ""
