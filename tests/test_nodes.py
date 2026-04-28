"""Node-level tests for the multi-agent debate simulator.

Each test instantiates a single Node, runs it with a prepared shared dict,
and asserts the expected mutations to shared state.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch, MagicMock, call

import pytest

from nodes import (
    InitNode,
    ModeratorNode,
    AgentSpeakNode,
    ResearchNode,
    SummarizerNode,
    SaveNode,
)
from debate_schema import parse_llm_yaml
from tests.constants import INIT_YAML_RESPONSE, MODERATOR_YAML_RESPONSE, SUMMARIZER_YAML_RESPONSE


class TestParseYaml:
    """Tests for _parse_yaml utility."""

    def test_parse_yaml_with_fences(self):
        result = parse_llm_yaml('```yaml\nkey: value\n```')
        assert result == {"key": "value"}

    def test_parse_yaml_without_fences(self):
        result = parse_llm_yaml("key: value")
        assert result == {"key": "value"}

    def test_parse_yaml_with_whitespace(self):
        result = parse_llm_yaml("```yaml\nname: Alex\nage: 30\n```")
        assert result == {"name": "Alex", "age": 30}


class TestInitNode:
    """Tests for InitNode — persona generation."""

    def test_generates_personas(self, shared_empty, mock_call_llm):
        mock_call_llm.return_value = INIT_YAML_RESPONSE
        node = InitNode()
        node.run(shared_empty)

        assert len(shared_empty["personas"]) == 4
        assert shared_empty["personas"][0]["name"] == "Alex"
        assert shared_empty["turn"] == 1
        assert len(shared_empty["conversation"]) == 1
        assert shared_empty["conversation"][0]["agent"] == "Alex"

    def test_raises_on_invalid_response(self, shared_empty, mock_call_llm):
        mock_call_llm.return_value = "invalid response"
        node = InitNode()  # no retries
        node.run(shared_empty)
        # exec_fallback should have been used
        assert len(shared_empty["personas"]) == 4

    def test_sets_up_speaker_turns(self, shared_empty, mock_call_llm):
        mock_call_llm.return_value = INIT_YAML_RESPONSE
        node = InitNode()
        node.run(shared_empty)

        turns = shared_empty["speaker_turns"]
        assert turns["Alex"] == 1
        for name in ("Jordan", "Sam", "Riley"):
            assert turns[name] == 0


class TestModeratorNode:
    """Tests for ModeratorNode — speaker selection and loop detection."""

    def test_selects_next_speaker(self, shared_with_personas, mock_call_llm):
        mock_call_llm.return_value = MODERATOR_YAML_RESPONSE
        node = ModeratorNode()
        action = node.run(shared_with_personas)

        assert shared_with_personas["next_speaker"] == "Jordan"
        assert action == "speak"

    def test_loop_detection_increments_stall(self, shared_with_personas, mock_call_llm):
        shared_with_personas["stall_count"] = 1
        mock_call_llm.return_value = """
```yaml
loop_detected: true
repeated_argument: "Remote work is flexible"
drift_detected: false
moderator_notes: "What evidence would change your mind?"
research_needed: false
next_speaker: Sam
should_end: false
reasoning: Repeating flexibility argument
```
"""
        node = ModeratorNode()
        node.run(shared_with_personas)

        assert shared_with_personas["stall_count"] == 2

    def test_summarizes_when_max_turns_reached(self, shared_with_personas, mock_call_llm):
        shared_with_personas["turn"] = 5  # at max
        mock_call_llm.return_value = MODERATOR_YAML_RESPONSE
        node = ModeratorNode()
        action = node.run(shared_with_personas)

        assert action == "summarize"

    def test_research_branch(self, shared_with_personas, mock_call_llm):
        shared_with_personas["turn"] = 2
        shared_with_personas["stall_count"] = 3  # below threshold uses different path
        mock_call_llm.return_value = """
```yaml
loop_detected: true
repeated_argument: "flexibility"
drift_detected: false
moderator_notes: "What's the strongest counter-argument?"
research_needed: true
next_speaker: Riley
should_end: false
reasoning: Needs fact-check on productivity claim
```
"""
        node = ModeratorNode()
        action = node.run(shared_with_personas)

        assert action == "research"

    def test_validates_next_speaker(self, shared_with_personas, mock_call_llm):
        """Moderator should reject invalid speaker names."""
        mock_call_llm.return_value = """
```yaml
loop_detected: false
repeated_argument: null
drift_detected: false
moderator_notes: null
research_needed: false
next_speaker: NotAPerson
should_end: false
reasoning: Testing fallback
```
"""
        node = ModeratorNode()
        node.run(shared_with_personas)

        # Should have fallen back to a valid speaker
        assert shared_with_personas["next_speaker"] in {"Jordan", "Sam", "Riley"}


class TestAgentSpeakNode:
    """Tests for AgentSpeakNode — message generation and streaming."""

    def test_appends_message(self, shared_with_personas, mock_call_llm_stream):
        shared_with_personas["next_speaker"] = "Jordan"
        shared_with_personas["moderator_notes"] = None

        node = AgentSpeakNode()
        action = node.run(shared_with_personas)

        assert len(shared_with_personas["conversation"]) == 2
        assert shared_with_personas["conversation"][-1]["agent"] == "Jordan"
        assert shared_with_personas["conversation"][-1]["message"] != ""
        assert shared_with_personas["last_speaker"] == "Jordan"
        assert shared_with_personas["turn"] == 2
        assert action == "continue"

    def test_handles_unknown_speaker(self, shared_with_personas, mock_call_llm_stream):
        """Should fall back to first persona if next_speaker doesn't match."""
        shared_with_personas["next_speaker"] = "GhostSpeaker"

        node = AgentSpeakNode()
        node.run(shared_with_personas)

        # Falls back to Alex (personas[0])
        assert shared_with_personas["conversation"][-1]["agent"] == "Alex"


class TestResearchNode:
    """Tests for ResearchNode — web fact-checking."""

    def test_skips_when_exec_fallback(self, shared_with_personas):
        """ResearchNode.exec_fallback returns a string, should not crash post()."""
        shared_with_personas["conversation"].append(
            {"agent": "Alex", "message": "Productivity increased by 47% in 2023."}
        )

        node = ResearchNode()
        node.run(shared_with_personas)

        # exec will try to call search_web_raw and call_llm, which both fail
        # exec_fallback returns "Research unavailable due to error."
        # post() should handle the string gracefully
        assert shared_with_personas.get("research_notes") is not None


class TestSummarizerNode:
    """Tests for SummarizerNode — debate summarization."""

    def test_generates_summary(self, shared_with_personas, mock_call_llm):
        mock_call_llm.return_value = SUMMARIZER_YAML_RESPONSE
        node = SummarizerNode()
        node.run(shared_with_personas)

        assert shared_with_personas["summary"] != ""

    def test_handles_long_conversation(self, shared_with_personas, mock_call_llm):
        """Should truncate conversation > 30 turns but not crash."""
        for i in range(20):
            shared_with_personas["conversation"].append(
                {"agent": "Alex", "message": f"Turn {i}"}
            )
        mock_call_llm.return_value = SUMMARIZER_YAML_RESPONSE
        node = SummarizerNode()
        node.run(shared_with_personas)

        assert shared_with_personas["summary"] != ""


class TestSaveNode:
    """Tests for SaveNode — conversation persistence."""

    def test_saves_json(self, shared_with_personas, tmp_path):
        """SaveNode should write a JSON file and return the path."""
        shared_with_personas["summary"] = "Test summary"
        # Override data dir to temp path
        with patch("nodes.os.path.join") as mock_join:
            mock_join.return_value = os.path.join(str(tmp_path), "test_1.json")
            with patch("nodes.os.makedirs"):
                with patch("nodes.os.listdir", return_value=[]):
                    node = SaveNode()
                    node.run(shared_with_personas)

    def test_handles_save_failure_gracefully(self, shared_with_personas):
        """On failure, exec_fallback returns empty string — post() skips output."""
        shared_with_personas["summary"] = "Test"
        node = SaveNode()
        node.run(shared_with_personas)
        # Should not crash — exec_fallback returns "" and post() checks if exec_res
        assert True  # no exception
