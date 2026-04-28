"""Test fixtures, mocks, and logging infrastructure for chat-agent tests."""
from __future__ import annotations

import json
import logging
import time
from unittest.mock import patch, MagicMock

import pytest

from utils.constants import (
    TOPIC, PERSONAS, CONVERSATION, TURN, MAX_TURNS, LAST_SPEAKER,
    MODERATOR_NOTES, MODERATOR_INTERVENTIONS, NEXT_SPEAKER,
    SPEAKER_TURNS, STALL_COUNT, RESEARCH_COUNT, RESEARCH_NOTES, SUMMARY,
)

logger = logging.getLogger(__name__)

# ── Shared store fixtures ──────────────────────────────────────────

@pytest.fixture
def shared_empty() -> dict[str, object]:
    """Fresh shared store with all required keys."""
    return {
        TOPIC: "AI Safety",
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


@pytest.fixture
def shared_with_personas() -> dict[str, object]:
    """Shared store populated with 4 personas and an opening message."""
    return {
        TOPIC: "Remote Work",
        PERSONAS: [
            {
                "name": "Alex",
                "role": "Productivity Expert",
                "perspective": "Remote work boosts productivity through focused time and flexibility.",
                "reasoning_approach": "Evidence-first",
                "belief_intensity": 8,
                "communication_style": "Cites studies and data",
                "argument_tendency": "Compares metrics side by side",
            },
            {
                "name": "Jordan",
                "role": "Organizational Psychologist",
                "perspective": "Isolation from remote work damages team cohesion and innovation.",
                "reasoning_approach": "Principle-first",
                "belief_intensity": 7,
                "communication_style": "Uses psychological frameworks",
                "argument_tendency": "Asks about human cost",
            },
            {
                "name": "Sam",
                "role": "Small Business Owner",
                "perspective": "Hybrid work is a compromise that gives the worst of both worlds.",
                "reasoning_approach": "Counterfactual",
                "belief_intensity": 5,
                "communication_style": "Practical real-world examples",
                "argument_tendency": "Tests ideas against reality",
            },
            {
                "name": "Riley",
                "role": "Futurist",
                "perspective": "We should redesign work entirely, not just debate location.",
                "reasoning_approach": "Systems-thinking",
                "belief_intensity": 6,
                "communication_style": "Uses metaphor and reframing",
                "argument_tendency": "Identifies hidden assumptions",
            },
        ],
        CONVERSATION: [
            {"agent": "Alex", "message": "Is remote work better or worse for society?"}
        ],
        TURN: 1,
        MAX_TURNS: 5,
        LAST_SPEAKER: "Alex",
        MODERATOR_NOTES: None,
        MODERATOR_INTERVENTIONS: [],
        NEXT_SPEAKER: None,
        SPEAKER_TURNS: {"Alex": 1, "Jordan": 0, "Sam": 0, "Riley": 0},
        STALL_COUNT: 0,
        RESEARCH_COUNT: 0,
        RESEARCH_NOTES: [],
        SUMMARY: "",
    }


# ── Mock fixtures ──────────────────────────────────────────────────

@pytest.fixture
def mock_call_llm():
    """Patch call_llm to return canned YAML responses."""
    with patch("nodes.call_llm") as mock:
        yield mock


@pytest.fixture
def mock_call_llm_stream():
    """Patch call_llm_stream to yield a canned response."""
    with patch("nodes.call_llm_stream") as mock:
        mock.return_value = iter(["Remote work offers ", "unprecedented flexibility."])
        yield mock


@pytest.fixture
def mock_search_web():
    """Patch search_web_raw to return canned search results."""
    with patch("nodes.search_web_raw") as mock:
        mock.return_value = "[1] Test Result\n    URL: https://example.com\n    No relevant data found.\n"
        yield mock


# ── Test observability ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _log_test_boundary(request):
    """Log test start/end with timing. Runs for every test automatically."""
    test_name = f"{request.module.__name__}::{request.node.name}"
    logger.info("▶ TEST START: %s", test_name)
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info("■ TEST END: %s (%.3fs)", test_name, elapsed)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """On failure, dump shared store fixtures to log for debugging."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        # Dump any shared store fixture the test had access to
        for fixture_name in ("shared_empty", "shared_with_personas", "shared"):
            if fixture_name in item.funcargs:
                store = item.funcargs[fixture_name]
                logger.error(
                    "✗ FAILED %s — shared store snapshot:\n%s",
                    item.nodeid,
                    json.dumps(_summarize_shared(store), indent=2, default=str),
                )
                break

        # Log mock call counts if available
        for fixture_name in ("mock_call_llm", "mock_call_llm_stream", "mock_search_web"):
            if fixture_name in item.funcargs:
                mock_obj = item.funcargs[fixture_name]
                logger.error(
                    "  mock %s: call_count=%d",
                    fixture_name, getattr(mock_obj, "call_count", "N/A"),
                )


def _summarize_shared(store: dict) -> dict:
    """Produce a test-friendly summary of the shared store (truncates large values)."""
    summary = {}
    for k, v in store.items():
        if isinstance(v, list):
            summary[k] = f"[{len(v)} items]"
        elif isinstance(v, dict):
            summary[k] = f"{{dict with {len(v)} keys}}"
        elif isinstance(v, str) and len(v) > 100:
            summary[k] = v[:100] + "..."
        else:
            summary[k] = v
    return summary


def pytest_assertrepr_compare(op, left, right):
    """Rich diff output when dict assertions fail."""
    if isinstance(left, dict) and isinstance(right, dict) and op == "==":
        lines = ["Dict comparison (differences only):"]
        all_keys = set(left) | set(right)
        for k in sorted(all_keys):
            lv, rv = left.get(k, "<MISSING>"), right.get(k, "<MISSING>")
            if lv != rv:
                lines.append(f"  {k}:")
                lines.append(f"    left:  {lv!r}")
                lines.append(f"    right: {rv!r}")
        return lines if len(lines) > 1 else None
