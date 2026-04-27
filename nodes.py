import os
import json
import re
from datetime import datetime, timezone

import yaml
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from pocketflow import Node
from utils.call_llm import call_llm
from utils.call_llm_stream import call_llm_stream

AGENT_COLORS = ["cyan", "magenta", "yellow", "green"]
console = Console()


def _parse_yaml(text: str) -> dict:
    match = re.search(r"```yaml\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return yaml.safe_load(match.group(1))
    return yaml.safe_load(text)


def _conversation_str(conversation: list) -> str:
    lines = []
    for msg in conversation:
        lines.append(f"{msg['agent']}: {msg['message']}")
    return "\n\n".join(lines)


class InitNode(Node):
    def prep(self, shared):
        return shared["topic"]

    def exec(self, topic):
        prompt = f"""Generate 4 random personas for a conversation. Each persona should have a name, role, and perspective. The personas must be diverse, creative, and engaging — NOT tied to any specific topic.

Also, generate a thought-provoking 1-2 sentence opening QUESTION from the first persona about this topic: "{topic}"

Output in YAML:
```yaml
personas:
  - name: <name>
    role: <role>
    perspective: <perspective>
  - name: <name>
    role: <role>
    perspective: <perspective>
  - name: <name>
    role: <role>
    perspective: <perspective>
  - name: <name>
    role: <role>
    perspective: <perspective>
opening_question: <question from persona 0, do NOT prefix with the name>
```"""
        response = call_llm(prompt)
        data = _parse_yaml(response)
        if "personas" not in data or "opening_question" not in data:
            raise ValueError(f"Invalid init response: {response}")
        for p in data["personas"]:
            if not all(k in p for k in ("name", "role", "perspective")):
                raise ValueError(f"Persona missing required fields: {p}")
        return data

    def exec_fallback(self, prep_res, exc):
        return {
            "personas": [
                {"name": "Alex", "role": "technologist", "perspective": "optimistic"},
                {"name": "Jordan", "role": "philosopher", "perspective": "skeptical"},
                {"name": "Sam", "role": "pragmatist", "perspective": "practical"},
                {"name": "Riley", "role": "visionary", "perspective": "idealistic"},
            ],
            "opening_question": f"What do you think about {prep_res}?",
        }

    def post(self, shared, prep_res, exec_res):
        shared["personas"] = exec_res["personas"]
        shared["conversation"] = [
            {"agent": exec_res["personas"][0]["name"], "message": exec_res["opening_question"]}
        ]
        shared["turn"] = 1
        shared["last_speaker"] = exec_res["personas"][0]["name"]
        shared["moderator_notes"] = None
        shared["moderator_interventions"] = []


class ModeratorNode(Node):
    def prep(self, shared):
        return (
            shared["conversation"],
            shared["personas"],
            shared["turn"],
            shared["max_turns"],
            shared["last_speaker"],
            shared["topic"],
        )

    def exec(self, prep_res):
        conversation, personas, turn, max_turns, last_speaker, topic = prep_res

        personas_str = "\n".join(
            f"- {p['name']} ({p['role']}): {p['perspective']}" for p in personas
        )
        conv_str = _conversation_str(conversation)
        speaker_names = [p["name"] for p in personas if p["name"] != last_speaker]

        prompt = f"""You are a strict conversation moderator. Your job is to prevent the conversation from becoming repetitive or going in circles.

TOPIC: {topic}

PERSONAS:
{personas_str}

FULL CONVERSATION HISTORY:
{conv_str}

CURRENT TURN: {turn}/{max_turns}
LAST SPEAKER: {last_speaker}

CRITICAL ANALYSIS:
1. LOOP DETECTION: Compare the last 5+ messages. Are they saying the same things with different words? Are the same keywords, metaphors, or phrases being recycled? If the conversation feels stagnant or circular, set loop_detected: true and provide a moderator_notes that forcefully redirects to a COMPLETELY NEW angle, subtopic, or question about the topic.
2. DRIFT DETECTION: Are any agents saying things completely inconsistent with their persona's role or perspective? If so, set drift_detected: true.
3. NEXT SPEAKER: Pick who speaks next. Must NOT be {last_speaker}. Choose from: {", ".join(speaker_names)}. Favor agents who have spoken less OR whose unique perspective would add the most FRESH value.

Output ONLY valid YAML:
```yaml
loop_detected: true/false
drift_detected: true/false
moderator_notes: <one specific sentence redirecting to a new angle or subtopic, or null>
next_speaker: <name>
reasoning: <brief reason>
```"""
        response = call_llm(prompt)
        data = _parse_yaml(response)
        if "next_speaker" not in data:
            raise ValueError(f"Invalid moderator response: {response}")
        return data

    def exec_fallback(self, prep_res, exc):
        conversation, personas, turn, max_turns, last_speaker, topic = prep_res
        candidates = [p["name"] for p in personas if p["name"] != last_speaker]
        import random
        return {
            "loop_detected": False,
            "drift_detected": False,
            "moderator_notes": None,
            "next_speaker": random.choice(candidates) if candidates else personas[0]["name"],
            "reasoning": "fallback random selection",
        }

    def post(self, shared, prep_res, exec_res):
        shared["next_speaker"] = exec_res["next_speaker"]
        notes = exec_res.get("moderator_notes")
        shared["moderator_notes"] = notes if notes else None

        if notes:
            shared["moderator_interventions"].append({
                "turn": shared["turn"],
                "loop_detected": exec_res.get("loop_detected", False),
                "drift_detected": exec_res.get("drift_detected", False),
                "note": notes,
            })

        if shared["turn"] < shared["max_turns"]:
            return "speak"
        return "summarize"


class AgentSpeakNode(Node):
    def prep(self, shared):
        next_speaker = shared["next_speaker"]
        personas = shared["personas"]
        conversation = shared["conversation"]
        topic = shared["topic"]
        moderator_notes = shared.get("moderator_notes")

        persona = next(p for p in personas if p["name"] == next_speaker)
        color_idx = next(i for i, p in enumerate(personas) if p["name"] == next_speaker)
        color = AGENT_COLORS[color_idx % len(AGENT_COLORS)]

        return persona, conversation, topic, moderator_notes, color

    def exec(self, prep_res):
        persona, conversation, topic, moderator_notes, color = prep_res

        conv_str = _conversation_str(conversation)
        mod_line = ""
        if moderator_notes:
            mod_line = f"\n[MODERATOR NOTE — follow this guidance: {moderator_notes}]\n"

        prompt = f"""INSTRUCTION:
You are {persona['name']}, {persona['role']}. Your perspective: {persona['perspective']}.
Write a 1-3 paragraph response in character. Be conversational. Engage directly with what previous speakers said. Do NOT repeat points already made. Do NOT prefix with your name. Do NOT include meta-commentary, instructions, or labels in your response — output ONLY the message content.

TOPIC: {topic}

CONVERSATION SO FAR:
{conv_str}
{mod_line}
YOUR RESPONSE (as {persona['name']}, message content only):"""
        full_text = ""

        with Live(
            Panel("", title=persona["name"], border_style=color),
            console=console,
            refresh_per_second=15,
            transient=False,
        ) as live:
            gen = call_llm_stream(prompt)
            while True:
                try:
                    chunk = next(gen)
                    full_text += chunk
                    live.update(
                        Panel(full_text, title=persona["name"], border_style=color)
                    )
                except StopIteration:
                    break

        return full_text

    def exec_fallback(self, prep_res, exc):
        persona = prep_res[0]
        return f"[{persona['name']} is unable to respond right now. Error: {exc}]"

    def post(self, shared, prep_res, exec_res):
        persona = prep_res[0]
        shared["conversation"].append({
            "agent": persona["name"],
            "message": exec_res,
        })
        shared["last_speaker"] = persona["name"]
        shared["moderator_notes"] = None
        shared["turn"] += 1
        return "continue"


class SummarizerNode(Node):
    def prep(self, shared):
        return shared["conversation"], shared["personas"], shared["topic"]

    def exec(self, prep_res):
        conversation, personas, topic = prep_res

        personas_str = "\n".join(
            f"- {p['name']} ({p['role']}): {p['perspective']}" for p in personas
        )

        truncated = conversation
        if len(conversation) > 30:
            truncated = (
                conversation[:10]
                + [{"agent": "...", "message": "[... middle of conversation omitted for summary ...]"}]
                + conversation[-15:]
            )
        conv_str = _conversation_str(truncated)

        if len(conv_str) > 8000:
            conv_str = conv_str[:8000] + "\n[... truncated for length ...]"

        prompt = f"""Summarize the following multi-agent conversation.

TOPIC: {topic}

PERSONAS:
{personas_str}

CONVERSATION (first 10 and last 15 messages):
{conv_str}

Provide a structured summary:
- Key takeaways (3-5 bullet points)
- Points of agreement among the agents
- Points of disagreement
- Notable insights or surprising moments
- Overall arc and conclusion of the conversation"""
        return call_llm(prompt)

    def exec_fallback(self, prep_res, exc):
        return "Summary could not be generated due to an error."

    def post(self, shared, prep_res, exec_res):
        shared["summary"] = exec_res
        console.print()
        console.print(
            Panel(exec_res, title="Conversation Summary", border_style="bold blue")
        )


class SaveNode(Node):
    def prep(self, shared):
        return {
            "topic": shared["topic"],
            "personas": shared["personas"],
            "conversation": shared["conversation"],
            "turn_count": shared["turn"],
            "moderator_interventions": shared.get("moderator_interventions", []),
            "summary": shared["summary"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def exec(self, data):
        slug = re.sub(r"[^a-z0-9]+", "-", data["topic"].lower()).strip("-")
        dir_path = os.path.join("data", "conversations")
        os.makedirs(dir_path, exist_ok=True)

        existing = [
            f for f in os.listdir(dir_path)
            if f.startswith(f"{slug}_") and f.endswith(".json")
        ]
        run_number = len(existing) + 1
        filename = f"{slug}_{run_number}.json"
        filepath = os.path.join(dir_path, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return filepath

    def exec_fallback(self, prep_res, exc):
        return None

    def post(self, shared, prep_res, exec_res):
        if exec_res:
            console.print(f"\n[dim]Conversation saved to {exec_res}[/dim]")
