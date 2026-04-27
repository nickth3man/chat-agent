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
        prompt = f"""Create 4 debate personas for: "{topic}"

Each persona needs: name, role (2-5 word job title), perspective (2-3 sentences with belief/fear/aspiration), reasoning_approach (each DIFFERENT: Evidence-first | Principle-first | Counterfactual | Systems-thinking), belief_intensity (1-10, include one 3-4 and one 8-9), communication_style (1 sentence on HOW they argue), argument_tendency (1 sentence on their go-to move).

Output ONLY valid YAML:
```yaml
personas:
  - name: <full name>
    role: <2-5 word job title>
    perspective: <2-3 sentence stance>
    reasoning_approach: <pick one, each unique>
    belief_intensity: <1-10>
    communication_style: <1 sentence>
    argument_tendency: <1 sentence>
  - name: <full name>
    role: <...>
    perspective: <...>
    reasoning_approach: <...>
    belief_intensity: <...>
    communication_style: <...>
    argument_tendency: <...>
  - name: <...>
    role: <...>
    perspective: <...>
    reasoning_approach: <...>
    belief_intensity: <...>
    communication_style: <...>
    argument_tendency: <...>
  - name: <...>
    role: <...>
    perspective: <...>
    reasoning_approach: <...>
    belief_intensity: <...>
    communication_style: <...>
    argument_tendency: <...>
opening_question: <provocative question, no name prefix>
```"""
        response = call_llm(prompt)
        data = _parse_yaml(response)
        if "personas" not in data or "opening_question" not in data:
            raise ValueError(f"Invalid init response: {response}")
        for p in data["personas"]:
            required = ("name", "role", "perspective", "reasoning_approach", "belief_intensity", "communication_style", "argument_tendency")
            if not all(k in p for k in required):
                raise ValueError(f"Persona missing required fields: {p}")
            if len(str(p.get("role", ""))) < 2 or len(str(p.get("role", ""))) > 60:
                raise ValueError(f"Persona role must be 2-60 chars (job title): {p}")
            if len(str(p.get("perspective", ""))) < 15:
                raise ValueError(f"Persona perspective too short (<15 chars): {p}")
        return data

    def exec_fallback(self, prep_res, exc):
        return {
            "personas": [
                {"name": "Alex", "role": "Techno-Optimist", "perspective": "Technology will solve humanity's biggest problems if we embrace innovation without fear. We need less regulation, not more.", "reasoning_approach": "Evidence-first", "belief_intensity": 8, "communication_style": "Cites historical tech breakthroughs as proof of pattern", "argument_tendency": "Compares regulatory pushback to past Luddite movements"},
                {"name": "Jordan", "role": "Ethics Professor", "perspective": "I've studied how unchecked technology erodes social trust and privacy. We need strong oversight and democratic control over AI and automation.", "reasoning_approach": "Principle-first", "belief_intensity": 7, "communication_style": "Grounds arguments in ethical frameworks and historical harms", "argument_tendency": "Asks who bears the cost when things go wrong"},
                {"name": "Sam", "role": "Small Business Owner", "perspective": "When regulations pile up, small businesses suffer most. I need practical rules, not utopian ideals. Show me what works for real people.", "reasoning_approach": "Counterfactual", "belief_intensity": 5, "communication_style": "Speaks from personal experience running a business", "argument_tendency": "Tests proposals against reality: 'Would this actually work on Main Street?'"},
                {"name": "Riley", "role": "Futurist & Artist", "perspective": "We keep talking as if the future is something happening TO us. What if WE designed it instead? Let's imagine radically different systems, not just tweak the current ones.", "reasoning_approach": "Systems-thinking", "belief_intensity": 6, "communication_style": "Uses metaphor and reframing to shift perspective", "argument_tendency": "Identifies the assumption nobody is questioning"},
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
        turns = {p["name"]: 0 for p in exec_res["personas"]}
        turns[exec_res["personas"][0]["name"]] = 1
        shared["speaker_turns"] = turns

class ModeratorNode(Node):
    def prep(self, shared):
        return (
            shared["conversation"],
            shared["personas"],
            shared["turn"],
            shared["max_turns"],
            shared["last_speaker"],
            shared["topic"],
            shared.get("speaker_turns", {}),
        )

    def exec(self, prep_res):
        conversation, personas, turn, max_turns, last_speaker, topic, speaker_turns = prep_res

        personas_str = "\n".join(
            f"- {p['name']} ({p['role']}) [{p.get('reasoning_approach', 'N/A')}]: {p['perspective']}" for p in personas
)
        conv_str = _conversation_str(conversation)
        speaker_names = [p["name"] for p in personas if p["name"] != last_speaker]

        prompt = f"""Moderate this debate. Output YAML only.

TOPIC: {topic}
PERSONAS:
{personas_str}
CONVERSATION:
{conv_str}
TURN: {turn}/{max_turns}
LAST SPEAKER: {last_speaker}
SPEAKER TURNS: {speaker_turns}

1. NEXT: Pick from [{", ".join(speaker_names)}]. Choose the agent with the FEWEST turns. If tied, pick whose reasoning_approach contrasts most with {last_speaker}.
2. LOOP: Flag true ONLY if the last 3+ turns restate the SAME specific claim with nearly identical wording. Name the repeated argument. When in doubt, flag false.
3. DRIFT: Flag true ONLY for 180° reversal from stated perspective. Subtle evolution is NOT drift.
4. NOTE: If loop is true, write ONE provocative question. VARY your approach each time — do NOT reuse the same template. Rotate through: "What evidence would change your mind?" / "If you had to bet against your own position, what makes you nervous?" / "What's the strongest counter-argument you haven't addressed?" / "Name a specific situation where the other side would be right." / "What hidden assumption are you relying on?" / "What would a compromise look like that neither side has proposed yet?" — pick a DIFFERENT one each turn. Do NOT repeat the same question type twice in a row.

```yaml
loop_detected: true/false
repeated_argument: <what is repeating, or null>
drift_detected: true/false
moderator_notes: <provocative question or null>
next_speaker: <name>
should_end: true/false
reasoning: <1 line>
```"""
        response = call_llm(prompt)
        data = _parse_yaml(response)
        if "next_speaker" not in data:
            raise ValueError(f"Invalid moderator response: {response}")
        return data

    def exec_fallback(self, prep_res, exc):
        conversation, personas, turn, max_turns, last_speaker, topic, speaker_turns = prep_res
        candidates = [p["name"] for p in personas if p["name"] != last_speaker]
        import random
        return {
            "loop_detected": False,
            "repeated_argument": None,
            "drift_detected": False,
            "moderator_notes": None,
            "next_speaker": random.choice(candidates) if candidates else personas[0]["name"],
            "should_end": turn >= max_turns,
            "reasoning": "fallback random selection",
        }

    def post(self, shared, prep_res, exec_res):
        next_speaker = exec_res["next_speaker"]
        valid_names = {p["name"] for p in shared["personas"]}
        if next_speaker not in valid_names:
            other = [n for n in valid_names if n != shared["last_speaker"]]
            next_speaker = other[0] if other else shared["last_speaker"]
        shared["next_speaker"] = next_speaker
        loop_detected = exec_res.get("loop_detected", False)

        # Stall counter with hysteresis (MagenticOne pattern)
        stall = shared.get("stall_count", 0)
        if loop_detected:
            stall += 1
        else:
            stall = max(0, stall - 1)
        shared["stall_count"] = stall

        # Only pass notes when stall threshold reached (3+)
        notes = None
        if stall >= 3:
            notes = exec_res.get("moderator_notes")
        shared["moderator_notes"] = notes if notes else None

        if notes:
            shared["moderator_interventions"].append({
                "turn": shared["turn"],
                "loop_detected": loop_detected,
                "drift_detected": exec_res.get("drift_detected", False),
                "note": notes,
                "stall_count": stall,
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

        matches = [p for p in personas if p["name"] == next_speaker]
        if not matches:
            persona = personas[0]
        else:
            persona = matches[0]
        color_idx = next(i for i, p in enumerate(personas) if p["name"] == persona["name"])
        color = AGENT_COLORS[color_idx % len(AGENT_COLORS)]

        return persona, conversation, topic, moderator_notes, color

    def exec(self, prep_res):
        persona, conversation, topic, moderator_notes, color = prep_res

        conv_str = _conversation_str(conversation)
        mod_line = ""
        if moderator_notes:
            mod_line = f"\n[MODERATOR NOTE — follow this guidance: {moderator_notes}]\n"

        prompt = f"""=== PERSONA CARD ===
You are {persona['name']}, {persona['role']}.
Stance: {persona['perspective']}
How you argue: {persona.get('communication_style', 'Direct and plainspoken')}
Your go-to move: {persona.get('argument_tendency', 'Draws from personal experience')}
Belief intensity: {persona.get('belief_intensity', 5)}/10
Reasoning approach: {persona.get('reasoning_approach', 'Context-driven')}
=== END CARD ===

Speak in character — 1-3 paragraphs. Before writing, run this self-check:
1. NOVELTY: Am I adding a NEW angle, story, metaphor, or counter-argument? If I'm about to echo, I pivot.
2. WHY ME: What does MY specific background bring that no one else here can say? Lead with that.
3. WEAK SPOT: Is there a vulnerability in the last argument I can probe — not attack, but honestly question?

Output only your message. No name prefix. No bullet points. No sign-offs.
Vary sentence rhythm — short. Then longer, building. Then short again.

TOPIC: {topic}

CONVERSATION:
{conv_str}
{mod_line}
YOUR THOUGHTS:"""
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
        if "speaker_turns" in shared:
            shared["speaker_turns"][persona["name"]] = shared["speaker_turns"].get(persona["name"], 0) + 1
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

        prompt = f"""Analyze and summarize this multi-agent debate. Focus on substance, not speakers.

TOPIC: {topic}
PERSONAS:
{personas_str}
CONVERSATION:
{conv_str}

=== DEBATE SUMMARY ===

## Core Positions
For each agent, state their opening stance and how it held up or shifted:

## Convergence Points
Where 2+ agents agreed or found common ground. If none, say "No consensus emerged."

## Key Divergences (3 max)
The most significant unresolved disagreements. For each: the claim, the competing positions, and the core reason they couldn't reconcile. Focus on IDEAS, not personalities.

## Novel Insights
What emerged from the DEBATE ITSELF — ideas not present in anyone's opening stance. If the debate generated nothing new, say so honestly.

## Surprising Moments
1-2 quotes or moments that shifted the debate's direction or reframed the question.

## The Arc
How positions evolved from start to finish. Did anyone concede? Did debate reveal a deeper question beneath the surface?

## Overall Takeaway
2-3 sentences. What does this debate tell us about the topic that a single perspective would miss?"""
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
