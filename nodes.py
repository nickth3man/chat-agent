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
        prompt = f"""Create 4 DEBATE personas for the topic: "{topic}"

For each persona, provide these fields:
  name: Realistic full name (vary gender, age, cultural background)
  role: Job title, 2-5 words
  perspective: Their stance on this topic (2-3 sentences). Include WHY they believe this — a value, fear, lived experience, or aspiration that anchors their view.
  reasoning_approach: One of [Evidence-first, Principle-first, Counterfactual, Systems-thinking]. Each persona must use a DIFFERENT approach.
  belief_intensity: 1-10 scale. How strongly they hold their view. Spread these out — include at least one 3-4 and one 8-9.
  communication_style: 1 sentence describing HOW they argue (e.g. "Uses statistics and case studies", "Appeals to moral principles", "Speaks from personal experience", "Challenges assumptions with hypotheticals")
  argument_tendency: 1 sentence on their go-to move (e.g. "Defaults to historical precedent", "Finds the overlooked stakeholder", "Asks what we're not measuring")

Output ONLY valid YAML:
```yaml
personas:
  - name: <full name>
    role: <2-5 word job title>
    perspective: <2-3 sentence stance>
    reasoning_approach: <Evidence-first|Principle-first|Counterfactual|Systems-thinking>
    belief_intensity: <1-10>
    communication_style: <1 sentence>
    argument_tendency: <1 sentence>
  - name: <full name>
    role: <2-5 word job title>
    perspective: <2-3 sentence stance>
    reasoning_approach: <Evidence-first|Principle-first|Counterfactual|Systems-thinking>
    belief_intensity: <1-10>
    communication_style: <1 sentence>
    argument_tendency: <1 sentence>
  - name: <full name>
    role: <2-5 word job title>
    perspective: <2-3 sentence stance>
    reasoning_approach: <Evidence-first|Principle-first|Counterfactual|Systems-thinking>
    belief_intensity: <1-10>
    communication_style: <1 sentence>
    argument_tendency: <1 sentence>
  - name: <full name>
    role: <2-5 word job title>
    perspective: <2-3 sentence stance>
    reasoning_approach: <Evidence-first|Principle-first|Counterfactual|Systems-thinking>
    belief_intensity: <1-10>
    communication_style: <1 sentence>
    argument_tendency: <1 sentence>
opening_question: <A provocative question that forces debate. Do not prefix with persona name.>
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

Analyze:
1. LOOP: Same argument exchanged 3+ times without new angle? true/false
2. CONVERGENCE: 2+ agents agree on a specific point? List them.
3. STALLED: Agent's last 2 responses nearly identical in argument? List names.
4. DRIFT: Agent made claims TOTALLY OPPOSITE to their stated perspective? Flag true. Not subtle evolution — only 180° reversal.
5. NEXT: Who speaks next? Choose from: {", ".join(speaker_names)}. Prioritize least-spoken agents whose reasoning_approach differs from {last_speaker}.
6. NOTE: Only write when loop or stall detected. Make it PROVOCATIVE — a question that forces a new angle. NEVER say "Let's examine" or "Consider". Use: "What's the strongest argument against your own position?" / "If you had to bet on the opposite outcome, what makes you nervous?" / "What evidence would change your mind?" / "Name one thing the other side gets right."

```yaml
loop_detected: true/false
drift_detected: true/false
convergence_points: []
stalled_agents: []
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
        conversation, personas, turn, max_turns, last_speaker, topic = prep_res
        candidates = [p["name"] for p in personas if p["name"] != last_speaker]
        import random
        return {
            "loop_detected": False,
            "drift_detected": False,
            "convergence_points": [],
            "stalled_agents": [],
            "moderator_notes": None,
            "next_speaker": random.choice(candidates) if candidates else personas[0]["name"],
            "should_end": turn >= max_turns,
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
