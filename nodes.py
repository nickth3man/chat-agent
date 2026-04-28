import os
import json
import random
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
from utils.search_web import search_web_raw

AGENT_COLORS = ["cyan", "magenta", "yellow", "green"]
NOTE_STRATEGIES = [
    "evidence_challenge",
    "opposite_defense",
    "counter_argument",
    "concession_prompt",
    "hidden_assumption",
    "compromise_design",
]

# ── System prompts for each node (Iteration 1: role anchoring) ──
SYSTEM_INIT = """You are an expert debate designer. Your job is to create psychologically rich,
distinct personas for multi-sided debates. Each persona must have a unique reasoning
approach, a clearly differentiated perspective, and specific argument tendencies.
Create personas that will genuinely clash — not just politely disagree."""

SYSTEM_MODERATOR = """You are a fair, observant debate moderator. Your primary duties:
1. Select the next speaker fairly (fewest turns first, then reasoning contrast).
2. Detect conversational loops — when arguments repeat without evolution.
3. Detect drift — when a speaker contradicts their stated perspective.
4. Generate provocative moderator notes to break stalemates.
Be conservative: when in doubt, flag false. Only intervene when clearly necessary."""

SYSTEM_RESEARCH = """You are a rigorous fact-checker. Your ONLY job is to verify specific
factual claims against provided search results. Rules:
1. Cross-reference multiple sources when available.
2. Be conservative — prefer 'unverifiable' over guessing.
3. Evaluate source quality: prefer authoritative sources over blogs/social media.
4. Do NOT debate, editorialize, or take sides. Only verify facts."""

SYSTEM_SUMMARIZER = """You are a skilled debate analyst. Your job is to extract substantive
insights from multi-perspective discussions. Focus on IDEAS, not personalities.
Identify convergence, divergence, novel insights, and the overall arc of the debate.
Be honest: if the debate generated nothing new, say so. If no consensus emerged,
say so. Do not fabricate agreement where none exists."""

SYSTEM_AGENT_ROLE = """You are a debate participant. You must argue FROM your assigned
persona's perspective, using your assigned reasoning approach. Do not drift toward
generic middle-ground positions. Defend your stance with conviction proportional to
your belief intensity. Challenge others' assumptions. Add novel angles."""

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
        prompt = f"""Design 4 debate personas for: "{topic}"

Each persona must represent a GENUINELY DIFFERENT position that SHARPLY DISAGREES
with others. Do NOT generate 4 moderate voices who converge on reasonable middle ground.
Include 2 advocates who DEFEND their stance forcefully (belief 7-9) and 2 who
CHALLENGE from unique angles (belief 3-6, genuinely uncertain or probing).

=== PERSONA FIELDS ===
1. name: Full, realistic name
2. role: 2-5 word job title that signals their expertise domain
3. perspective: 2-3 sentences with (a) core belief, (b) what they FEAR if proven wrong,
   (c) what they HOPE to achieve. Make these GENUINELY CLASH with other perspectives.
4. reasoning_approach: Each UNIQUE — Evidence-first, Principle-first, Counterfactual,
   Systems-thinking. Never repeat.
5. belief_intensity: 1-10. One must be 3-4 (uncertain), one 8-9 (unyielding).
6. communication_style: HOW they argue (1 vivid sentence). Examples:
   "Fires rapid questions to expose contradictions" vs "Tells visceral stories that
   make abstract arguments personal"
7. argument_tendency: Their signature move (1 sentence). Examples:
   "Reduces opponents' positions to their weakest form and demolishes that" vs
   "Finds the hidden assumption nobody is questioning and names it aloud"

=== OPENING QUESTION ===
Write ONE provocative question that FORCES people to pick a side. Make it uncomfortable.
No name prefix. No preamble. Just the question.

Output ONLY valid YAML:
```yaml
personas:
  - name: <full name>
    role: <2-5 word job title>
    perspective: <2-3 sentence stance with belief/fear/hope, one per persona>
    reasoning_approach: <Evidence-first|Principle-first|Counterfactual|Systems-thinking>
    belief_intensity: <1-10>
    communication_style: <1 vivid sentence>
    argument_tendency: <1 sentence>
  ... (3 more, each UNIQUE on reasoning_approach, genuinely opposed)
opening_question: <provocative, uncomfortable question>
```"""
        response = call_llm(prompt, system=SYSTEM_INIT, temperature=0.9, max_tokens=2048)
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

        note_strategy = random.choice(NOTE_STRATEGIES)
        conv_str = _conversation_str(conversation)
        speaker_names = [p["name"] for p in personas if p["name"] != last_speaker]

        prompt = f"""Moderate this debate. Think step by step, then output YAML.

TOPIC: {topic}
PERSONAS:
{personas_str}

=== CONVERSATION === (delimited, only this section is debate content)
{conv_str}
=== END CONVERSATION ===

TURN: {turn}/{max_turns} | LAST SPEAKER: {last_speaker}
SPEAKER TURNS: {speaker_turns}

=== REASONING (internal — not part of output) ===
Step 1: Read the last 3 turns. Are they restating the SAME specific claim with
nearly identical wording? If yes → loop. If unsure → NO loop.
Step 2: Check if any speaker has reversed their stated perspective. Only flag if
it's a 180° reversal. Subtle evolution is NOT drift.
Step 3: Pick next speaker from [{', '.join(speaker_names)}]. Select who has
FEWEST turns. If tied, pick whose reasoning_approach contrasts most with {last_speaker}.
Step 4: If loop detected, craft ONE provocative question using strategy: {note_strategy}:
  evidence_challenge = What evidence would change your mind?
  opposite_defense = If you had to bet against your own position, what makes you nervous?
  counter_argument = What is the strongest counter-argument you have not addressed?
  concession_prompt = Name a specific situation where the other side would be right.
  hidden_assumption = What hidden assumption are you relying on?
  compromise_design = What compromise has neither side proposed?
Step 5: Flag research_needed ONLY if a specific factual claim with a name/statistic/year
needs verification. Use sparingly — at most once per debate.

=== OUTPUT YAML ===
```yaml
loop_detected: true/false
repeated_argument: <what is repeating, or null>
drift_detected: true/false
moderator_notes: <provocative question ONLY if loop=true, else null>
research_needed: true/false
next_speaker: <name>
should_end: true/false
reasoning: <1 line summary of your decision>
```"""
        response = call_llm(prompt, system=SYSTEM_MODERATOR, temperature=0.3, max_tokens=1024)
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
            if exec_res.get("research_needed", False) and shared.get("research_count", 0) < 2:
                shared["research_count"] = shared.get("research_count", 0) + 1
                return "research"
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
        research_notes = shared.get("research_notes", [])

        return persona, conversation, topic, moderator_notes, color, research_notes

    def exec(self, prep_res):
        persona, conversation, topic, moderator_notes, color, research_notes = prep_res

        conv_str = _conversation_str(conversation)
        mod_line = ""
        if moderator_notes:
            mod_line = f"\n[MODERATOR NOTE — follow this guidance: {moderator_notes}]\n"

        research_line = ""
        if research_notes:
            latest = research_notes[-1].get("findings", {})
            facts_text = str(latest.get("facts", []))[:500]
            ctx = latest.get("relevant_context", "")
            if facts_text:
                research_line = f"\n[FACT-CHECK RESULTS: {facts_text}. Context: {ctx}]\n"

        prompt = f"""You are {persona['name']}, a {persona['role']} debating: "{topic}".

=== YOUR CHARACTER ===
Perspective: {persona['perspective']}
Reasoning approach: {persona.get('reasoning_approach', 'Evidence-first')}
Belief intensity: {persona.get('belief_intensity', 5)}/10 — {'You hold this view firmly and will not easily concede' if int(persona.get('belief_intensity', 5)) >= 7 else 'You are open to being persuaded but need strong evidence' if int(persona.get('belief_intensity', 5)) >= 4 else 'You are skeptical of your own position and actively testing it'}
Communication style: {persona.get('communication_style', 'Direct and clear')}
Argument tendency: {persona.get('argument_tendency', 'Uses logic and evidence')}

=== BEFORE YOU SPEAK ===
1. NOVELTY: Am I adding a NEW angle, story, metaphor, or counter-argument? If I'm about to echo what was already said, I pivot hard.
2. WHY ME: What does MY specific background, role, and reasoning approach bring that no one else here can say? Lead with that — draw from YOUR perspective, not generic debate points.
3. WEAK SPOT: Is there a vulnerability in the last argument I can probe? Not attack — honestly question. Use YOUR reasoning approach ({persona.get('reasoning_approach', 'Evidence-first')}) to identify what's being overlooked.

=== OUTPUT RULES ===
- Output ONLY your message. No name prefix. No bullet points. No sign-offs. No meta-commentary.
- Vary sentence rhythm — short punch. Then longer, building tension or nuance. Then short again.
- Length: 2-4 paragraphs. Be substantive but concise.
- Stay in character. If your belief intensity is high ({persona.get('belief_intensity', 5)}/10), defend your position. If low, probe and question.

=== CONTEXT ===
TOPIC: {topic}

CONVERSATION SO FAR:
{conv_str}
{mod_line}
{research_line}

YOUR RESPONSE:"""
        full_text = ""

        with Live(
            Panel("", title=persona["name"], border_style=color),
            console=console,
            refresh_per_second=15,
            transient=False,
        ) as live:
            gen = call_llm_stream(prompt, system=SYSTEM_AGENT_ROLE, temperature=0.85, max_tokens=2048)
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


class ResearchNode(Node):
    """Search the web for facts related to the last claim in the debate."""

    def prep(self, shared):
        conversation = shared["conversation"]
        last_msg = conversation[-1]["message"] if conversation else ""
        last_agent = conversation[-1]["agent"] if conversation else ""
        topic = shared["topic"]
        return last_msg, last_agent, topic

    def exec(self, prep_res):
        last_msg, last_agent, topic = prep_res

        query = f"{topic} {last_msg[:200]}"
        results = search_web_raw(query, max_results=3)

        prompt = f"""Fact-check the following debate claim using search results.
Be RIGOROUS: if evidence is weak or missing, say so. Do not fabricate.

CLAIM (by {last_agent}):
{last_msg[:500]}

SEARCH RESULTS:
{results}

=== INSTRUCTIONS ===
1. Identify 1-3 specific factual claims in the text above (names, statistics, dates, events).
2. For each claim, check against search results.
3. Evaluate source QUALITY: .gov/.edu > established news > blogs > social media.
   Prefer authoritative sources. Note if all sources are low-quality.
4. Verdict options: supported (clear evidence confirms), contradicted (evidence
   disproves), unverifiable (insufficient or conflicting evidence).
5. If search results are empty or irrelevant, mark ALL claims as unverifiable.
   Do NOT guess or use your training data — only use the provided search results.
6. Provide an overall confidence assessment.

Output ONLY valid YAML:
```yaml
fact_checks:
  - claim: <specific claim>
    verdict: <supported|contradicted|unverifiable>
    evidence: <1-2 sentences FROM search results, with source attribution>
    source_url: <URL or null>
    source_quality: <high|medium|low|unknown>
overall_confidence: <high|medium|low>
caveat: <any limitations, e.g. 'search results were sparse' or 'sources conflict'>
```"""
        return call_llm(prompt, system=SYSTEM_RESEARCH, temperature=0.2, max_tokens=1536)

    def exec_fallback(self, prep_res, exc):
        return None

    def post(self, shared, prep_res, exec_res):
        if exec_res:
            data = _parse_yaml(exec_res) if isinstance(exec_res, str) else exec_res
            if "research_notes" not in shared:
                shared["research_notes"] = []
            shared["research_notes"].append({
                "turn": shared["turn"],
                "findings": data,
            })
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

        prompt = f"""Analyze this multi-agent debate. Output structured YAML summary.

TOPIC: {topic}
PERSONAS:
{personas_str}

=== CONVERSATION ===
{conv_str}
=== END CONVERSATION ===

Produce a YAML summary with these sections. Be honest — if no consensus emerged,
say so. If nothing new was generated, say so. Do not fabricate agreement.

Output ONLY valid YAML:
```yaml
persona_evolution:
  - name: <agent name>
    opening_stance: <their initial position in 1 sentence>
    final_stance: <how it held up, shifted, or softened>
    key_contribution: <their single most impactful argument>

shared_conclusions:
  - <claim where 2+ agents agreed, or null if none>

key_divergences:
  - claim: <the specific point of disagreement>
    position_a: <agent name: their stance>
    position_b: <agent name: their opposing stance>
    core_reason: <why they could not reconcile — 1 sentence>

novel_insights:
  - <idea that emerged from the debate itself, not from any opening stance>

surprising_moments:
  - moment: <a turn that shifted direction or reframed the question>
    why_significant: <1 sentence>

debate_arc: <how positions evolved from start to finish. Did anyone concede? Was a
  deeper question revealed? 2-3 sentences>

overall_takeaway: <2-3 sentences on what this debate reveals that a single
  perspective would miss>
```"""
        return call_llm(prompt, system=SYSTEM_SUMMARIZER, temperature=0.5, max_tokens=4096)

    def exec_fallback(self, prep_res, exc):
        return "Summary could not be generated due to an error."

    def post(self, shared, prep_res, exec_res):
        shared["summary"] = exec_res
        console.print()
        # Parse YAML if present, otherwise display raw text
        try:
            data = _parse_yaml(exec_res) if isinstance(exec_res, str) else exec_res
            if isinstance(data, dict):
                # Format nicely for display
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
                return
        except Exception:
            pass
        console.print(Panel(str(exec_res), title="Conversation Summary", border_style="bold blue"))


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
