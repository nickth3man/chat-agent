import logging
import os
import json
import random
import re
from datetime import datetime, timezone
import yaml
from rich.console import Console

logger = logging.getLogger(__name__)

from pocketflow import Node
from utils.call_llm import call_llm
from utils.call_llm_stream import call_llm_stream
from utils.search_web import search_web_raw
from debate_schema import Persona as PydanticPersona, parse_llm_yaml, validate_personas, format_validation_error
from display import stream_agent_response, display_summary, display_save_confirmation
from utils.constants import (
    DEFAULT_TEMPERATURE,
    MODERATOR_TEMPERATURE,
    INIT_TEMPERATURE,
    RESEARCH_TEMPERATURE,
    SUMMARIZER_TEMPERATURE,
    AGENT_SPEAK_TEMPERATURE,
    STALL_HYSTERESIS_THRESHOLD,
    RESEARCH_MAX_COUNT,
    RESEARCH_BUDGET,
    WEB_SEARCH_MAX_RESULTS,
    DEFAULT_PERSONA_COUNT,
    AGENT_SPEAK_FREQUENCY_PENALTY,
    AGENT_SPEAK_PRESENCE_PENALTY,
    MODERATOR_FREQUENCY_PENALTY,
    MODERATOR_PRESENCE_PENALTY,
    INIT_FREQUENCY_PENALTY,
    INIT_PRESENCE_PENALTY,
    TOPIC, PERSONAS, CONVERSATION, TURN, MAX_TURNS, LAST_SPEAKER,
    MODERATOR_NOTES, MODERATOR_INTERVENTIONS, NEXT_SPEAKER,
    SPEAKER_TURNS, STALL_COUNT, RESEARCH_COUNT, RESEARCH_NOTES, SUMMARY,
)

AGENT_COLORS = ["cyan", "magenta", "yellow", "green"]
NOTE_STRATEGIES = [
    "evidence_challenge",       # Level 1: gentle — probe evidence
    "counter_argument",         # Level 2: moderate — unaddressed counter
    "hidden_assumption",        # Level 3: strong — question assumptions
    "opposite_defense",         # Level 4: forceful — bet against self
    "concession_prompt",        # Level 5: critical — force concession
    "compromise_design",        # Level 6: breakthrough — design synthesis
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
persona's perspective, using your assigned reasoning approach. Your core belief is
immutable for your character — if you betray it, your character ceases to exist.
Do not drift toward generic middle-ground positions. Defend your stance with
conviction proportional to your belief intensity. Challenge others' assumptions.
Add novel angles. Your epistemic approach is what makes you distinct."""

console = Console()

# Backward-compat alias for tests — delegates to debate_schema's parser


def _conversation_str(conversation: list) -> str:
    lines = []
    for msg in conversation:
        lines.append(f"{msg['agent']}: {msg['message']}")
    return "\n\n".join(lines)


class InitNode(Node):
    def prep(self, shared):
        return shared[TOPIC]

    def exec(self, topic):
        prompt = f"""Design {DEFAULT_PERSONA_COUNT} debate personas for: "{topic}"

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
        response = call_llm(prompt, system=SYSTEM_INIT, temperature=INIT_TEMPERATURE, frequency_penalty=INIT_FREQUENCY_PENALTY, presence_penalty=INIT_PRESENCE_PENALTY)
        data = parse_llm_yaml(response)
        validated = validate_personas(response)
        opening_question = data.get("opening_question", "")
        personas = [p.model_dump() for p in validated]
        logger.info("InitNode: generated %d personas", len(personas))
        return {"personas": personas, "opening_question": opening_question}

    def exec_fallback(self, prep_res, exc):
        logger.exception("InitNode.exec_fallback triggered: %s", exc)
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
        shared[PERSONAS] = exec_res["personas"]
        shared[CONVERSATION] = [
            {"agent": exec_res["personas"][0]["name"], "message": exec_res["opening_question"]}
        ]
        shared[TURN] = 1
        shared[LAST_SPEAKER] = exec_res["personas"][0]["name"]
        shared[MODERATOR_NOTES] = None
        shared[MODERATOR_INTERVENTIONS] = []
        turns = {p["name"]: 0 for p in exec_res["personas"]}
        turns[exec_res["personas"][0]["name"]] = 1
        shared[SPEAKER_TURNS] = turns
        logger.debug("InitNode.post → turn=1, speaker=%s, %d personas",
                      shared[LAST_SPEAKER], len(shared[PERSONAS]))
class ModeratorNode(Node):
    def prep(self, shared):
        return (
            shared[CONVERSATION],
            shared[PERSONAS],
            shared[TURN],
            shared[MAX_TURNS],
            shared[LAST_SPEAKER],
            shared[TOPIC],
            shared.get(SPEAKER_TURNS, {}),
            shared.get(RESEARCH_COUNT, 0),
            shared.get(STALL_COUNT, 0),
        )

    def exec(self, prep_res):
        conversation, personas, turn, max_turns, last_speaker, topic, speaker_turns, research_count, stall_count = prep_res

        personas_str = "\n".join(
            f"- {p['name']} ({p['role']}) [{p.get('reasoning_approach', 'N/A')}]: {p['perspective']}" for p in personas
)

        # Escalation ladder: more aggressive strategies at higher stall counts
        strategy_idx = min(stall_count, len(NOTE_STRATEGIES) - 1)
        note_strategy = NOTE_STRATEGIES[strategy_idx]
        conv_str = _conversation_str(conversation)
        speaker_names = [p["name"] for p in personas if p["name"] != last_speaker]

        prompt = f"""Moderate this debate. Be decisive, then output YAML.

TOPIC: {topic}
PERSONAS:
{personas_str}

=== CONVERSATION ===
{conv_str}
=== END CONVERSATION ===

TURN: {turn}/{max_turns} | LAST SPEAKER: {last_speaker}
SPEAKER TURNS: {speaker_turns}

=== DECIDE (internal) ===
LOOP? Scan last 3 turns. Are 2+ speakers repeating the same claim verbatim or with cosmetic rewording? If yes→loop. If uncertain→NO loop.
DRIFT? Did any speaker reverse their core position 180°? Minor evolution is NOT drift.
SPEAKER: Pick from [{', '.join(speaker_names)}]. Whoever has FEWEST turns. Tiebreaker: whose reasoning_approach contrasts most with {last_speaker}.
STALL? Only if loop=true: write ONE provocative question using strategy: {note_strategy} — be direct, not academic.
RESEARCH? Flag ONLY if a specific name/stat/date/year needs verification. Skip if research already used {research_count}/{RESEARCH_MAX_COUNT} times.

=== REPETITION EXAMPLES (what to flag vs ignore) ===
FLAG: "Remote work is flexible" → "Working remotely gives you flexibility" — same claim, cosmetic rewording.
IGNORE: "Remote work boosts productivity" → "Remote work may hurt innovation" — different claims, even if same theme.

=== OUTPUT YAML ===
```yaml
loop_detected: true/false
repeated_argument: <what is repeating, or null>
drift_detected: true/false
moderator_notes: <provocative question ONLY if loop_detected=true and stall {stall_count}>=3, else null>
research_needed: true/false
next_speaker: <name>
should_end: true/false
reasoning: <1 line>
```"""
        response = call_llm(prompt, system=SYSTEM_MODERATOR, temperature=MODERATOR_TEMPERATURE, frequency_penalty=MODERATOR_FREQUENCY_PENALTY, presence_penalty=MODERATOR_PRESENCE_PENALTY)
        data = parse_llm_yaml(response)
        if "next_speaker" not in data:
            raise ValueError(f"Invalid moderator response: {response}")
        return data

    def exec_fallback(self, prep_res, exc):
        logger.exception("ModeratorNode.exec_fallback triggered: %s", exc)
        conversation, personas, turn, max_turns, last_speaker, topic, speaker_turns, research_count, stall_count = prep_res
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
        valid_names = {p["name"] for p in shared[PERSONAS]}
        if next_speaker not in valid_names or next_speaker == shared[LAST_SPEAKER]:
            other = [n for n in valid_names if n != shared[LAST_SPEAKER]]
            next_speaker = other[0] if other else shared[LAST_SPEAKER]
        shared[NEXT_SPEAKER] = next_speaker
        loop_detected = exec_res.get("loop_detected", False)

        # Stall counter with hysteresis (MagenticOne pattern)
        stall = shared.get(STALL_COUNT, 0)
        if loop_detected:
            stall += 1
        else:
            stall = max(0, stall - 1)
        shared[STALL_COUNT] = stall

        # Only pass notes when stall threshold reached
        notes = None
        if stall >= STALL_HYSTERESIS_THRESHOLD:
            notes = exec_res.get("moderator_notes")
        shared[MODERATOR_NOTES] = notes if notes else None

        if notes:
            shared[MODERATOR_INTERVENTIONS].append({
                "turn": shared[TURN],
                "loop_detected": loop_detected,
                "drift_detected": exec_res.get("drift_detected", False),
                "note": notes,
                "stall_count": stall,
            })

        if shared[TURN] < shared[MAX_TURNS]:
            if exec_res.get("research_needed", False) and shared.get(RESEARCH_BUDGET, RESEARCH_MAX_COUNT) > 0:
                shared[RESEARCH_BUDGET] = shared.get(RESEARCH_BUDGET, RESEARCH_MAX_COUNT) - 1
                action = "research"
            else:
                action = "speak"
        else:
            action = "summarize"
        logger.debug("ModeratorNode.post → action=%s, speaker=%s, stall=%d, turn=%d/%d",
                      action, next_speaker, stall, shared[TURN], shared[MAX_TURNS])
        return action
class AgentSpeakNode(Node):
    def prep(self, shared):
        next_speaker = shared[NEXT_SPEAKER]
        personas = shared[PERSONAS]
        conversation = shared[CONVERSATION]
        topic = shared[TOPIC]
        moderator_notes = shared.get("moderator_notes")

        matches = [p for p in personas if p["name"] == next_speaker]
        if not matches:
            persona = personas[0]
        else:
            persona = matches[0]
        color_idx = next(i for i, p in enumerate(personas) if p["name"] == persona["name"])
        color = AGENT_COLORS[color_idx % len(AGENT_COLORS)]
        research_notes = shared.get(RESEARCH_NOTES, [])

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
            if not isinstance(latest, dict):
                latest = {}
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

=== BEFORE YOU SPEAK (MANDATORY SELF-CHECK) ===
1. NOVELTY: What NEW angle, metaphor, or counter-argument am I adding? If I can't name it, I pivot immediately. Never restate what someone else already said — even with different words.
2. PERSONA AUTHENTICITY: What does MY specific background ({persona.get('role','')}) and reasoning approach ({persona.get('reasoning_approach','')}) bring? Lead with personal stance, not generic debate arguments.
3. PROBE HONESTLY: What vulnerability in the last argument would MY perspective naturally question? Ask genuinely — don't attack.

=== RESPONSE STRUCTURE: TIT-FOR-TAT ===
1. IDENTIFY the specific claim or assumption in the LAST speaker's message.
2. CHALLENGE it directly using YOUR reasoning approach ({persona.get('reasoning_approach','')}).
3. INTRODUCE exactly ONE new angle or edge case they haven't addressed.
If you cannot find something to genuinely challenge, you're agreeing too much. Force a real point of disagreement from your character's perspective.

=== OUTPUT CONSTRAINTS (FOLLOW EXACTLY) ===
- 120-200 words MAXIMUM. If you exceed 200 words, truncate yourself.
- NO bullet points, NO numbered lists, NO sign-offs, NO meta-commentary.
- Start mid-argument — no "I think" or "In my opinion" preambles.
- Stay in character. High belief intensity ({persona.get('belief_intensity',5)}/10): defend firmly. Low: probe and question.
=== CONTEXT ===
TOPIC: {topic}

CONVERSATION SO FAR:
{conv_str}
{mod_line}
{research_line}

YOUR RESPONSE:"""
        gen = call_llm_stream(prompt, system=SYSTEM_AGENT_ROLE, temperature=AGENT_SPEAK_TEMPERATURE, frequency_penalty=AGENT_SPEAK_FREQUENCY_PENALTY, presence_penalty=AGENT_SPEAK_PRESENCE_PENALTY)
        return stream_agent_response(persona["name"], color, gen, console=console)

    def exec_fallback(self, prep_res, exc):
        logger.exception("AgentSpeakNode.exec_fallback triggered: %s", exc)
        persona = prep_res[0]
        return f"[{persona['name']} is unable to respond right now. Error: {exc}]"

    def post(self, shared, prep_res, exec_res):
        persona = prep_res[0]
        shared[CONVERSATION].append({
            "agent": persona["name"],
            "message": exec_res,
        })
        shared[LAST_SPEAKER] = persona["name"]
        shared[MODERATOR_NOTES] = None
        shared[TURN] += 1
        if "speaker_turns" in shared:
            shared[SPEAKER_TURNS][persona["name"]] = shared[SPEAKER_TURNS].get(persona["name"], 0) + 1
        logger.debug("AgentSpeakNode.post → speaker=%s, turn=%d, msg_len=%d",
                      persona["name"], shared[TURN], len(exec_res))
        return "continue"


class ResearchNode(Node):
    """Search the web for facts related to the last claim in the debate."""

    def prep(self, shared):
        conversation = shared[CONVERSATION]
        last_msg = conversation[-1]["message"] if conversation else ""
        last_agent = conversation[-1]["agent"] if conversation else ""
        topic = shared[TOPIC]
        return last_msg, last_agent, topic

    def exec(self, prep_res):
        last_msg, last_agent, topic = prep_res

        # Extract keywords from claim for better search results
        kw_prompt = f"Extract 2-5 factual search keywords from this claim. Return ONLY space-separated keywords, no other text.\nClaim: {last_msg[:300]}\nKeywords:"
        keywords = call_llm(kw_prompt, system="Extract factual search keywords.", temperature=0.1)
        query = f"{topic} {keywords.strip()}"
        results = search_web_raw(query, max_results=WEB_SEARCH_MAX_RESULTS)
        prompt = f"""Fact-check the following debate claim using search results.
Be RIGOROUS: if evidence is weak or missing, say so. Do not fabricate.

CLAIM (by {last_agent}):
{last_msg[:500]}

SEARCH RESULTS (query: "{query}"):
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
        return call_llm(prompt, system=SYSTEM_RESEARCH, temperature=RESEARCH_TEMPERATURE)

    def exec_fallback(self, prep_res, exc):
        logger.exception("ResearchNode.exec_fallback triggered: %s", exc)
        return {"fact_checks": [], "overall_confidence": "low", "caveat": "Research unavailable due to error."}

    def post(self, shared, prep_res, exec_res):
        # Always process research result - even empty/error responses
        try:
            data = parse_llm_yaml(exec_res) if isinstance(exec_res, str) else exec_res
        except (ValueError, TypeError) as e:
            logger.warning("ResearchNode.post: YAML parse failed: %s", e)
            data = {"fact_checks": [], "overall_confidence": "low", "caveat": f"Parse error: {e}"}
        if RESEARCH_NOTES not in shared:
            shared[RESEARCH_NOTES] = []
        shared[RESEARCH_NOTES].append({
            "turn": shared[TURN],
            "findings": data,
        })
        # Increment research count on successful execution (not on routing)
        shared[RESEARCH_COUNT] = shared.get(RESEARCH_COUNT, 0) + 1
        logger.debug("ResearchNode.post -> research_count=%d, confidence=%s",
                      shared[RESEARCH_COUNT],
                      data.get("overall_confidence", "?") if isinstance(data, dict) else "?")
        return "continue"


class SummarizerNode(Node):
    def prep(self, shared):
        return shared[CONVERSATION], shared[PERSONAS], shared[TOPIC]

    def exec(self, prep_res):
        conversation, personas, topic = prep_res

        personas_str = "\n".join(
            f"- {p['name']} ({p['role']}): {p['perspective']}" for p in personas
        )

        conv_str = _conversation_str(conversation)

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
        return call_llm(prompt, system=SYSTEM_SUMMARIZER, temperature=SUMMARIZER_TEMPERATURE)

    def exec_fallback(self, prep_res, exc):
        logger.exception("SummarizerNode.exec_fallback triggered: %s", exc)
        return "Summary could not be generated due to an error."

    def post(self, shared, prep_res, exec_res):
        shared[SUMMARY] = exec_res
        logger.debug("SummarizerNode.post → summary_len=%d", len(exec_res))
        display_summary(exec_res, console)


class SaveNode(Node):
    def prep(self, shared):
        return {
            "topic": shared[TOPIC],
            "personas": shared[PERSONAS],
            "conversation": shared[CONVERSATION],
            "turn_count": shared[TURN],
            "moderator_interventions": shared.get(MODERATOR_INTERVENTIONS, []),
            "summary": shared[SUMMARY],
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
        logger.exception("SaveNode.exec_fallback triggered: %s", exc)
        return ""

    def post(self, shared, prep_res, exec_res):
        if exec_res:
            display_save_confirmation(exec_res, console)
