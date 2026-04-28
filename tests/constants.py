"""Shared test constants — canned YAML responses for LLM mocks."""

INIT_YAML_RESPONSE = """
```yaml
personas:
  - name: Alex
    role: Productivity Expert
    perspective: Remote work boosts productivity through focus and flexibility.
    reasoning_approach: Evidence-first
    belief_intensity: 8
    communication_style: Cites studies and data
    argument_tendency: Compares metrics side by side
  - name: Jordan
    role: Organizational Psychologist
    perspective: Isolation from remote work damages team cohesion.
    reasoning_approach: Principle-first
    belief_intensity: 7
    communication_style: Uses frameworks
    argument_tendency: Asks about human cost
  - name: Sam
    role: Small Business Owner
    perspective: Hybrid work fails both ways.
    reasoning_approach: Counterfactual
    belief_intensity: 5
    communication_style: Practical examples
    argument_tendency: Tests against reality
  - name: Riley
    role: Futurist
    perspective: Redesign work entirely.
    reasoning_approach: Systems-thinking
    belief_intensity: 6
    communication_style: Uses metaphor
    argument_tendency: Finds hidden assumptions
opening_question: Is remote work the future of work or a temporary experiment?
```
"""

MODERATOR_YAML_RESPONSE = """
```yaml
loop_detected: false
repeated_argument: null
drift_detected: false
moderator_notes: null
research_needed: false
next_speaker: Jordan
should_end: false
reasoning: Jordan has fewest turns
```
"""

SUMMARIZER_YAML_RESPONSE = """
```yaml
persona_evolution:
  - name: Alex
    opening_stance: Pro remote work
    final_stance: Still pro but acknowledged challenges
    key_contribution: Productivity data
shared_conclusions:
  - Remote work needs better tooling
key_divergences:
  - claim: Remote work hurts innovation
    position_a: Alex: Data shows otherwise
    position_b: Jordan: Psychological studies disagree
    core_reason: Different definitions of innovation
novel_insights:
  - The debate revealed that flexibility matters more than location
surprising_moments:
  - moment: Alex conceded on mental health impact
    why_significant: Shifted the debate toward hybrid solutions
debate_arc: Started polarized but converged on nuanced hybrid approach.
overall_takeaway: Pure remote and pure office are both suboptimal.
```
"""
