from pocketflow import Flow
from nodes import InitNode, ModeratorNode, AgentSpeakNode, ResearchNode, SummarizerNode, SaveNode


def create_conversation_flow():
    init = InitNode(wait=5)
    moderator = ModeratorNode(max_retries=2, wait=5)
    agent_speak = AgentSpeakNode(max_retries=2, wait=5)
    research = ResearchNode(max_retries=1, wait=5)
    summarizer = SummarizerNode(wait=5)
    save = SaveNode()

    init >> moderator
    moderator - "speak" >> agent_speak
    agent_speak - "continue" >> moderator
    moderator - "research" >> research
    research - "continue" >> moderator
    moderator - "summarize" >> summarizer
    summarizer >> save

    return Flow(start=init)
