from pocketflow import Flow
from nodes import InitNode, ModeratorNode, AgentSpeakNode, SummarizerNode, SaveNode


def create_conversation_flow():
    init = InitNode()
    moderator = ModeratorNode(max_retries=2)
    agent_speak = AgentSpeakNode(max_retries=2)
    summarizer = SummarizerNode()
    save = SaveNode()

    init >> moderator
    moderator - "speak" >> agent_speak
    agent_speak - "continue" >> moderator
    moderator - "summarize" >> summarizer
    summarizer >> save

    return Flow(start=init)
