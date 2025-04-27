import pydantic

from .llm import LLMClient


class SummarizeOutput(pydantic.BaseModel):
    summary: str


class Memory:
    def __init__(self, agent, llm: LLMClient):
        self.agent = agent
        self.llm: LLMClient = llm

        self.history_log = []
        self.agent_memory = {}
        self.agent_summaries = {}

    def add_to_history(self, timestamp, message: str):
        self.history_log.append((timestamp, message))

    def add_agent_memory(self, agent, message: str):
        if agent not in self.agent_memory:
            self.agent_memory[agent] = []
        self.agent_memory[agent].append(message)

    def summarize_memories(self, timestamp):
        for agent, messages in self.agent_memory.items():
            self.summarize_agent_memory(timestamp, agent, messages)

    def summarize_agent_memory(self, timestamp, agent, messages: list[str]):
        prompt = f"""
You are {self.agent}, a {self.agent.role} in a world of other agents.
You are currently reflecting on your past interactions with {agent}.
You received the following messages from {agent}:
{messages}

Summarize your interactions with {agent} holistically, focusing only on the most important points and insights.
You can capture personal details and characteristics of the agent if you want or focus on relevant information about their role and actions.
Keep your sentences short and concise.
Generate at most 5 sentences.

For example:
[
    "<Name> as <role> acts <characteristic>.",
    "<Name> treats me <characteristic>.",
    "<Name> appears to be <characteristic>.",
]
"""

        response = self.llm.generate(prompt, format=SummarizeOutput)
        summary = response.summary
        self.agent_summaries[agent] = summary
        self.history_log.append((timestamp, f"reflecting on {agent}"))
