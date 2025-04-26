from .agent import Agent
from .tools import Tool


class System:
    def __init__(self, goal: str, agents: list[Agent], tools: list[Tool]):
        self.goal = goal
        self.time = 0
        self.agents: list[Agent] = agents
        self.tools: list[Tool] = tools

    def tick(self):
        for tool in self.tools:
            tool.update(
                {
                    "time": self.time,
                }
            )

        for agent in self.agents:
            agent.update(
                {
                    "your_team": [a for a in self.agents if a != agent],
                    "team_goal": self.goal,
                    "system_time": self.time,
                }
            )

        self.time += 1
